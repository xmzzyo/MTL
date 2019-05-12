# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sentiment_classifier
   Description :
   Author :       xmz
   date：          19-4-10
-------------------------------------------------
"""
from typing import Dict, Optional, List

import numpy as np
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_device_of, get_final_encoder_states, sequence_cross_entropy_with_logits
from allennlp.training.util import get_batch_size
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder, InputVariationalDropout, \
    Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from torch import nn
from torch.autograd import Function, Variable
from torch.nn import Dropout, LayerNorm

from mtl.common.logger import logger
from train_stmcls import TASKS_NAME


class CNNEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 shared_encoder: Seq2VecEncoder,
                 private_encoder: Seq2VecEncoder,
                 with_domain_embedding: bool = True,
                 domain_embeddings: Embedding = None,
                 input_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None) -> None:
        super(CNNEncoder, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._shared_encoder = shared_encoder
        self._private_encoder = private_encoder
        self._domain_embeddings = domain_embeddings
        # self._U = nn.Linear()
        self._with_domain_embedding = with_domain_embedding
        self._attention = DotProductAttention()
        self._input_dropout = Dropout(input_dropout)

    def forward(self,
                task_index: torch.IntTensor,
                tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        embedded_text_input = self._text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)
        batch_size = get_batch_size(tokens)
        domain_embedding = self._domain_embeddings(task_index)
        output_dict = {"domain_embedding": domain_embedding}
        embedded_text_input = self._input_dropout(embedded_text_input)

        shared_encoded_text = self._shared_encoder(embedded_text_input, tokens_mask)
        private_encoded_text = self._private_encoder(embedded_text_input, tokens_mask)

        if self._with_domain_embedding:
            domain_embedding = domain_embedding.expand(batch_size, -1)
            shared_encoded_text = torch.cat([domain_embedding, shared_encoded_text], -1)
            private_encoded_text = torch.cat([domain_embedding, shared_encoded_text], -1)

            # shared_encoded_text = shared_encoded_text.view(batch_size, 4, -1)
            # scores = F.softmax(torch.matmul(shared_encoded_text, domain_embedding), -1)
            # shared_encoded_text = torch.matmul(shared_encoded_text.transpose(1, 2), scores.unsqueeze(2))
            # shared_encoded_text = shared_encoded_text.view(batch_size, -1)
            # private_encoded_text = private_encoded_text.view(batch_size, 4, -1)
            # scores = F.softmax(torch.matmul(private_encoded_text, domain_embedding), -1)
            # private_encoded_text = torch.matmul(private_encoded_text.transpose(1, 2), scores.unsqueeze(2))
            # private_encoded_text = private_encoded_text.view(batch_size, -1)

            # domain_embedding = domain_embedding.expand(batch_size, -1)
            # shared_encoded_text = torch.cat([domain_embedding, shared_encoded_text], -1)
            # private_encoded_text = torch.cat([domain_embedding, private_encoded_text], -1)
        output_dict["share_embedding"] = shared_encoded_text
        output_dict["private_embedding"] = private_encoded_text

        # embedded_text = torch.cat([domain_embedding, shared_encoded_text, private_encoded_text], -1)
        embedded_text = torch.cat([shared_encoded_text, private_encoded_text], -1)
        output_dict["embedded_text"] = embedded_text
        return output_dict

    def get_output_dim(self):
        return 2 * self._domain_embeddings.get_output_dim() + self._shared_encoder.get_output_dim() \
               + self._private_encoder.get_output_dim()


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return self.lambd * grad_output.neg()


def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class Discriminator(nn.Module):
    def __init__(self, source_size, target_size):
        super(Discriminator, self).__init__()
        self._classifier = nn.Linear(int(source_size), target_size)

    def forward(self, representation, epoch_trained=None, reverse=torch.tensor(False), lambd=1.0):
        if reverse.all():
            # TODO increase lambda from 0
            representation = grad_reverse(representation, lambd)
        return self._classifier(representation)


class RNNEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 shared_encoder: Seq2SeqEncoder,
                 private_encoder: Seq2SeqEncoder,
                 with_domain_embedding: bool = True,
                 domain_embeddings: Embedding = None,
                 input_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None) -> None:
        super(RNNEncoder, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._shared_encoder = shared_encoder
        self._private_encoder = private_encoder
        self._domain_embeddings = domain_embeddings
        self._with_domain_embedding = with_domain_embedding
        self._seq2vec = BagOfEmbeddingsEncoder(embedding_dim=self.get_output_dim())
        self._input_dropout = Dropout(input_dropout)

    @overrides
    def forward(self,
                task_index: torch.IntTensor,
                tokens: Dict[str, torch.LongTensor],
                epoch_trained: torch.IntTensor,
                valid_discriminator: Discriminator,
                reverse: torch.ByteTensor,
                for_training: torch.ByteTensor) -> Dict[str, torch.Tensor]:
        embedded_text_input = self._text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)
        batch_size = get_batch_size(tokens)
        # TODO
        if np.random.rand() < 0.3 and for_training.all():
            logger.info("Domain Embedding with Perturbation")
            domain_embeddings = self._domain_embeddings(torch.arange(0, len(TASKS_NAME)).cuda())
            domain_embedding = get_perturbation_domain_embedding(domain_embeddings, task_index, epoch_trained)
            # domain_embedding = FGSM(self._domain_embeddings, task_index, valid_discriminator)
            output_dict = {"valid": torch.tensor(0)}
        else:
            logger.info("Domain Embedding without Perturbation")
            domain_embedding = self._domain_embeddings(task_index)
            output_dict = {"valid": torch.tensor(1)}

        output_dict["domain_embedding"] = domain_embedding
        embedded_text_input = self._input_dropout(embedded_text_input)
        if self._with_domain_embedding:
            domain_embedding = domain_embedding.expand(batch_size, 1, -1)
            embedded_text_input = torch.cat((domain_embedding, embedded_text_input), 1)
            tokens_mask = torch.cat([tokens_mask.new_ones(batch_size, 1), tokens_mask], 1)

        shared_encoded_text = self._shared_encoder(embedded_text_input, tokens_mask)
        # shared_encoded_text = self._seq2vec(shared_encoded_text, tokens_mask)
        shared_encoded_text = get_final_encoder_states(shared_encoded_text, tokens_mask, bidirectional=True)
        output_dict["share_embedding"] = shared_encoded_text

        private_encoded_text = self._private_encoder(embedded_text_input, tokens_mask)
        # private_encoded_text = self._seq2vec(private_encoded_text)
        private_encoded_text = get_final_encoder_states(private_encoded_text, tokens_mask, bidirectional=True)
        output_dict["private_embedding"] = private_encoded_text

        embedded_text = torch.cat([shared_encoded_text, private_encoded_text], -1)
        output_dict["embedded_text"] = embedded_text
        return output_dict

    def get_output_dim(self):
        return self._shared_encoder.get_output_dim() + self._private_encoder.get_output_dim()


def get_perturbation_domain_embedding(domain_embeddings, index, epoch_trained):
    epoch_trained = epoch_trained + 1 if epoch_trained is not None else 10
    u, s, v = torch.svd(domain_embeddings)
    noise = 0.01 * torch.normal(mean=0.5,
                                # std=torch.std(domain_embeddings).sign_())
                                std=torch.tensor([1.0 for _ in range(domain_embeddings.shape[0])]))
    # TODO remove
    noise[:2] = 0.0
    s += noise.cuda()
    reconstruct = torch.mm(torch.mm(u, torch.diag(s)), v.t())
    # logger.info("{} Embedding's mean is {}, std is {}", TASKS_NAME[index], np.mean(domain_embeddings[index]),
    #             np.std(domain_embeddings[index]))
    # logger.info("{} Embedding's perturbation is {}", TASKS_NAME[index],
    #             np.mean(np.subtract(domain_embeddings[index], reconstruct[index])))
    perturbation_domain_embedding = Embedding(len(TASKS_NAME), 100, weight=reconstruct)
    return perturbation_domain_embedding(index)


def FGSM(domain_embeddings, index, valid_discriminator):
    domain_embedding = domain_embeddings(index)
    domain_embedding = torch.tensor(domain_embedding.detach(), requires_grad=True)
    valid_logits = valid_discriminator(domain_embedding)
    criterion = torch.nn.BCEWithLogitsLoss()
    valid_label = torch.tensor(0)
    loss = criterion(valid_logits, torch.zeros(2).scatter_(0, valid_label, torch.tensor(1.0)).cuda())
    if domain_embedding.grad is not None:
        domain_embedding.grad.data.fill_(0)
    loss.backward()
    epsilon = 0.03
    domain_embedding.grad.data.sign_()
    domain_embedding = domain_embedding - epsilon * domain_embedding.grad
    # domain_embedding = np.clip(domain_embedding, 0, 1)
    # domain_embedding = torch.clamp(domain_embedding, -1, 1)
    return domain_embedding


class SentimentClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 share_encoder: _EncoderBase,
                 private_encoder: _EncoderBase,
                 domain_embeddings: Embedding,
                 s_domain_discriminator: Discriminator,
                 p_domain_discriminator: Discriminator,
                 valid_discriminator: Discriminator,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 label_smoothing: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        if isinstance(share_encoder, Seq2VecEncoder) and isinstance(private_encoder, Seq2VecEncoder):
            self._encoder = CNNEncoder(vocab, text_field_embedder, share_encoder, private_encoder,
                                       with_domain_embedding=True,
                                       domain_embeddings=domain_embeddings, input_dropout=input_dropout)
        else:
            self._encoder = RNNEncoder(vocab, text_field_embedder, share_encoder, private_encoder,
                                       with_domain_embedding=True,
                                       domain_embeddings=domain_embeddings, input_dropout=input_dropout)

        self._num_classes = self.vocab.get_vocab_size("label")
        self._sentiment_discriminator = Discriminator(self._encoder.get_output_dim(), self._num_classes)
        self._s_domain_discriminator = s_domain_discriminator
        self._p_domain_discriminator = p_domain_discriminator
        self._valid_discriminator = valid_discriminator
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._label_smoothing = label_smoothing

        self.metrics = {
            "sentiment_acc": CategoricalAccuracy(),
            "p_domain_acc": CategoricalAccuracy(),
            "s_domain_acc": CategoricalAccuracy(),
            "valid_acc": CategoricalAccuracy()
        }

        self._loss = torch.nn.CrossEntropyLoss()
        self._domain_loss = torch.nn.CrossEntropyLoss()
        self._valid_loss = torch.nn.BCEWithLogitsLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                task_index: torch.IntTensor,
                reverse: torch.ByteTensor,
                epoch_trained: torch.IntTensor,
                for_training: torch.ByteTensor,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        embeddeds = self._encoder(task_index, tokens, epoch_trained, self._valid_discriminator, reverse, for_training)
        batch_size = get_batch_size(embeddeds["embedded_text"])

        sentiment_logits = self._sentiment_discriminator(embeddeds["embedded_text"])

        p_domain_logits = self._p_domain_discriminator(embeddeds["private_embedding"])

        # TODO set reverse = true
        s_domain_logits = self._s_domain_discriminator(embeddeds["share_embedding"], reverse=reverse)
        # TODO set reverse = true
        # TODO use share_embedding instead of domain_embedding
        valid_logits = self._valid_discriminator(embeddeds["domain_embedding"], reverse=reverse)

        valid_label = embeddeds['valid']

        logits = [sentiment_logits, p_domain_logits, s_domain_logits, valid_logits]

        # domain_logits = self._domain_discriminator(embedded_text)
        output_dict = {'logits': sentiment_logits}
        if label is not None:
            loss = self._loss(sentiment_logits, label)
            # task_index = task_index.unsqueeze(0)
            task_index = task_index.expand(batch_size)
            targets = [label, task_index, task_index, valid_label]
            # print(p_domain_logits.shape, task_index, task_index.shape)
            p_domain_loss = self._domain_loss(p_domain_logits, task_index)
            s_domain_loss = self._domain_loss(s_domain_logits, task_index)
            logger.info("Share domain logits standard variation is {}",
                        np.mean(np.std(s_domain_logits.detach().cpu().numpy(), axis=0)))
            if self._label_smoothing is not None and self._label_smoothing > 0.0:
                valid_loss = sequence_cross_entropy_with_logits(valid_logits,
                                                                valid_label.unsqueeze(0).cuda(),
                                                                torch.tensor(1).unsqueeze(0).cuda(),
                                                                average="token",
                                                                label_smoothing=self._label_smoothing)
            else:
                valid_loss = self._valid_loss(valid_logits,
                                              torch.zeros(2).scatter_(0, valid_label, torch.tensor(1.0)).cuda())
            output_dict['stm_loss'] = loss
            output_dict['p_d_loss'] = p_domain_loss
            output_dict['s_d_loss'] = s_domain_loss
            output_dict['valid_loss'] = valid_loss
            # TODO add share domain logits std loss
            output_dict['loss'] = loss + p_domain_loss + 0.005 * s_domain_loss + 0.005 * valid_loss
            # + torch.mean(torch.std(s_domain_logits, dim=1))
            # output_dict['loss'] = loss + p_domain_loss + 0.005 * s_domain_loss

            for (metric, logit, target) in zip(self.metrics.values(), logits, targets):
                metric(logit, target)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="label")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
