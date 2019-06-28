# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     multi_domain_share
   Description :
   Author :       xmz
   date：          2019/6/17
-------------------------------------------------
"""
from typing import Dict, Optional, List

import numpy as np
from allennlp.modules.attention import DotProductAttention, BilinearAttention
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.nn.util import get_device_of, get_final_encoder_states, sequence_cross_entropy_with_logits, \
    get_text_field_mask
from allennlp.training.util import get_batch_size
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder, InputVariationalDropout, \
    Embedding, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from torch import nn
from torch.autograd import Function, Variable
from torch.nn import Dropout, LayerNorm

from mtl.tasks.task import Task
from mtl.common.logger import logger
from train_transfer import TASKS_NAME


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
        self._classifier = FeedForward(source_size, 1, target_size, Activation.by_name("elu")())

    def forward(self, representation, reverse=torch.tensor(False), lambd=1.0):
        if reverse.all():
            # TODO increase lambda from 0
            representation = grad_reverse(representation, lambd)
        return self._classifier(representation)


class RNNEncoder(Model):
    def __init__(self,
                 vocab,
                 shared_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None) -> None:
        super(RNNEncoder, self).__init__(vocab, regularizer)
        self._shared_encoder = shared_encoder
        self._input_dropout = Dropout(input_dropout)

    @overrides
    def forward(self, embedded_text_input: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedded_text_input = self._input_dropout(embedded_text_input)
        shared_encoded_text = self._shared_encoder(embedded_text_input, mask)
        # shared_encoded_text = get_final_encoder_states(shared_encoded_text, tokens_mask, bidirectional=True)

        return shared_encoded_text

    def get_output_dim(self):
        return self._shared_encoder.get_output_dim()


@Model.register("stmcls")
class SentimentClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 label_smoothing: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        share_rnn = nn.LSTM(input_size=self._text_field_embedder.get_output_dim(),
                            hidden_size=150,
                            batch_first=True,
                            # dropout=dropout,
                            bidirectional=True)
        share_encoder = PytorchSeq2SeqWrapper(share_rnn)

        self._encoder = RNNEncoder(vocab, share_encoder, input_dropout, regularizer)
        self._seq_vec = CnnEncoder(self._encoder.get_output_dim(), 25)
        self._de_dim = len(TASKS_NAME)
        weight = torch.empty(self._de_dim, self._text_field_embedder.get_output_dim())
        torch.nn.init.orthogonal_(weight)
        self._domain_embeddings = Embedding(self._de_dim, self._text_field_embedder.get_output_dim(), weight=weight)
        self._de_attention = BilinearAttention(self._seq_vec.get_output_dim(),
                                               self._domain_embeddings.get_output_dim())
        self._de_feedforward = FeedForward(self._domain_embeddings.get_output_dim(), 1,
                                           self._seq_vec.get_output_dim(), Activation.by_name("elu")())

        self._num_classes = self.vocab.get_vocab_size("label")
        self._sentiment_discriminator = Discriminator(self._seq_vec.get_output_dim(), self._num_classes)
        self._s_domain_discriminator = Discriminator(self._seq_vec.get_output_dim(), len(TASKS_NAME))
        self._valid_discriminator = Discriminator(self._domain_embeddings.get_output_dim(), 2)
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._label_smoothing = label_smoothing

        self.metrics = {
            "s_domain_acc": CategoricalAccuracy(),
            "valid_acc": CategoricalAccuracy()
        }
        for task_name in TASKS_NAME:
            self.metrics["{}_stm_acc".format(task_name)] = CategoricalAccuracy()

        self._loss = torch.nn.CrossEntropyLoss()
        self._domain_loss = torch.nn.CrossEntropyLoss()
        # TODO torch.nn.BCELoss
        self._valid_loss = torch.nn.BCEWithLogitsLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                task_index: torch.IntTensor,
                reverse: torch.ByteTensor,
                for_training: torch.ByteTensor,
                train_stage: torch.IntTensor,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param task_index:
        :param reverse:
        :param for_training:
        :param train_stage: ["share_senti", "share_classify",
        "share_classify_adversarial", "domain_valid", "domain_valid_adversarial"]
        :param tokens:
        :param label:
        :return:
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        embed_tokens = self._encoder(embedded_text, mask)
        batch_size = get_batch_size(embed_tokens)
        # bs * (25*4)
        seq_vec = self._seq_vec(embed_tokens, mask)
        # TODO add linear layer

        domain_embeddings = self._domain_embeddings(torch.arange(self._de_dim).cuda())

        de_scores = F.softmax(
            self._de_attention(seq_vec, domain_embeddings.expand(batch_size, *domain_embeddings.size())), dim=1)
        de_valid = False
        if np.random.rand() < 0.3:
            de_valid = True
            noise = 0.01 * torch.normal(mean=0.5,
                                        # std=torch.std(domain_embeddings).sign_())
                                        std=torch.empty(*de_scores.size()).fill_(1.0))
            de_scores = de_scores + noise.cuda()
        domain_embedding = torch.matmul(de_scores, domain_embeddings)
        domain_embedding = self._de_feedforward(domain_embedding)
        # train sentiment classify
        if train_stage.cpu() == torch.tensor(0) or not for_training:

            de_representation = torch.tanh(torch.add(domain_embedding, seq_vec))

            sentiment_logits = self._sentiment_discriminator(de_representation)
            if label is not None:
                loss = self._loss(sentiment_logits, label)
                self.metrics["{}_stm_acc".format(TASKS_NAME[task_index.cpu()])](sentiment_logits, label)

        if train_stage.cpu() == torch.tensor(1) or not for_training:
            s_domain_logits = self._s_domain_discriminator(seq_vec, reverse=reverse)
            task_index = task_index.expand(batch_size)
            loss = self._domain_loss(s_domain_logits, task_index)
            self.metrics["s_domain_acc"](s_domain_logits, task_index)

        if train_stage.cpu() == torch.tensor(2) or not for_training:
            valid_logits = self._valid_discriminator(domain_embedding, reverse=reverse)
            valid_label = torch.ones(batch_size).cuda()
            if de_valid:
                valid_label = torch.zeros(batch_size).cuda()
            if self._label_smoothing is not None and self._label_smoothing > 0.0:
                loss = sequence_cross_entropy_with_logits(valid_logits,
                                                          valid_label.unsqueeze(0).cuda(),
                                                          torch.tensor(1).unsqueeze(0).cuda(),
                                                          average="token",
                                                          label_smoothing=self._label_smoothing)
            else:
                loss = self._valid_loss(valid_logits,
                                        torch.zeros(2).scatter_(0, valid_label, torch.tensor(1.0)).cuda())
            self.metrics["valid_acc"](valid_logits, valid_label)
        # TODO add orthogonal loss
        output_dict = {"loss": loss}

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
    def get_metrics(self, task_name: str, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items() if
                (task_name or "s_domain_acc" or "valid_acc") in metric_name}
