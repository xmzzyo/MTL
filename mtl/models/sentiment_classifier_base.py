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

import numpy
from allennlp.nn.util import get_device_of
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
from torch.nn import Dropout


class Encoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 shared_encoder: Seq2VecEncoder,
                 private_encoder: Seq2VecEncoder,
                 with_domain_embedding: bool = True,
                 domain_embeddings: Embedding = None,
                 input_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None) -> None:
        super(Encoder, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._shared_encoder = shared_encoder
        self._private_encoder = private_encoder
        self._domain_embeddings = domain_embeddings
        self._with_domain_embedding = with_domain_embedding
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
        output_dict["share_embedding"] = shared_encoded_text
        private_encoded_text = self._private_encoder(embedded_text_input, tokens_mask)
        output_dict["private_embedding"] = private_encoded_text

        domain_embedding = domain_embedding.expand(batch_size, -1)
        embedded_text = torch.cat([domain_embedding, shared_encoded_text, private_encoded_text], -1)
        output_dict["embedded_text"] = embedded_text
        return output_dict

    def get_output_dim(self):
        return self._domain_embeddings.get_output_dim() + self._shared_encoder.get_output_dim() \
               + self._private_encoder.get_output_dim()


class Discriminator(nn.Module):
    def __init__(self, source_size, target_size):
        super(Discriminator, self).__init__()
        self._classifier = nn.Linear(source_size, target_size)

    def forward(self, representation):
        return self._classifier(representation)


class SentimentClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Model,
                 s_domain_discriminator: Discriminator,
                 p_domain_discriminator: Discriminator,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 label_smoothing: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        self._encoder = encoder

        self._num_classes = self.vocab.get_vocab_size("label")
        self._sentiment_discriminator = Discriminator(self._encoder.get_output_dim(), self._num_classes)
        self._s_domain_discriminator = s_domain_discriminator
        self._p_domain_discriminator = p_domain_discriminator
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }

        self._loss = torch.nn.CrossEntropyLoss()
        self._domain_loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                task_index: torch.IntTensor,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        embeddeds = self._encoder(task_index, tokens)
        batch_size = get_batch_size(embeddeds["embedded_text"])

        sentiment_logits = self._sentiment_discriminator(embeddeds["embedded_text"])

        p_domain_logits = self._p_domain_discriminator(embeddeds["private_embedding"])

        s_domain_logits = self._s_domain_discriminator(embeddeds["share_embedding"])

        # domain_logits = self._domain_discriminator(embedded_text)
        output_dict = {'logits': sentiment_logits}
        if label is not None:
            loss = self._loss(sentiment_logits, label)
            for metric in self.metrics.values():
                metric(sentiment_logits, label)
            output_dict["sentiment_loss"] = loss
            # task_index = task_index.unsqueeze(0)
            task_index = task_index.expand(batch_size)
            # print(p_domain_logits.shape, task_index, task_index.shape)
            p_domain_loss = self._domain_loss(p_domain_logits, task_index)
            output_dict["p_domain_loss"] = p_domain_loss
            s_domain_loss = self._domain_loss(s_domain_logits, task_index)
            output_dict["s_domain_loss"] = s_domain_loss
            output_dict["loss"] = (loss + p_domain_loss) / 2

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
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="label")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
