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
from torch.nn import Dropout


class SentimentClassifier(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the sentence to a vector.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 shared_encoder: Seq2SeqEncoder = None,
                 private_encoder: Seq2SeqEncoder = None,
                 with_domain_embedding: bool = True,
                 domain_embeddings: Embedding = None,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 label_smoothing: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.shared_encoder = shared_encoder
        self.private_encoder = private_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._domain_embeddings = domain_embeddings
        self._with_domain_embedding = with_domain_embedding
        self.shared_encoder_dim = shared_encoder.get_output_dim()
        self.private_encoder_dim = private_encoder.get_output_dim()
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()
        self.num_classes = self.vocab.get_vocab_size("label")
        self.classifier_feedforward = torch.nn.Linear(self._classifier_input_dim, self.num_classes)
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                task_index: int,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        task_index : task index
        tokens : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)
        batch_size = get_batch_size(tokens)
        domain_embedding = self._domain_embeddings(task_index)
        if self._with_domain_embedding:
            domain_embedding = domain_embedding.expand(batch_size, 1, self.text_field_embedder.get_output_dim())
            # Concatenate the domain embedding onto the sentence representation.
            embedded_text_input = torch.cat((domain_embedding, embedded_text_input), 1)
            tokens_mask = torch.cat([tokens_mask.new_ones(batch_size, 1), tokens_mask], 1)
            embedded_text_input = self._input_dropout(embedded_text_input)
        shared_encoded_text = self.shared_encoder(embedded_text_input, tokens_mask)
        # TODO compare whether to add domain embedding into private representation
        private_encoded_text = self.private_encoder(embedded_text_input, tokens_mask)

        embedded_text = self._seq2vec_encoder(torch.cat([shared_encoded_text, private_encoded_text], -1),
                                              mask=tokens_mask)
        logits = self.classifier_feedforward(embedded_text)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label.long().view(-1))
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

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
