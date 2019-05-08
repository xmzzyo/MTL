# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sentiment_classifier_mtl
   Description :
   Author :       xmz
   date：          19-4-14
-------------------------------------------------
"""
import copy
from typing import Dict

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import get_device_of, move_to_device
from overrides import overrides

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Embedding, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from torch import nn

from mtl.common.logger import logger
from mtl.models.sentiment_classifier_base import SentimentClassifier, Discriminator
from train_stmcls import TASKS_NAME


@Model.register("joint_stmcls")
class JointSentimentClassifier(Model):
    """
    Parameters
    ----------
    vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
    params: ``allennlp.common.Params``, required
        Configuration parameters for the multi-tasks model.
    regularizer: ``allennlp.nn.RegularizerApplicator``, optional (default = None)
        A reguralizer to apply to the model's layers.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 share_encoder: Seq2VecEncoder = None,
                 private_encoder: Seq2VecEncoder = None,
                 dropout: float = None,
                 input_dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super(JointSentimentClassifier, self).__init__(vocab=vocab, regularizer=regularizer)

        self._text_field_embedder = text_field_embedder
        self._domain_embeddings = Embedding(len(TASKS_NAME), 50)
        if share_encoder is None and private_encoder is None:
            share_rnn = nn.LSTM(input_size=self._text_field_embedder.get_output_dim(),
                                hidden_size=150,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)
            share_encoder = PytorchSeq2SeqWrapper(share_rnn)
            private_rnn = nn.LSTM(input_size=self._text_field_embedder.get_output_dim(),
                                  hidden_size=150,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=True)
            private_encoder = PytorchSeq2SeqWrapper(private_rnn)
            logger.info("Using LSTM as encoder")
            self._domain_embeddings = Embedding(len(TASKS_NAME), self._text_field_embedder.get_output_dim())
        self._share_encoder = share_encoder

        self._s_domain_discriminator = Discriminator(share_encoder.get_output_dim(), len(TASKS_NAME))

        self._p_domain_discriminator = Discriminator(private_encoder.get_output_dim(), len(TASKS_NAME))

        self._valid_discriminator = Discriminator(self._domain_embeddings.get_output_dim(), 2)

        for task in TASKS_NAME:
            tagger = SentimentClassifier(
                vocab=vocab,
                text_field_embedder=self._text_field_embedder,
                share_encoder=self._share_encoder,
                private_encoder=copy.deepcopy(private_encoder),
                domain_embeddings=self._domain_embeddings,
                s_domain_discriminator=self._s_domain_discriminator,
                p_domain_discriminator=self._p_domain_discriminator,
                valid_discriminator=self._valid_discriminator,
                dropout=dropout,
                input_dropout=input_dropout,
                label_smoothing=0.1,
                initializer=initializer
            )
            self.add_module("_tagger_{}".format(task), tagger)

        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides
    def forward(self, tensor_batch, task_name: str, reverse=False, for_training=False) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        task_tagger = getattr(self, "_tagger_%s" % task_name)
        task_index = TASKS_NAME.index(task_name)
        tensor_batch['task_index'] = torch.tensor(task_index)
        tensor_batch["reverse"] = torch.tensor(reverse)
        tensor_batch['for_training'] = torch.tensor(for_training)
        tensor_batch = move_to_device(tensor_batch, 0)
        return task_tagger.forward(**tensor_batch)

    @overrides
    def get_metrics(self, task_name: str, reset: bool = False) -> Dict[str, float]:
        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    # @classmethod
    # def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> Model:
    #     return cls(vocab=vocab, params=params, regularizer=regularizer)
