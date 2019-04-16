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
import logging
from typing import Dict

from allennlp.nn.util import get_device_of
from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Embedding, Seq2VecEncoder, TokenEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from mtl.models.sentiment_classifier_base import SentimentClassifier
from train_stmcls import TASKS_NAME

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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

    def __init__(self, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator = None):
        super(JointSentimentClassifier, self).__init__(vocab=vocab, regularizer=regularizer)

        # Base text Field Embedder
        # TODO with no grad
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder

        seq2vec_encoder_params = params.pop("seq2vec_encoder")
        seq2vec_encoder = Seq2VecEncoder.from_params(seq2vec_encoder_params)

        self.domain_embeddings = Embedding(len(TASKS_NAME), self._text_field_embedder.get_output_dim())
        # self.domain_embedding = TokenEmbedder.from_params(vocab=vocab, params=params.pop("domain_embeddings"))

        # Encoder
        encoder_params = params.pop("encoder")
        encoder = Seq2SeqEncoder.from_params(encoder_params)

        self._shared_encoder = encoder

        self._dropout = params.pop('dropout')
        self._input_dropout = params.pop('input_dropout')
        init_params = params.pop("initializer", None)
        self._initializer = (
            InitializerApplicator.from_params(init_params) if init_params is not None else InitializerApplicator()
        )

        tasks = TASKS_NAME
        for task in tasks:
            tagger = SentimentClassifier(
                vocab=vocab,
                text_field_embedder=self._text_field_embedder,
                seq2vec_encoder=copy.deepcopy(seq2vec_encoder),
                shared_encoder=self._shared_encoder,
                private_encoder=copy.deepcopy(encoder),
                with_domain_embedding=True,
                domain_embeddings=self.domain_embeddings,
                dropout=self._dropout,
                input_dropout=self._input_dropout,
                initializer=self._initializer
            )
            setattr(self, "_tagger_%s" % task, tagger)

        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides
    def forward(self, tensor_batch, task_name: str, for_training: bool = False) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        task_tagger = getattr(self, "_tagger_%s" % task_name)
        task_index = TASKS_NAME.index(task_name)
        tensor_batch['task_index'] = torch.tensor(task_index).cuda()
        return task_tagger.forward(**tensor_batch)

    @overrides
    def get_metrics(self, task_name: str, reset: bool = False) -> Dict[str, float]:
        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> Model:
        return cls(vocab=vocab, params=params, regularizer=regularizer)
