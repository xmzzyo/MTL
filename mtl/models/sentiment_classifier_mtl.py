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

from allennlp.nn.util import get_device_of
from overrides import overrides

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Embedding, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator

from mtl.common.logger import logger
from mtl.models.sentiment_classifier_base import SentimentClassifier, Encoder, Discriminator
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
                 share_encoder: Seq2VecEncoder,
                 private_encoder: Seq2VecEncoder,
                 dropout: float = None,
                 input_dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super(JointSentimentClassifier, self).__init__(vocab=vocab, regularizer=regularizer)

        self._domain_embeddings = Embedding(len(TASKS_NAME), 50)
        self._encoder = Encoder(vocab, text_field_embedder, share_encoder, private_encoder,
                                domain_embeddings=self._domain_embeddings, input_dropout=input_dropout)

        self._s_domain_discriminator = Discriminator(share_encoder.get_output_dim(), len(TASKS_NAME))

        self._p_domain_discriminator = Discriminator(private_encoder.get_output_dim(), len(TASKS_NAME))

        for task in TASKS_NAME:
            tagger = SentimentClassifier(
                vocab=vocab,
                encoder=self._encoder,
                s_domain_discriminator=self._s_domain_discriminator,
                p_domain_discriminator=self._p_domain_discriminator,
                dropout=dropout,
                input_dropout=input_dropout,
                initializer=initializer
            )
            self.add_module("_tagger_{}".format(task), tagger)

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

    # @classmethod
    # def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> Model:
    #     return cls(vocab=vocab, params=params, regularizer=regularizer)
