# coding: utf-8

import logging
from typing import Dict, List

from allennlp.models import SpanConstituencyParser
from allennlp.modules.span_extractors import SpanExtractor
from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Embedding, FeedForward
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from mtl.models import BiaffineParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'PUN', 'SYM'}


@Model.register("joint_dsp_csp_srl")
class JointDCS(Model):
    """
    A class that implement three tasks : DSP, CSP and SRL.

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
        super(JointDCS, self).__init__(vocab=vocab, regularizer=regularizer)

        # Base text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder

        # Encoder
        encoder_params = params.pop("encoder")
        encoder = Seq2SeqEncoder.from_params(encoder_params)
        self._encoder = encoder

        self._tag_representation_dim = params.pop('tag_representation_dim')
        self._arc_representation_dim = params.pop('arc_representation_dim')

        self._dropout = params.pop('dropout')
        self._input_dropout = params.pop('input_dropout')

        ############
        # DSP Stuffs
        ############
        dsp_params = params.pop("dsp")

        init_params = dsp_params.pop("initializer", None)
        self._initializer = (
            InitializerApplicator.from_params(init_params) if init_params is not None else InitializerApplicator()
        )
        pos_params = dsp_params.pop("pos_tag_embedding")
        self._pos_tag_embedding = Embedding.from_params(vocab, pos_params)

        # Tagger DSP - Biaffine Tagger
        tagger_dsp = BiaffineParser(
            vocab=vocab,
            task_type='dsp',
            text_field_embedder=self._text_field_embedder,
            encoder=self._encoder,
            tag_representation_dim=self._tag_representation_dim,
            arc_representation_dim=self._arc_representation_dim,
            pos_tag_embedding=self._pos_tag_embedding,
            dropout=self._dropout,
            input_dropout=self._input_dropout,
            initializer=self._initializer
        )
        self._tagger_dsp = tagger_dsp

        # arc shared
        self._arc_attention = tagger_dsp.arc_attention
        self._head_arc_feedforward = tagger_dsp.head_arc_feedforward
        self._child_arc_feedforward = tagger_dsp.child_arc_feedforward

        ############
        # SRL Stuffs
        ############
        srl_params = params.pop("srl")

        # init_params = srl_params.pop("initializer", None)
        # self._initializer = (
        #     InitializerApplicator.from_params(init_params) if init_params is not None else InitializerApplicator()
        # )
        # pos_params = srl_params.pop("pos_tag_embedding")
        # self._pos_tag_embedding = Embedding.from_params(vocab, pos_params)

        # Tagger: SRL - Biaffine Tagger
        tagger_srl = BiaffineParser(
            vocab=vocab,
            task_type='srl',
            text_field_embedder=self._text_field_embedder,
            encoder=self._encoder,
            tag_representation_dim=self._tag_representation_dim,
            arc_representation_dim=self._arc_representation_dim,
            pos_tag_embedding=self._pos_tag_embedding,
            dropout=self._dropout,
            input_dropout=self._input_dropout,
            initializer=self._initializer
        )
        tagger_srl.arc_attention = self._arc_attention
        tagger_srl.head_arc_feedforward = self._head_arc_feedforward
        tagger_srl.child_arc_feedforward = self._child_arc_feedforward
        self._tagger_srl = tagger_srl

        ############
        # CSP Stuffs
        ############

        csp_params = params.pop("csp")
        init_params = csp_params.pop("initializer", None)
        self._initializer = (
            InitializerApplicator.from_params(init_params) if init_params is not None else InitializerApplicator()
        )
        # pos_params = csp_params.pop("pos_tag_embedding")
        # self._pos_tag_embedding = Embedding.from_params(vocab, pos_params)

        span_params = csp_params.pop("span_extractor")
        self._span_extractor = SpanExtractor.from_params(span_params)

        feed_forward_params = csp_params.pop("feedforward")
        self._feed_forward = FeedForward.from_params(feed_forward_params)

        # Tagger: CSP - SpanConstituencyParser Tagger
        tagger_csp = SpanConstituencyParser(
            vocab=vocab,
            text_field_embedder=self._text_field_embedder,
            span_extractor=self._span_extractor,
            encoder=self._encoder,
            feedforward=self._feed_forward,
            pos_tag_embedding=self._pos_tag_embedding,
            initializer=self._initializer
        )

        self._tagger_csp = tagger_csp

        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides
    def forward(self, tensor_batch, for_training: bool = False, task_name: str = "") -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        tagger = getattr(self, "_tagger_%s" % task_name)

        # print(tensor_batch)
        # for key, value in tensor_batch.items():
        #     tensor_batch[key] = value.half()

        return tagger.forward(**tensor_batch)

    def forward(self, tensor_batch, for_training: bool = False, task_names: list = [])->List[Dict[str, torch.Tensor]]:

        taggers = [getattr(self, "_tagger_%s" % task_name) for task_name in task_names]

        return [tagger.forward(**tensor_batch) for tagger in taggers]


    @overrides
    def get_metrics(self, task_name: str, reset: bool = False, full: bool = False) -> Dict[str, float]:
        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> Model:
        return cls(vocab=vocab, params=params, regularizer=regularizer)
