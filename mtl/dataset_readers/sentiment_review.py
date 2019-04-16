# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     senmantic_review
   Description :
   Author :       xmz
   date：          19-4-10
-------------------------------------------------
"""

from typing import Dict
import logging

from allennlp.data import Field
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("semantic_review")
class SentimentReviewDatasetReader(DatasetReader):
    """
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentence into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_stemmer=PorterStemmer())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                segments = line.split('\t')
                if len(segments) == 2:
                    label = segments[0]
                    sentence = segments[1]
                yield self.text_to_instance(sentence, label)

    @overrides
    def text_to_instance(self, text: str, label: int) -> Instance:  # type: ignore
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label : ``int``, optional, (default = None).
            The label for this text.
        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        tokens = self._tokenizer.tokenize(text)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
