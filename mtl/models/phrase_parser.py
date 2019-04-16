# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Any

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.models import SpanConstituencyParser, Model
from allennlp.nn.util import get_text_field_mask, get_lengths_from_binary_sequence_mask, masked_softmax, \
    sequence_cross_entropy_with_logits

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("phrase_parser")
class PhraseParser(SpanConstituencyParser):

    def forward(self, tokens: Dict[str, torch.LongTensor], spans: torch.LongTensor, metadata: List[Dict[str, Any]],
                pos_tags: Dict[str, torch.LongTensor] = None, span_labels: torch.LongTensor = None) -> Dict[
        str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self.pos_tag_embedding is not None:
            embedded_pos_tags = self.pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self.pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(tokens)
        # Looking at the span start index is enough to know if
        # this is padding or not. Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).long()
        if span_mask.dim() == 1:
            # This happens if you use batch_size 1 and encounter
            # a length 1 sentence in PTB, which do exist. -.-
            span_mask = span_mask.unsqueeze(-1)
        if span_labels is not None and span_labels.dim() == 1:
            span_labels = span_labels.unsqueeze(-1)

        num_spans = get_lengths_from_binary_sequence_mask(span_mask)

        encoded_text = self.encoder(embedded_text_input, mask)

        span_representations = self.span_extractor(encoded_text, spans, mask, span_mask)

        if self.feedforward_layer is not None:
            span_representations = self.feedforward_layer(span_representations)

        logits = self.tag_projection_layer(span_representations)
        class_probabilities = masked_softmax(logits, span_mask.unsqueeze(-1))

        output_dict = {
            "class_probabilities": class_probabilities,
            "spans": spans,
            "tokens": [meta["tokens"] for meta in metadata],
            "pos_tags": [meta.get("pos_tags") for meta in metadata],
            "num_spans": num_spans
        }
        if span_labels is not None:
            loss = sequence_cross_entropy_with_logits(logits.float(), span_labels, span_mask)
            self.tag_accuracy(class_probabilities, span_labels, span_mask)
            output_dict["loss"] = loss

        # The evalb score is expensive to compute, so we only compute
        # it for the validation and test sets.
        batch_gold_trees = [meta.get("gold_tree") for meta in metadata]
        if all(batch_gold_trees) and self._evalb_score is not None and not self.training:
            gold_pos_tags: List[List[str]] = [list(zip(*tree.pos()))[1]
                                              for tree in batch_gold_trees]
            predicted_trees = self.construct_trees(class_probabilities.cpu().data,
                                                   spans.cpu().data,
                                                   num_spans.data,
                                                   output_dict["tokens"],
                                                   gold_pos_tags)
            self._evalb_score(predicted_trees, batch_gold_trees)

        return output_dict
