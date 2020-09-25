# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

# Adapted from Hugging Face library to fine-tune on synthetic divergences using multi task loss
# @eleftheria

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, src_words_eq, tgt_words_eq, src_words_dv, tgt_words_dv, src_labels_dv, tgt_labels_dv,
                 label):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.src_words_eq = src_words_eq
        self.tgt_words_eq = tgt_words_eq
        self.src_words_dv = src_words_dv
        self.tgt_words_dv = tgt_words_dv
        self.src_labels_dv = src_labels_dv
        self.tgt_labels_dv = tgt_labels_dv
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_eq, input_mask_eq, segment_ids_eq, input_ids_dv, input_mask_dv, segment_ids_dv,
                 label_ids_dv, label):
        self.input_ids_eq = input_ids_eq
        self.input_mask_eq = input_mask_eq
        self.segment_ids_eq = segment_ids_eq
        self.input_ids_dv = input_ids_dv
        self.input_mask_dv = input_mask_dv
        self.segment_ids_dv = segment_ids_dv
        self.label_ids_dv = label_ids_dv
        self.label=label


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.tsv".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        for id_,line in enumerate(f):
            line = line.rstrip().split('\t')
            if mode == 'train' or mode == 'dev':
                eqv_src = line[0].split(' ')
                eqv_tgt = line[1].split(' ')
                div_src = line[2].split(' ')
                div_tgt = line[3].split(' ')
                div_src_token_ = line[4].split(' ')
                div_tgt_token_ = line[5].split(' ')
                lbl=1
            else:
                # Skip header
                if id_ == 0:
                    continue
                # Pad first encoding for test instances
                eqv_src, eqv_tgt = None, None
                lbl = line[0]
                div_src = line[3].split(' ')
                div_tgt = line[4].split(' ')
                div_src_token_ = line[5].split(' ')
                div_tgt_token_ = line[6].split(' ')

            assert len(div_src) == len(div_src_token_)
            assert len(div_tgt) == len(div_tgt_token_)

            span_label_maps = {'CHA':'D','ADD':'D','O':'O','D':'D','1':'D','2':'D','3':'D', '0':'O'}

            # Binarize span labels
            div_src_token = [span_label_maps[x] for x in div_src_token_]
            div_tgt_token = [span_label_maps[x] for x in div_tgt_token_]


            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         src_words_eq=eqv_src,
                                         tgt_words_eq=eqv_tgt,
                                         src_words_dv=div_src,
                                         tgt_words_dv=div_tgt,
                                         src_labels_dv=div_src_token,
                                         tgt_labels_dv=div_tgt_token,
                                         label=lbl))

            guid_index += 1

    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 mode,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):


    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens_dv, label_ids_dv, segment_ids_dv = [], [], []
        input_ids_eq, input_mask_eq, segment_ids_eq, segment_ids_eq = None, None, None, None
        # Token-level annotation for divergent source sentence
        for word, label in zip(example.src_words_dv, example.src_labels_dv):
            word_tokens = tokenizer.tokenize(word)
            tokens_dv.extend(word_tokens)
            # Propagate token-level label to all of its subwords
            label_ids_dv.extend([label_map[label]] * (len(word_tokens)))

        # Add [SEP] token and pad its respective label_id
        tokens_dv += [sep_token]
        label_ids_dv += [pad_token_label_id]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens_dv += [sep_token]
            label_ids_dv += [pad_token_label_id]

        source_length = len(tokens_dv)
        segment_ids_dv.extend([sequence_a_segment_id] * source_length)

        # Token-level annotation for divergent target sentence
        for word, label in zip(example.tgt_words_dv, example.tgt_labels_dv):
            word_tokens = tokenizer.tokenize(word)
            tokens_dv.extend(word_tokens)
            # Use the same id for all tokens of the word
            label_ids_dv.extend([label_map[label]] * (len(word_tokens)))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_dv) > max_seq_length - special_tokens_count:
            tokens_dv = tokens_dv[:(max_seq_length - special_tokens_count)]
            label_ids_dv = label_ids_dv[:(max_seq_length - special_tokens_count)]

        tokens_dv += [sep_token]
        label_ids_dv += [pad_token_label_id]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens_dv += [sep_token]
            label_ids_dv += [pad_token_label_id]
        segment_ids_dv.extend([sequence_b_segment_id] * (len(tokens_dv)-source_length))

        if cls_token_at_end:
            tokens_dv += [cls_token]
            label_ids_dv += [pad_token_label_id]
            segment_ids_dv += [cls_token_segment_id]
        else:
            tokens_dv = [cls_token] + tokens_dv
            label_ids_dv = [pad_token_label_id] + label_ids_dv
            segment_ids_dv = [cls_token_segment_id] + segment_ids_dv

        input_ids_dv = tokenizer.convert_tokens_to_ids(tokens_dv)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_dv = [1 if mask_padding_with_zero else 0] * len(input_ids_dv)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids_dv)
        if pad_on_left:
            input_ids_dv = ([pad_token] * padding_length) + input_ids_dv
            input_mask_dv = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask_dv
            segment_ids_dv = ([pad_token_segment_id] * padding_length) + segment_ids_dv
            label_ids_dv = ([pad_token_label_id] * padding_length) + label_ids_dv
        else:
            input_ids_dv += ([pad_token] * padding_length)
            input_mask_dv += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids_dv += ([pad_token_segment_id] * padding_length)
            label_ids_dv += ([pad_token_label_id] * padding_length)

        
        # Encode the equivalent sentence-pair
        if mode == 'train' or mode == 'dev':
            tokens_eq = []
            segment_ids_eq = []

            # Source sentence
            for word in example.src_words_eq:
                word_tokens = tokenizer.tokenize(word)
                tokens_eq.extend(word_tokens)

            # Add [SEP] token and pad its respective label_id
            tokens_eq += [sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_eq += [sep_token]

            source_length = len(tokens_eq)
            segment_ids_eq.extend([sequence_a_segment_id] * source_length)

            # Target sentence
            for word in example.tgt_words_eq:
                word_tokens = tokenizer.tokenize(word)
                tokens_eq.extend(word_tokens)

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_eq) > max_seq_length - special_tokens_count:
                tokens_eq = tokens_eq[:(max_seq_length - special_tokens_count)]

            tokens_eq += [sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_eq += [sep_token]
            segment_ids_eq.extend([sequence_b_segment_id] * (len(tokens_eq)-source_length))

            if cls_token_at_end:
                tokens_eq += [cls_token]
                segment_ids_eq += [cls_token_segment_id]
            else:
                tokens_eq = [cls_token] + tokens_eq
                segment_ids_eq = [cls_token_segment_id] + segment_ids_eq

            input_ids_eq = tokenizer.convert_tokens_to_ids(tokens_eq)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask_eq = [1 if mask_padding_with_zero else 0] * len(input_ids_eq)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids_eq)
            if pad_on_left:
                input_ids_eq = ([pad_token] * padding_length) + input_ids_eq
                input_mask_eq = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask_eq
                segment_ids_eq = ([pad_token_segment_id] * padding_length) + segment_ids_eq
            else:
                input_ids_eq += ([pad_token] * padding_length)
                input_mask_eq += ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids_eq += ([pad_token_segment_id] * padding_length)

            assert len(input_ids_eq) == max_seq_length
            assert len(input_mask_eq) == max_seq_length
            assert len(segment_ids_eq) == max_seq_length

        if ex_index < 5:
            if mode == 'train' or mode == 'dev':
                logger.info("*** Example (first encoding) ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens_eq]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids_eq]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask_eq]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids_eq]))

            logger.info("*** Example (second encoding) ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens_dv]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids_dv]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask_dv]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids_dv]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids_dv]))
        features.append(
                InputFeatures(input_ids_eq=input_ids_eq,
                              input_mask_eq=input_mask_eq,
                              segment_ids_eq=segment_ids_eq,
                              input_ids_dv=input_ids_dv,
                              input_mask_dv=input_mask_dv,
                              segment_ids_dv=segment_ids_dv,
                              label_ids_dv=label_ids_dv,
                              label=example.label
                              ))
    return features

def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "D"]
