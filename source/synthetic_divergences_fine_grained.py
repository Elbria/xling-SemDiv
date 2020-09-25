# -*- coding: utf-8 -*-

"""
Created on 29 March 2020

@author: eleftheria

The modules for creating the 'unever, replace, random delete, insert' where implemented

        Module implemented by: https://github.com/SYSTRAN/similarity/blob/master/src/build_data.py


"""

import numpy as np
import random
import torch
import itertools
import spacy

from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from utils import leaf_hyponyms, alignments2dic, flatten_tree, sublist_indices

tokenizer = RegexpTokenizer(r'\w+')
flatten = itertools.chain.from_iterable
stopwords_list = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


class synthetic_divergences:

    def __init__(self):
        self.SRC = []
        self.TGT = []
        self.POS = []
        self.ALI = []
        self.length = 0
        self.max_ntry = 80

    def __len__(self):
        return len(self.SRC)

    def add(self, src, tgt, pos, ali):
        self.SRC.append(src)
        self.TGT.append(tgt)
        self.POS.append(pos)
        self.ALI.append(alignments2dic(ali))
        self.length += 1

    def bert_control(self, en_text, target, model, tokenizer):

        """

        Return bert predictions based on masked words

        :param en_text: english text
        :param target: token to be masked
        :param model: pre-trained model
        :param tokenizer: pre-trained tokenizer
        :return: top predictions based on language model control

        """
        en_text = en_text.encode('utf-8').decode('utf-8')
        en_tokenized_text = tokenizer.tokenize("[CLS] " + en_text)

        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_index = en_tokenized_text.index(target)
        en_tokenized_text[masked_index] = '[MASK]'

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(en_tokenized_text)

        if len(indexed_tokens) > 512:
            return []
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0] * (len(en_tokenized_text))

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load pre-trained model (weights)
        model.eval()

        top_predictions = []
        # Predict all tokens
        predictions = model(tokens_tensor)
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        top_predictions.append(predicted_token[0].lower())

        for i in range(100):
            predictions[0, masked_index, predicted_index] = -11100000
            predicted_index = torch.argmax(predictions[0, masked_index]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
            top_predictions.append(predicted_token[0].lower())

        # Exclude target from predictions
        top_predictions_exclude_target = [element for element in top_predictions if element != target]

        return top_predictions_exclude_target

    def uneven_pair(self, i, o):

        """

        Create uneven pairs from set of parallel sentences

        Module implemented by: https://github.com/SYSTRAN/similarity/blob/master/src/build_data.py

        :param i: index of current example
        :param o: main arguments
        :return: uneven example

        """

        n_try = 0
        while True:
            n_try += 1
            if n_try > self.max_ntry:
                # cannot find the right pair
                return None
            j = np.random.randint(0, len(self.SRC))
            if i % 2 == 0:
                # replace src
                if self.SRC[i] == self.SRC[j]:
                    continue

                src = list(self.SRC[j])
                tgt = list(self.TGT[i])

                if len(src) > len(tgt) and len(src) * 0.8 > len(tgt):
                    continue
                if len(tgt) > len(src) and len(tgt) * 0.8 > len(src):
                    continue

                tgt_span = ['O'] * len(tgt)
                src_span = ['CHA'] * len(src)

                if o.debug:
                    print('\nUNEVEN ON SRC.   original : {0}'.format(str(' '.join(self.SRC[i]))))
                    print('\t\t synthetic : {0}'.format(str(' '.join(src))))
            else:
                # replace tgt
                if self.TGT[i] == self.TGT[j]:
                    continue

                src = list(self.SRC[i])
                tgt = list(self.TGT[j])

                if len(src) > len(tgt) and len(src) * 0.8 > len(tgt):
                    continue
                if len(tgt) > len(src) and len(tgt) * 0.8 > len(src):
                    continue

                src_span = ['O'] * len(src)
                tgt_span = ['CHA'] * len(tgt)

                if o.debug:
                    print('\nUNEVEN ON TGT.   original : {0}'.format(str(' '.join(self.TGT[i]))))
                    print('\t\t synthetic : {0}'.format(str(' '.join(tgt))))

            break

        return [src, tgt, src_span, tgt_span]

    def generalization_pair(self, i, o, model, tokenizer, multiple=True):

        """

        Create generalization examples using LM control

        :param i: index of current example
        :param o: main arguments
        :param model: pre-trained BERT-based model
        :param tokenizer: tokenizer for pre-trained BERT-based model
        :param multiple: flag controlling whether to perform one or multiple replacements
        :return: generalization example

        """
        print(i)
        english_sentence = self.SRC[i]
        pos_english_sentence = self.POS[i]
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        ali = self.ALI[i]
        src_span = ['O' for i in range(len(src))]
        tgt_span = ['O' for i in range(len(tgt))]

        synth_src = []
        pos_found = [id for id, s in enumerate(pos_english_sentence) if
                     s in ['VBN', 'JJ', 'VB', 'VBD', 'VBP', 'JJR', 'VBZ', 'VBG', 'JJS', 'NN', 'NNS']]
        # For each pos-tag
        attempts = 0
        while len(pos_found) != 0 and attempts < 20:
            attempts += 1
            to_replace_idx = random.choice(pos_found)
            to_replace = english_sentence[to_replace_idx]
            # prevent pos-tag ambiguity
            to_replace_pos = pos_english_sentence[to_replace_idx]
            if to_replace in stopwords_list:
                pos_found.remove(to_replace_idx)
                continue

            if to_replace_pos in ['NNS', 'NN']:
                synset = wn.synsets(to_replace, pos=wn.NOUN)
            elif to_replace_pos in ['VBN', 'VB', 'VBD', 'VBP', 'VBZ', 'VBG']:
                synset = wn.synsets(to_replace, pos=wn.VERB)
            else:
                synset = wn.synsets(to_replace, pos=wn.ADJ)
            text = ' '.join(english_sentence)
            target = to_replace

            if synset:
                word_hypernmyms = []

                # Extract one path that connects word to root for each synset corresponding to the word
                for syn_id in range(len(synset)):
                    full_word_hypernmyms = list(
                        flatten([tmp.lemma_names() for tmp in synset[syn_id].hypernym_paths()[0][:-1]][::-1][0:2]))

                    root_hypernyms = list(flatten([tmp.lemma_names() for tmp in synset[syn_id].root_hypernyms()]))
                    word_hypernmyms.extend(
                        [hyp for hyp in full_word_hypernmyms if hyp not in root_hypernyms and len(hyp.split('_')) == 1])

                if word_hypernmyms:
                    # Search candidate hypernym in bert list, fix grammatical errors
                    try:
                        bert_list = self.bert_control(text, target, model, tokenizer)
                    except ValueError:
                        pos_found.remove(to_replace_idx)
                        continue
                    lm_control = False

                    # Search at the same level of all hypernym paths
                    for bert_word in bert_list:
                        if bert_word in word_hypernmyms:
                            replace_with = bert_word
                            lm_control = True
                            break

                    if lm_control:

                        synth_src = []
                        for id, token in enumerate(english_sentence):
                            if id != to_replace_idx:
                                synth_src.append(token)
                            else:
                                synth_src.append(replace_with)

                        # Update source and target spans
                        src_span[to_replace_idx] = 'CHA'
                        try:
                            tgt_span[ali[to_replace_idx]] = 'CHA'
                        except KeyError:
                            continue

                        if o.debug:
                            print('\nWord to replace: {0}'.format(str(to_replace)))
                            print('\nWordNet Hypernyms: {0}'.format(str(bert_list)))
                            print('\nGENERALIZATION. replace : {0}'.format(str(' '.join(english_sentence))))
                            print('\t\t      with : {0}'.format(str(' '.join(synth_src))))
                            print('\n------------------------------------------------------------------------------\n')
                        if multiple:
                            pos_found.remove(to_replace_idx)
                            english_sentence = synth_src
                        else:
                            if len(synth_src) == 0:
                                return None
                            else:
                                return [synth_src, self.TGT[i]]
                    else:
                        pos_found.remove(to_replace_idx)
                else:
                    pos_found.remove(to_replace_idx)
            else:
                pos_found.remove(to_replace_idx)

        if len(synth_src) == 0:
            return None
        else:
            return [synth_src, self.TGT[i], src_span, tgt_span]

    def particularization_pair(self, i, o, model, tokenizer, multiple=True):

        """

        Create particularization examples using LM control

        :param i: index of current example
        :param o: main arguments
        :param model: pre-trained BERT-based model
        :param tokenizer: tokenizer for pre-trained BERT-based model
        :param multiple: flag controlling whether to perform one or multiple replacements
        :return: particularization example

        """
        print(i)
        english_sentence = self.SRC[i]
        pos_english_sentence = self.POS[i]
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        ali = self.ALI[i]
        src_span = ['O' for i in range(len(src))]
        tgt_span = ['O' for i in range(len(tgt))]

        synth_src = []
        pos_found = [id for id, s in enumerate(pos_english_sentence) if
                     s in ['VBN', 'JJ', 'VB', 'VBD', 'VBP', 'JJR', 'VBZ', 'VBG', 'JJS', 'NN', 'NNS']]

        attempts = 0
        # For each pos-tag
        while len(pos_found) != 0 and attempts < 20:
            attempts += 1
            to_replace_idx = random.choice(pos_found)
            to_replace = english_sentence[to_replace_idx]
            # Prevent pos-tag ambiguity
            to_replace_pos = pos_english_sentence[to_replace_idx]
            if to_replace in stopwords_list:
                pos_found.remove(to_replace_idx)
                continue

            if to_replace_pos in ['NNS', 'NN']:
                synset = wn.synsets(to_replace, pos=wn.NOUN)
            elif to_replace_pos in ['VBN', 'VB', 'VBD', 'VBP', 'VBZ', 'VBG']:
                synset = wn.synsets(to_replace, pos=wn.VERB)
            else:
                synset = wn.synsets(to_replace, pos=wn.ADJ)

            text = ' '.join(english_sentence)
            target = to_replace

            if synset:
                word_hyponyms = []

                # Extract one path that connects word to root for each synset corresponding to the word
                for syn_id in range(len(synset)):
                    try:
                        word_hyponyms.extend(
                            [tmp.lemma_names() for tmp in leaf_hyponyms(synset[syn_id])])
                    except RecursionError:
                        print('Recursion Error.. Abandon example :/ ')
                        continue
                word_hyponyms = list(flatten(word_hyponyms))

                if word_hyponyms:
                    # Search candidate hypernym in bert list, fix grammatical errors
                    try:
                        bert_list = self.bert_control(text, target, model, tokenizer)
                    except ValueError:
                        pos_found.remove(to_replace_idx)
                        continue
                    lm_control = False

                    # Search at the same level of all hypernym paths
                    for bert_word in bert_list:
                        if bert_word in word_hyponyms:
                            replace_with = bert_word
                            lm_control = True
                            break

                    if lm_control:

                        synth_src = []
                        for id, token in enumerate(english_sentence):
                            if id != to_replace_idx:
                                synth_src.append(token)

                            else:
                                synth_src.append(replace_with)

                        # Update source and target spans
                        src_span[to_replace_idx] = 'CHA'
                        try:
                            tgt_span[ali[to_replace_idx]] = 'CHA'
                        except KeyError:
                            continue

                        if o.debug:
                            print('\nWord to replace: {0}'.format(str(to_replace)))
                            print('\nWordNet Hyponyms: {0}'.format(str(bert_list)))
                            print('\nPARTICULARIZATION. replace : {0}'.format(str(' '.join(english_sentence))))
                            print('\t\t      with : {0}'.format(str(' '.join(synth_src))))
                            print('\n------------------------------------------------------------------------------\n')
                        if multiple:
                            pos_found.remove(to_replace_idx)
                            english_sentence = synth_src
                        else:
                            if len(synth_src) == 0:
                                return None
                            else:
                                return [synth_src, self.TGT[i]]
                    else:
                        pos_found.remove(to_replace_idx)
                else:
                    pos_found.remove(to_replace_idx)
            else:
                pos_found.remove(to_replace_idx)
        if len(synth_src) == 0:
            return None
        else:
            return [synth_src, self.TGT[i], src_span, tgt_span]

    def insert_pair(self, i, o):

        """

        Create insert pairs from set of parallel sentences

        Module implemented by: https://github.com/SYSTRAN/similarity/blob/master/src/build_data.py

        :param i: index of current example
        :param o: main arguments
        :return: insert example

        """

        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        src_span = ['O' for i in range(len(src))]
        tgt_span = ['O' for i in range(len(tgt))]

        where = ""
        if len(src) <= len(tgt):
            # Add in src side
            n_try = 0
            while True:
                n_try += 1
                if n_try > self.max_ntry:
                    # Cannot find the right pair
                    return
                j = np.random.randint(0, len(self.SRC))
                if j == i:
                    continue
                # Replace src
                add = list(self.SRC[j])
                new_src = len(src) + len(add)
                if new_src > len(tgt) and new_src > len(tgt) * 2:
                    continue
                if len(tgt) > new_src and len(tgt) > new_src * 2:
                    continue
                break
            if i % 2 == 0:
                # Add in the beginning
                where = "src:begin"
                for k in range(len(add)):
                    src.insert(0, add[len(add) - k - 1])
                    src_span.insert(0, 'ADD')
            else:
                # Add in the end
                where = "src:end"
                for k in range(len(add)):
                    src.append(add[k])
                    src_span.append('ADD')
            if o.debug:
                print('\nINSERT ON SRC.   original : {0}'.format(str(' '.join(self.SRC[i]))))
                print('\t\t synthetic : {0}'.format(str(' '.join(src))))
        else:
            # Add in tgt side
            n_try = 0
            while True:
                n_try += 1
                if n_try > self.max_ntry:
                    # Cannot find the right pair
                    return
                j = np.random.randint(0, len(self.SRC))
                if j == i:
                    continue
                # Replace tgt
                add = list(self.TGT[j])
                new_tgt = len(tgt) + len(add)
                if len(src) > new_tgt and len(src) > new_tgt * 2:
                    continue
                if new_tgt > len(src) and new_tgt > len(src) * 2:
                    continue
                break
            if i % 2 == 0:
                # Add in the begining
                where = "tgt:begin"
                for k in range(len(add)):
                    tgt.insert(0, add[len(add) - k - 1])
                    tgt_span.insert(0, 'ADD')
            else:
                # add in the end
                where = "tgt:end"
                for k in range(len(add)):
                    tgt.append(add[k])
                    tgt_span.append('ADD')
            if o.debug:
                print('\nINSERT ON TGT.   original : {0}'.format(str(' '.join(self.TGT[i]))))
                print('\t\t synthetic : {0}'.format(str(' '.join(tgt))))

        return [src, tgt, src_span, tgt_span]

    def replace_pair(self, i, o, pos_to_wrd):

        """

        Create replace pairs from set of parallel sentences

        Module implemented by: https://github.com/SYSTRAN/similarity/blob/master/src/build_data.py

        :param i: index of current example
        :param o: main arguments
        :return: replace example

        """

        if len(self.POS) == 0:
            return
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        pos = list(self.POS[i])
        ali = self.ALI[i]
        src_span = ['O' for i in range(len(src))]
        tgt_span = ['O' for i in range(len(tgt))]
        if len(src) <= 3:
            return

        attempts = 0

        while attempts < 100:
            span_range = random.randint(2, int(len(src) / 2))
            position = random.randint(0, len(src))

            if position + span_range > len(src):
                start = position - span_range
                end = position
            else:
                start = position
                end = position + span_range

            candidate = ' '.join(pos[start:end])
            canditate_to_be_replaced = ' '.join(src[start:end])
            attempts += 1

            if candidate in pos_to_wrd.keys():
                tmp = pos_to_wrd[candidate]
                try:
                    pos_to_wrd[candidate].remove(canditate_to_be_replaced)
                except ValueError:
                    continue
                if len(tmp) != 0:
                    synthetic_src = src[0:start] + random.choice(pos_to_wrd[candidate]).split() + src[end:]
                    for i_ in range(start, end):
                        src_span[i_] = 'CHA'
                        try:
                            tgt_span[ali[i_]] = 'CHA'
                        except KeyError:
                            continue
                    if o.debug:
                        print('\nREPLACE.   original : {0}'.format(str(' '.join(self.SRC[i]))))
                        print('\t   synthetic : {0}'.format(str(' '.join(synthetic_src))))

                    return [synthetic_src, tgt, src_span, tgt_span]
                else:
                    continue
        return None

    def delete_pair_random(self, i, o):

        """

        Create delete pairs from set of parallel sentences

        Module implemented by: https://github.com/SYSTRAN/similarity/blob/master/src/build_data.py

        :param i: index of current example
        :param o: main arguments
        :return: replace example

        """
        src_subtrees = []
        if len(self.POS) == 0:
            return
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        ali = self.ALI[i]
        tgt_span = ['O' for i in range(len(tgt))]
        if len(src) <= 3:
            return

        span_range = random.randint(2, int(len(src) / 2))
        position = random.randint(0, len(src))

        if position + span_range > len(src):
            start = position - span_range
            end = position
        else:
            start = position
            end = position + span_range

        synthetic_src = src[0:start] + src[end:]
        src_span = ['O'] * len(synthetic_src)

        for i_ in range(start, end):
            try:
                tgt_span[ali[i_]] = 'ADD'
            except KeyError:
                continue
        if o.debug:
            print('\nDELETE.    original : {0}'.format(str(' '.join(self.SRC[i]))))
            print('\t   synthetic : {0}'.format(str(' '.join(synthetic_src))))
        return [synthetic_src, tgt, src_span, tgt_span]

    def delete_pair(self, i, o):

        """

        Create delete pairs from set of parallel sentences

        :param i: index of current example
        :param o: main arguments
        :return: replace example

        """
        subtrees = []
        if len(self.POS) == 0:
            return
        src = list(self.SRC[i])
        tgt = list(self.TGT[i])
        ali = self.ALI[i]

        if len(src) <= 3:
            return

        src_span = ['O' for i in range(len(src))]
        tgt_span = ['O' for i in range(len(tgt))]

        # Retrieve all subtrees of source sentence (dependency parsed tree)
        depend_src = nlp(' '.join(self.SRC[i]))
        for id_ in range(len(depend_src)):
            subtrees.append(flatten_tree(depend_src[id_].subtree))

        # Deleted spans should be have more than 1 token and less than half of the sentence
        filtered_subtrees = [x for x in subtrees if len(x.split(' ')) <= len(src) / 2]
        filtered_subtrees = [x for x in filtered_subtrees if len(x.split(' ')) != 1]

        attempts = 0

        # Randomly choose a span to be deleted
        while attempts < 20:

            attempts += 1

            try:
                to_delete_ind = random.randint(2, len(filtered_subtrees) - 1)
            except ValueError:
                continue

            span_to_delete = filtered_subtrees[to_delete_ind].split(' ')

            # Find indices in source sentence
            try:
                start, end = sublist_indices(span_to_delete, src)
            except TypeError:
                continue

            change_side = random.randint(0, 1)

            # Delete on source side (add on target side)
            if change_side == 0:
                synthetic_src = src[0:start] + src[end:]
                synthetic_tgt = tgt
                for i_ in range(start, end):
                    try:
                        tgt_span[ali[i_]] = 'ADD'
                    except KeyError:
                        continue
                src_span = ['O'] * len(synthetic_src)

                if o.debug:
                    print('\n')
                    print('DELETE on src.  original : {0}'.format(str(' '.join(self.SRC[i]))))
                    print('\t        synthetic : {0}'.format(str(' '.join(synthetic_src))))
                    print('\nTarget: {0}'.format(str(' '.join(tgt))))
                    print('\n\tSrc span. : {0}'.format(str(' '.join(src_span))))
                    print('\tTgt span. : {0}'.format(str(' '.join(tgt_span))))

            else:
                synthetic_src = src
                synthetic_tgt = []
                to_exclude = []
                for i_ in range(start, end):
                    src_span[i_] = 'ADD'
                    try:
                        to_exclude.append(ali[i_])
                    except KeyError:
                        continue
                for id_, tmp in enumerate(tgt):
                    if id_ in to_exclude:
                        continue
                    else:
                        synthetic_tgt.append(tmp)

                tgt_span = ['O'] * len(synthetic_tgt)

                if o.debug:
                    print('\n')
                    print('DELETE on tgt.  original  : {0}'.format(str(' '.join(self.TGT[i]))))
                    print('\t        synthetic : {0}'.format(str(' '.join(synthetic_tgt))))
                    print('\nSource: ' + str(' '.join(src)))
                    print('\n\tSrc span. : {0}'.format(str(' '.join(src_span))))
                    print('\tTgt span. : {0}'.format(str(' '.join(tgt_span))))

            if len(set(src_span)) == 1 and len(tgt_span) == 1:
                continue

            return [synthetic_src, synthetic_tgt, src_span, tgt_span]
