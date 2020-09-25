# -*- coding: utf-8 -*-

"""
Created on 29 March 2020

@author: eleftheria
"""

import argparse
import os
import random

from utils import divergent_mappings
from utils import bert_header

global none_
none_='None\n'

# =====================================================================================================================#
#                                           Divergent list dictionary                                                  #
# ---------------------------------------------------------------------------------------------------------------------#
#           Abbreviation        |           Short name              |               Full name                          #
# ---------------------------------------------------------------------------------------------------------------------#
#               r               |           replace                 |           phrase replacement                     #
#               d               |           delete                  |           subtree deletion                       #
#               p               |           particularization       |           lexical substitution (hyponym)         #
#               g               |           generalization          |           lexical substitution (hypernym)        #
# =====================================================================================================================#


def main():
    parser = argparse.ArgumentParser(description='Create training/dev data for divergnemBERT')
    parser.add_argument('--contrastive', help='defines if the sampling mode is contrastive', action='store_true')
    parser.add_argument('--num_training', help='number of training data', default=5000)
    parser.add_argument('--num_dev', help='number of dev data', default=500)
    parser.add_argument('--num_test', help='number of test data', default=500)
    parser.add_argument('--learn-to-rank', help='create triplet pairs version; only used when sampling mode is D', \
                        action='store_true')
    parser.add_argument('--divergence-ranking', help='create hard triplets', action='store_true')
    parser.add_argument('--multi-task', help='create multitask data', action='store_true')
    parser.add_argument('--balanced', help='controls whether the distribution of equivalents \
                                           divergences is balanced when multiple divergences are considered', \
                        action='store_true')
    parser.add_argument('--path_to_unlabeled', help='path to seed equivalents; we assume a sorted corpus')
    parser.add_argument('--path_to_divergences', help='path to divergent data')
    parser.add_argument('--divergent_list', help='list of divergences to consider', default='rdpg')
    parser.add_argument('--output', help='output directory of syntheticetic training data', \
                        default='./for_divergentmBERT')
    o = parser.parse_args()

    if o.contrastive:
        sampling_mode = 'contrastive'
    else:
        sampling_mode = 'non-contrastive'

    if o.divergence_ranking and o.balanced:
        print('Divergence ranking cannot be extracted under the balanced mode. Abort...')
        exit(0)

    # =================================================================================================================#
    #                                           CREATE OUTPUT DIRECTORIES                                              #
    # =================================================================================================================#

    original_corpus = o.path_to_divergences.split('/')[-1]
    output_path = os.path.join(o.output, original_corpus, sampling_mode, o.divergent_list)
    total_sents = o.num_training + o.num_dev + o.num_test

    # Learning to rank output path: Prepare data for divergentmBERT with margin loss
    if o.learn_to_rank:
        if sampling_mode == 'contrastive':
            mode = sampling_mode + '_learn2rank'
            output_path_learn2rank = os.path.join(o.output, original_corpus, mode, o.divergent_list)
            try:
                os.makedirs(output_path_learn2rank)
            except FileExistsError:
                # directory already exists
                pass
        else:
            print('Learning to rank can only be performed under contrastive sampling mode. Abort...')
            exit(0)

    # Learning to rank divergences: Prepare data for divergentmBERT with margin loss and divergence ranking
    if o.divergence_ranking:
        if sampling_mode == 'contrastive':
            mode = sampling_mode + '_divergence_ranking'
            output_path_divergence_ranking = os.path.join(o.output, original_corpus, mode, o.divergent_list)
            try:
                os.makedirs(output_path_divergence_ranking)
            except FileExistsError:
                # directory already exists
                pass
        else:
            print('Divergence ranking can only be performed under contrastive sampling mode. Abort...')
            exit(0)

    # Multi-task learning: Learning to rank seed equivalent VS divergences (sentence and token prediction)
    if o.multi_task and not o.divergence_ranking:
        if sampling_mode == 'contrastive':
            mode = sampling_mode + '_multi_task'
            output_path_multi_task = os.path.join(o.output, original_corpus, mode, o.divergent_list)
            try:
                os.makedirs(output_path_multi_task)
            except FileExistsError:
                # directory already exists
                pass
        else:
            print('Multi task was only implemented under contrastive sampling mode. Abort...')
            exit(0)

    # Multi-task learning: Learning to rank contrastive divergences (sentence and token prediction)
    if o.multi_task and o.divergence_ranking:
        if sampling_mode == 'contrastive':
            mode = sampling_mode + '_multi_hard'
            output_path_multi_task = os.path.join(o.output, original_corpus, mode, o.divergent_list)
            try:
                os.makedirs(output_path_multi_task)
            except FileExistsError:
                # directory already exists
                pass
        else:
            print('Multi task was only implemented under contrastive sampling mode. Abort...')
            exit(0)

    # CE loss: Prepare data for sentence-level Cross-Entropy loss
    try:
        os.makedirs(output_path)
    except FileExistsError:
        # directory already exists
        pass

    # =================================================================================================================#
    # =================================================================================================================#

    # Define number of divergences (negatives) to be sampled 
    if o.balanced:
        total_negs = o.num_training / len(o.divergent_list) + o.num_dev / len(o.divergent_list) + o.num_test / len(
            o.divergent_list)
    else:
        total_negs = o.num_training + o.num_dev / len(o.divergent_list) + o.num_test / len(o.divergent_list)

    # Read seed equivalents
    with open(o.path_to_unlabeled, 'r') as unlabeled:
        unlabeled = unlabeled.readlines()

    # Read divergents
    divs_sentences = []
    divs_sentences_spans = []
    for div in o.divergent_list:
        map_to_divergent = os.path.join(o.path_to_divergences, divergent_mappings[div])
        with open(map_to_divergent, 'r') as divergents, open('.'.join([map_to_divergent, 'span'])) as divergents_spans:
            divs_sentences.append(divergents.readlines())
            divs_sentences_spans.append(divergents_spans.readlines())

    # =================================================================================================================#
    #                                   WRITE OUTPUT FILES                                                             #
    # =================================================================================================================#

    unique_id_per_sentence = 0

    # ------------------------------- #
    #  Open train, test, dev files    #
    # ------------------------------- #

    with open('{0}/train.tsv'.format(output_path), 'w') as train, \
            open('{0}/test_synthetic.tsv'.format(output_path), 'w') as test_synthetic, \
            open('{0}/dev.tsv'.format(output_path), 'w') as dev:

        # Learning to rank files
        if o.learn_to_rank:
            train_learn2rank = open('{0}/train.tsv'.format(output_path_learn2rank), 'w')
            dev_learn2rank = open('{0}/dev.tsv'.format(output_path_learn2rank), 'w')
            test_synthetic_learn2rank = open('{0}/test_synthetic.tsv'.format(output_path_learn2rank), 'w')
            test_synthetic_learn2rank.write(bert_header)

        # Divergence ranking files
        if o.divergence_ranking and not o.multi_task:
            train_divergence_ranking = open('{0}/train.tsv'.format(output_path_divergence_ranking), 'w')
            dev_divergence_ranking = open('{0}/dev.tsv'.format(output_path_divergence_ranking), 'w')
            test_synthetic_divergence_ranking = open('{0}/test_synthetic.tsv'.format(output_path_divergence_ranking),
                                                     'w')
            test_synthetic_divergence_ranking.write(bert_header)

        # Create multi-task hard triplet version of dataset if mode is true
        if o.multi_task:
            multi_train_learn2rank = open('{0}/train.tsv'.format(output_path_multi_task), 'w')
            multi_dev_learn2rank = open('{0}/dev.tsv'.format(output_path_multi_task), 'w')
            multi_test_synthetic_learn2rank = open('{0}/test_synthetic.tsv'.format(output_path_multi_task), 'w')
            multi_test_synthetic_learn2rank.write(bert_header)

        train.write(bert_header)
        test_synthetic.write(bert_header)
        dev.write(bert_header)

        # ------------------------------------------------------------------------------------ #
        #  Divergences are sampled randomly from the pool (not paired with seed equivalents)   #
        # -------------------------------------------------------------------------------------#

        if sampling_mode == 'non-contrastive':

            # Seed equivalents
            train_n = range(0, o.num_training)
            dev_n = range(o.num_training, o.num_training + o.num_dev)
            test_n = range(o.num_training + o.num_dev, o.num_training + o.num_dev + o.num_test)

            for ex in train_n:
                train.write('0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                      unlabeled[ex]))
                unique_id_per_sentence += 1
            for ex in dev_n:
                dev.write('0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                    unlabeled[ex]))
                unique_id_per_sentence += 1
            for ex in test_n:
                test_synthetic.write(
                    '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                              unlabeled[ex]))
                unique_id_per_sentence += 1

            # Synthetic divergences
            for div_type in divs_sentences:

                # Randomly sample divergences of specific type
                random_p = []
                sampled_counter = 0

                while int(total_negs) != sampled_counter:
                    sample = random.sample(range(0, len(div_type)), 1)[0]
                    if not (not (sample not in random_p) or not (div_type[sample] != none_)):
                        random_p.append(sample)
                        sampled_counter += 1

                if o.balanced:
                    train_p = random_p[:int(o.num_training / len(o.divergent_list))]
                    dev_p = random_p[int(o.num_training / len(o.divergent_list)):int(
                        (o.num_training + o.num_dev) / len(o.divergent_list))]
                    test_p = random_p[int((o.num_training + o.num_dev) / len(o.divergent_list)):]
                else:
                    train_p = random_p[:o.num_training]
                    dev_p = random_p[o.num_training:int(o.num_training + o.num_dev / len(o.divergent_list))]
                    test_p = random_p[int(o.num_training + o.num_dev / len(o.divergent_list)):]

                for ex in train_p:
                    train.write(
                        '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  div_type[
                                                      ex]))
                    unique_id_per_sentence += 1
                for ex in dev_p:
                    dev.write(
                        '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  div_type[
                                                      ex]))
                    unique_id_per_sentence += 1
                for ex in test_p:
                    test_synthetic.write(
                        '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  div_type[
                                                      ex]))
                    unique_id_per_sentence += 1

        # ------------------------------------------------------------------------------------ #
        #  Divergences are contrasted with a seed equivalent                                   #
        # -------------------------------------------------------------------------------------#

        if sampling_mode == 'contrastive':

            # Seed equivalents and divergences are equally represented in training sets
            if o.balanced:

                pn = []
                counter = 0
                for sample in range(len(unlabeled)):
                    # If sample is not already included in the pool
                    if sample not in pn:
                        # If there is at least one divergent sentence for this index, add it
                        for div_type in divs_sentences:
                            if div_type[sample] != none_:
                                pn.append(sample)
                                counter += 1
                                break
                    if counter > total_sents:
                        break

                train_pn = pn[:o.num_training]
                dev_pn = pn[o.num_training:o.num_training + o.num_dev]
                test_pn = pn[o.num_training + o.num_dev:]

                # Prepare data for Cross-Entropy (CE)
                for ex in train_pn:
                    train.write(
                        '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))
                    unique_id_per_sentence += 2

                for ex in dev_pn:
                    dev.write(
                        '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))
                    unique_id_per_sentence += 2

                for ex in test_pn:

                    if o.multi_task:
                        multi_test_synthetic_learn2rank.write(
                            '0\t' + str(unique_id_per_sentence) + '\t' + str(unique_id_per_sentence + 1) + '\t' +
                            unlabeled[ex].rstrip() + \
                            '\t' + 'O ' * (len(unlabeled[ex].rstrip().split('\t')[0].split(' ')) - 1) + 'O\t' + \
                            'O ' * len(unlabeled[ex].rstrip().split('\t')[1].split(' ')) + '\n')

                    test_synthetic_learn2rank.write(
                        '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))
                    test_synthetic.write(
                        '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))
                    unique_id_per_sentence += 2

                # Span-version of training dataset
                for ex in train_pn:
                    tmp, tmp_span = [], []
                    for div_type, div_span in zip(divs_sentences, divs_sentences_spans):
                        if div_type[ex] != none_:
                            tmp.append(div_type[ex])
                            tmp_span.append(div_span[ex])
                    if len(tmp) > 1:
                        choose_ind = random.choice(range(len(tmp)))
                        choice = tmp[choose_ind]
                        choice_span = tmp_span[choose_ind]
                    else:
                        choice = tmp[0]
                        choice_span = tmp_span[0]
                    if o.learn_to_rank:
                        train_learn2rank.write('{0}\t{1}'.format(unlabeled[ex].rstrip(), choice))
                    if o.multi_task:
                        multi_train_learn2rank.write(
                            '{0}\t{1}\t{2}'.format(unlabeled[ex].rstrip(), choice.rstrip(), choice_span))
                    train.write(
                        '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1), choice))
                    unique_id_per_sentence += 2

                # Span-version of dev dataset
                for ex in dev_pn:
                    tmp, tmp_span = [], []
                    for div_type, div_span in zip(divs_sentences, divs_sentences_spans):
                        if div_type[ex] != none_:
                            tmp.append(div_type[ex])
                            tmp_span.append(div_span[ex])
                    if len(tmp) > 1:
                        choose_ind = random.choice(range(len(tmp)))
                        choice = tmp[choose_ind]
                        choice_span = tmp_span[choose_ind]
                    else:
                        choice = tmp[0]
                        choice_span = tmp_span[0]
                    if o.learn_to_rank:
                        dev_learn2rank.write('{0}\t{1}'.format(unlabeled[ex].rstrip(), choice))
                    if o.multi_task:
                        multi_dev_learn2rank.write(
                            '{0}\t{1}\t{2}'.format(unlabeled[ex].rstrip(), choice.rstrip(), choice_span))
                    dev.write(
                        '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1), choice))
                    unique_id_per_sentence += 2

                # Span-version of test dataset
                for ex in test_pn:
                    tmp, tmp_span = [], []
                    for div_type, div_span in zip(divs_sentences, divs_sentences_spans):
                        if div_type[ex] != none_:
                            tmp.append(div_type[ex])
                            tmp_span.append(div_span[ex])
                    if len(tmp) > 1:
                        choose_ind = random.choice(range(len(tmp)))
                        choice = tmp[choose_ind]
                        choice_span = tmp_span[choose_ind]
                    else:
                        choice = tmp[0]
                        choice_span = tmp_span[0]
                    test_synthetic_learn2rank.write(
                        '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1), choice))
                    if o.multi_task:
                        multi_test_synthetic_learn2rank.write(
                            '1\t{0}\t{1}\t{2}\t{3}'.format(str(unique_id_per_sentence), str(
                                unique_id_per_sentence + 1), choice.rstrip(), choice_span))
                    test_synthetic.write(
                        '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1), choice))
                    unique_id_per_sentence += 2

            # Seed equivalents and divergences are not balanced (not that divergence ranking is \
            # performed under unbalanced condition)
            else:

                pn = []
                counter = 0
                for sample in range(len(unlabeled)):
                    # If it is not already included
                    if sample not in pn:
                        # If there is at least one divergent sentence for this index, add it
                        for div_type in divs_sentences:
                            if div_type[sample] != none_:
                                pn.append(sample)
                                counter += 1
                                break
                    if counter > total_sents:
                        break

                train_pn = pn[:o.num_training]
                dev_pn = pn[o.num_training:o.num_training + o.num_dev]
                test_pn = pn[o.num_training + o.num_dev:]

                for ex in train_pn:
                    train.write(
                        '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))
                    unique_id_per_sentence += 2

                for ex in dev_pn:
                    dev.write(
                        '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))
                    unique_id_per_sentence += 2

                for ex in test_pn:
                    test_synthetic.write(
                        '0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))

                    if o.divergence_ranking:
                    	test_synthetic_divergence_ranking.write(
                        	'0\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                  unlabeled[
                                                      ex]))

                    if o.multi_task:
                        multi_test_synthetic_learn2rank.write(
                            '0\t' + str(unique_id_per_sentence) + '\t' + str(unique_id_per_sentence + 1) + '\t' +
                            unlabeled[ex].rstrip() + \
                            '\t' + 'O ' * (len(unlabeled[ex].rstrip().split('\t')[0].split(' ')) - 1) + 'O\t' + \
                            'O ' * len(unlabeled[ex].rstrip().split('\t')[1].split(' ')) + '\n')
                    unique_id_per_sentence += 2

                for div_id, div_type in enumerate(divs_sentences):

                    for ex in train_pn:

                        if div_type[ex] != none_:
                            train.write(
                                '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                          div_type[ex]))

                            if o.learn_to_rank:
                                train_learn2rank.write('{0}\t{1}'.format(unlabeled[ex].rstrip(), div_type[ex]))

                            if o.multi_task and not o.divergence_ranking:
                                multi_train_learn2rank.write(
                                    '{0}\t{1}\t{2}'.format(unlabeled[ex].rstrip(), div_type[ex].rstrip(),
                                                           divs_sentences_spans[div_id][ex]))

                            # Divergence ranking (sentence-only)
                            if o.divergence_ranking and not o.multi_task:

                                # If fine-grained divergence
                                if div_id == 2 or div_id == 3:

                                    train_divergence_ranking.write(
                                        '{0}\t{1}'.format(unlabeled[ex].rstrip(), div_type[ex]))

                                    # Choose randomly one a lexical substitution
                                    if divs_sentences[0][ex] != none_ and divs_sentences[1][ex] != none_:

                                        random_choice = random.randint(0, 1)
                                        train_divergence_ranking.write(
                                            '{0}\t{1}'.format(div_type[ex].rstrip(), divs_sentences[random_choice][ex]))

                                    elif divs_sentences[0][ex] != none_ and divs_sentences[1][ex] == none_:

                                        train_divergence_ranking.write(
                                            '{0}\t{1}'.format(div_type[ex].rstrip(), divs_sentences[0][ex]))

                                    elif divs_sentences[1][ex] != none_ and divs_sentences[0][ex] == none_:

                                        train_divergence_ranking.write(
                                            '{0}\t{1}'.format(div_type[ex].rstrip(), divs_sentences[1][ex]))

                                    else:

                                        continue

                            # Divergence ranking (multi task)
                            if o.multi_task and o.divergence_ranking and (div_id == 2 or div_id == 3):

                                multi_train_learn2rank.write(unlabeled[ex].rstrip() + '\t' +
                                                            div_type[ex].rstrip() + '\t' +
                                                          divs_sentences_spans[div_id][ex])

                                # Choose randomly a lexical substitution
                                if divs_sentences[0][ex] != none_ and divs_sentences[1][ex] != none_:

                                    random_choice = random.randint(0, 1)
                                    multi_train_learn2rank.write(
                                        '{0}\t{1}\t{2}'.format(div_type[ex].rstrip(),
                                                                    divs_sentences[random_choice][
                                                                        ex].rstrip(),
                                                                    divs_sentences_spans[random_choice][ex]))

                                elif divs_sentences[0][ex] != none_ and divs_sentences[1][ex] == none_:

                                    multi_train_learn2rank.write(
                                        '{0}\t{1}\t{2}'.format(div_type[ex].rstrip(),
                                                                    divs_sentences[0][ex].rstrip(),
                                                                    divs_sentences_spans[0][ex]))

                                elif divs_sentences[1][ex] != none_ and divs_sentences[0][ex] == none_:

                                    multi_train_learn2rank.write(
                                        '{0}\t{1}\t{2}'.format(div_type[ex].rstrip(),
                                                                    divs_sentences[1][ex].rstrip(),
                                                                    divs_sentences_spans[1][ex]))
                                else:
                                    continue

                            unique_id_per_sentence += 2

                    for ex in dev_pn:

                        if div_type[ex] != none_:

                            dev.write(
                                '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                          div_type[ex]))

                            if o.learn_to_rank:
                                dev_learn2rank.write('{0}\t{1}'.format(unlabeled[ex].rstrip(), div_type[ex]))

                            if o.multi_task and not o.divergence_ranking:
                                multi_dev_learn2rank.write(
                                    '{0}\t{1}\t{2}'.format(unlabeled[ex].rstrip(), div_type[ex].rstrip(),
                                                           divs_sentences_spans[div_id][ex]))

                            if o.divergence_ranking and not o.multi_task:

                                if div_id == 2 or div_id == 3:

                                    dev_divergence_ranking.write(
                                        '{0}\t{1}'.format(unlabeled[ex].rstrip(), div_type[ex]))

                                    # Choose randomly one of the lexical substitution
                                    if divs_sentences[0][ex] != none_ and divs_sentences[1][ex] != none_:

                                        random_choice = random.randint(0, 1)
                                        dev_divergence_ranking.write(
                                            '{0}\t{1}'.format(div_type[ex].rstrip(), divs_sentences[random_choice][ex]))

                                    elif divs_sentences[0][ex] != none_ and divs_sentences[1][ex] == none_:

                                        dev_divergence_ranking.write(
                                            '{0}\t{1}'.format(div_type[ex].rstrip(), divs_sentences[0][ex]))

                                    elif divs_sentences[1][ex] != none_ and divs_sentences[0][ex] == none_:

                                        dev_divergence_ranking.write(
                                            '{0}\t{1}'.format(div_type[ex].rstrip(), divs_sentences[1][ex]))
                                    else:
                                        continue

                            if o.multi_task and o.divergence_ranking:

                                if div_id == 2 or div_id == 3:

                                    multi_dev_learn2rank.write(unlabeled[ex].rstrip() + '\t' +
                                                               div_type[ex].rstrip() + '\t' +
                                                               divs_sentences_spans[div_id][ex])
                                    
                                    # Choose randomly one of the lexical substitution
                                    if divs_sentences[0][ex] != none_ and divs_sentences[1][ex] != none_:
                                        
                                        random_choice = random.randint(0, 1)
                                        multi_dev_learn2rank.write(
                                            div_type[ex].rstrip() + '\t' + divs_sentences[random_choice][
                                                ex].rstrip() + '\t' + divs_sentences_spans[random_choice][ex])

                                    elif divs_sentences[0][ex] != none_ and divs_sentences[1][ex] == none_:
                                        
                                        multi_dev_learn2rank.write(
                                            '{0}\t{1}\t{2}'.format(div_type[ex].rstrip(),
                                                                        divs_sentences[0][ex].rstrip(),
                                                                        divs_sentences_spans[0][ex]))

                                    elif divs_sentences[1][ex] != none_ and divs_sentences[0][ex] == none_:
                                        
                                        multi_dev_learn2rank.write(
                                            '{0}\t{1}\t{2}'.format(div_type[ex].rstrip(),
                                                                        divs_sentences[1][ex].rstrip(),
                                                                        divs_sentences_spans[1][ex]))
                                    else:
                                        continue
                            unique_id_per_sentence += 2

                    for ex in test_pn:
                        
                        if div_type[ex] != none_:
                            
                            test_synthetic.write(
                                '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(unique_id_per_sentence + 1),
                                                          div_type[ex]))
                            
                            if o.learn_to_rank:
                                
                                test_synthetic_learn2rank.write(
                                    '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(
                                        unique_id_per_sentence + 1), div_type[ex]))
                                
                            if o.multi_task:
                                
                                multi_test_synthetic_learn2rank.write(
                                    '1\t{0}\t{1}\t{2}\t{3}'.format(str(unique_id_per_sentence), str(
                                        unique_id_per_sentence + 1), div_type[ex].rstrip(),
                                                                   divs_sentences_spans[div_id][ex]))
                                
                            if o.divergence_ranking and not o.multi_task:
                                
                                test_synthetic_divergence_ranking.write(
                                    '1\t{0}\t{1}\t{2}'.format(str(unique_id_per_sentence), str(
                                        unique_id_per_sentence + 1), div_type[ex]))
                                
                            unique_id_per_sentence += 2


if __name__ == "__main__":
   
    # =================================================================================================================#
    # Note: Explanation of divergence ranking training data configuration                                              #
    # =================================================================================================================#
    #                                                                                                                  #
    # For token-level we want to expose the model to ALL possible divergences for fair comparison with other baselines #
    # including in the paper. For this reason starting from a seed equivalent we construct 4 contrastive pairs:        #
    #                                                                                                                  #
    #                   a) (seed equivalent)           VS      (generalization)                                        #
    #                   b) (seed equivalent)           VS      (particularization)                                     #
    #                   c) (generalization)            VS      (delete || replace)                                     #
    #                   d) (particularization)         VS      (delete || replace)                                     #
    #                                                                                                                  #
    # Thus starting from 5k seeds you should expect to create ~ 20K                                                    #
    # (in practice you will get less than that as not all seeds result in divergences of all 4 types)                  #
    #                                                                                                                  #
    # -----------------------------------------------------------------------------------------------------------------#
    #                                                                                                                  #
    # Same configuration holds for the sentence-level only divergence ranking experiment.                              #
    # =================================================================================================================#

    main()


