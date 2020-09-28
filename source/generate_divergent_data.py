# -*- coding: utf-8 -*-

"""
Created on 29 March 2020

@author: eleftheria
"""

import os
import io
import sys
import argparse
import nltk
import itertools

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils import pos_phrases_ngrams
from synthetic_divergences_fine_grained import synthetic_divergences

nltk.data.path.append('data/nltk_data/')
tokenizer = RegexpTokenizer(r'\w+')
flatten = itertools.chain.from_iterable
stopwords_list = stopwords.words('english')

global none_
none_='None\n'


def main():
    parser = argparse.ArgumentParser(description='Synthetic divergent data creation')
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--data', help='input positive examples')
    parser.add_argument('--output', help='output directory of synthetic training data', default='synthetic')
    parser.add_argument('--mode',
                        help='how data examples are generated (p: parallel, u:uneven, i:insert, r:replace d:delete',
                        default='i')
    parser.add_argument('--pretrained_bert', help='pretrained bert', default='bert-base-cased')
    parser.add_argument('--bert_local_cache', help='path to local directory where pretrained bert is saved')

    o = parser.parse_args()
    d = synthetic_divergences()

    # Create directory for bert local cache
    if not os.path.exists(o.bert_local_cache):
        os.makedirs(o.bert_local_cache)

    pos_to_wrd = defaultdict(list)
    indices = []
    with io.open(o.data, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        i = 0
        n_total = 0
        for line in f:
            n_total += 1
            if n_total % 100000 == 0:
                if n_total % 1000000 == 0:
                    sys.stderr.write(str(n_total))
                else:
                    sys.stderr.write(".")

            indices.append(i)
            tok = line.strip('\n').split("\t")
            src = tok.pop(0).strip().split(' ')
            tgt = tok.pop(0).strip().split(' ')
            ali = tok.pop(0).strip().split(' ')
            src = word_tokenize(' '.join(src))
            tagged_sent = nltk.pos_tag(src)
            words, tags = zip(*tagged_sent)
            pos = list(tags)
            d.add(src, tgt, pos, ali)
            pos_phrases_ngrams(src, pos, pos_to_wrd)
            i += 1

    # Configure write mode and output files
    write_mode = 'w'

    output_path = os.path.join(o.output, 'from_{0}'.format(str(o.data.split('/')[-1])))

    # Create output directories
    try:
        os.makedirs(output_path)
    except FileExistsError:
        sys.stderr.write('Warning: Output file already exists\n')

    if 'g' in o.mode:
        lm_model = BertForMaskedLM.from_pretrained(o.pretrained_bert, cache_dir=o.bert_local_cache)
        lm_tokenizer = BertTokenizer.from_pretrained(o.pretrained_bert, cache_dir=o.bert_local_cache)
        output_g = open(os.path.join(output_path, 'generalization'), write_mode)
        output_g_span = open(os.path.join(output_path, 'generalization.span'), write_mode)
    if 'p' in o.mode:
        lm_model = BertForMaskedLM.from_pretrained(o.pretrained_bert, cache_dir=o.bert_local_cache)
        lm_tokenizer = BertTokenizer.from_pretrained(o.pretrained_bert, cache_dir=o.bert_local_cache)
        output_p = open(os.path.join(output_path, 'particularization'), write_mode)
        output_p_span = open(os.path.join(output_path, 'particularization.span'), write_mode)
    if 'i' in o.mode:
        output_i = open(os.path.join(output_path, 'insert'), write_mode)
        output_i_span = open(os.path.join(output_path, 'insert.span'), write_mode)
    if 'u' in o.mode:
        output_u = open(os.path.join(output_path, 'uneven'), write_mode)
        output_u_span = open(os.path.join(output_path, 'uneven.span'), write_mode)
    if 'd' in o.mode:
        output_d = open(os.path.join(output_path, 'delete'), write_mode)
        output_d_span = open(os.path.join(output_path, 'delete.span'), write_mode)
    if 'r' in o.mode:
        output_r = open(os.path.join(output_path, 'replace'), write_mode)
        output_r_span = open(os.path.join(output_path, 'replace.span'), write_mode)

    for i in indices:

        # Insert sentence
        if 'i' in o.mode:
            synthetic_pair = d.insert_pair(i, o)
            if synthetic_pair:
                output_i.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[0]), ' '.join(synthetic_pair[1])))
                output_i_span.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[2]), ' '.join(synthetic_pair[3])))

            else:
                output_i.write(none_)
                output_i_span.write(none_)

        # Random pairing of sentences
        if 'u' in o.mode:
            synthetic_pair = d.uneven_pair(i, o)
            if synthetic_pair:
                output_u.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[0]), ' '.join(synthetic_pair[1])))
                output_u_span.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[2]), ' '.join(synthetic_pair[3])))
            else:
                output_u.write(none_)
                output_u_span.write(none_)

        # Create lexical substitution (generalization) instance
        if 'g' in o.mode:
            synthetic_pair = d.generalization_pair(i, o, lm_model, lm_tokenizer)
            if synthetic_pair:
                output_g.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[0]), ' '.join(synthetic_pair[1])))
                output_g_span.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[2]), ' '.join(synthetic_pair[3])))
            else:
                output_g.write(none_)
                output_g_span.write(none_)

        # Create lexical substitution (particularization) instance
        if 'p' in o.mode:
            synthetic_pair = d.particularization_pair(i, o, lm_model, lm_tokenizer)
            if synthetic_pair:
                output_p.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[0]), ' '.join(synthetic_pair[1])))
                output_p_span.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[2]), ' '.join(synthetic_pair[3])))
            else:
                output_p.write(none_)
                output_p_span.write(none_)

        # Create subtree deletion instance
        if 'd' in o.mode:
            synthetic_pair = d.delete_pair(i, o)
            if synthetic_pair:
                output_d.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[0]), ' '.join(synthetic_pair[1])))
                output_d_span.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[2]), ' '.join(synthetic_pair[3])))
            else:
                output_d.write(none_)
                output_d_span.write(none_)

        # Create phrase replacement instance
        if 'r' in o.mode:
            synthetic_pair = d.replace_pair(i, o, pos_to_wrd)
            if synthetic_pair:
                output_r.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[0]), ' '.join(synthetic_pair[1])))
                output_r_span.write('{0}\t{1}\n'.format(' '.join(synthetic_pair[2]), ' '.join(synthetic_pair[3])))
            else:
                output_r.write(none_)
                output_r_span.write(none_)


if __name__ == "__main__":
    main()
