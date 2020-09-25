#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 29 March 2020

author : eleftheria
"""

import numpy as np
from tqdm import tqdm
import argparse
import logging


def levenshtein(seq1, seq2):
    """

    Compute Levenshtein distance between two sequences of characters
    :param seq1: first sequence
    :param seq2: second sequence
    :return: Levenshtein distance

    """

    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def perc_numeric(seq):
    """

    Compute what percentage of a sentence contains numerical values
    :param seq: input sequence
    :return: percentage of sentence containing numbers

    """
    counter = 0
    for word in seq:
        word = str(word.encode('utf-8'))
        if word.isnumeric():
            counter += 1
    return counter / len(seq)


def main():
    parser = argparse.ArgumentParser(description='Filtering of parallel corpora')
    parser.add_argument('--input-corpus', help='corpus to be filtered')
    parser.add_argument('--output-corpus', help='filtered corpus')
    parser.add_argument('--similarity', help='flag that tests whether a similarity score is provided for bitexts',
                        default=True)
    parser.add_argument('--threshold', help='similarity threshold -- assumes ordered corpus', default=None)
    parser.add_argument('--src', help='source languaage')
    parser.add_argument('--tgt', help='target language')
    parser.add_argument('--verbose', help="increase output verbosity", default=True)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open(args.input_corpus, 'r') as file:
        lines = file.readlines()

        output_bitext = open(args.output_corpus, 'w')
        with tqdm(total=len(lines)) as pbar:
            for line in lines:
                line = line.split('\t')

                # Split differently based on the availability of similarity scores for bitexts
                if args.similarity:
                    sim = line[0]
                    src_sent = line[1].split()
                    tgt_sent = line[2].split()
                else:
                    src_sent = line[0].split()
                    tgt_sent = line[1].split()

                # Find whether English is the source or target language
                if args.src == 'en':
                    en = src_sent
                else:
                    en = tgt_sent
                pbar.update(1)

                # Filtered sentences should not be be to short or to long
                # That holds for both sides
                if 10 < len(src_sent) < 50 and (10 < len(tgt_sent) < 50) or args.tgt == 'zh':

                    # If the Levenshtein distance between two sentences is to small continue
                    if levenshtein(src_sent, tgt_sent) < 20:
                        continue
                    else:
                        # If numbers (dates) are a small percentage of the sentece pairs include
                        if perc_numeric(en) < 0.15:

                            # Write english first
                            if args.src == 'en':
                                output_bitext.write('{0}\t{1}\n'.format(' '.join(src_sent), ' '.join(tgt_sent)))
                            else:
                                output_bitext.write('{0}\t{1}\n'.format(' '.join(tgt_sent), ' '.join(src_sent)))
                        else:
                            continue
                else:
                    continue

    # Close filtered file
    output_bitext.close()


if __name__ == '__main__':
    main()
