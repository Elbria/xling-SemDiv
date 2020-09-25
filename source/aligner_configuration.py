#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Created on 29 March 2020

@author : eleftheria
"""

import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description='Define alignment configuration')
    parser.add_argument('--input-corpus', help='corpus to be filtered')
    parser.add_argument('--src', help='source language')
    parser.add_argument('--tgt', help='target language')
    parser.add_argument('--verbose', help="increase output verbosity", default=True)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args.src != 'en':
        foreignsuffix = args.src
    else:
        foreignsuffix = args.tgt

    output = open('aligner.' + args.src + '-' + args.tgt + '.conf','w')
    output.write('forwardModels\tMODEL1 HMM\nreverseModels\tMODEL1 HMM\nmode\tJOINT JOINT\niters\t2 2\n')
    output.write('execDir output.' + args.src + '-' + args.tgt)
    output.write('\ncreate\nsaveParams\ttrue\nnumThreads\t1\nmsPerLine\t10000\nalignTraining\n')
    output.write('foreignSuffix\t' + foreignsuffix + '\nenglishSuffix\ten\nlowercase\n')
    output.write('trainSources\t' + args.input_corpus)
    output.write('\nsentences\tMAX\ncompetitiveThresholding\n')

if __name__ == '__main__':
    main ()
