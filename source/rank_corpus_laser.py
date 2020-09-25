# -*- coding: utf-8 -*-

"""
Created on 29 March 2020

@author: eleftheria
"""

import argparse
import os
import tqdm

from utils import load_laser_embs
from utils import similarity
from utils import levenshtein_distance
from utils import perc_numeric


def main():
    parser = argparse.ArgumentParser(description='Rank corpus based on laser cosine distance')
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--src_sents', help='source sentences')
    parser.add_argument('--tgt_sents', help='target sentences')
    parser.add_argument('--src_embs', help='laser embeddings for source sentences')
    parser.add_argument('--tgt_embs', help='laser embeddings for target sentences')
    parser.add_argument('--output_path', help='path to ranked corpus')
    parser.add_argument('--output_corpus', help='path to ranked corpus')
    o = parser.parse_args()

    try:
        os.makedirs(o.output_path)
    except FileExistsError:
        # directory already exists
        pass

    output_corpus = os.path.join(o.output_path, o.output_corpus)


    src_emb = load_laser_embs(o.src_embs)
    tgt_emb = load_laser_embs(o.tgt_embs)

    sim = []
    for v1,v2 in zip(src_emb, tgt_emb):
        sim.append(similarity(v1,v2))

    sim_sorted = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)

    with open(output_corpus, 'w') as output, open(o.src_sents, 'r') as src, open(o.tgt_sents, 'r') as tgt:
        src = src.readlines()
        tgt = tgt.readlines()

        pbar = tqdm.tqdm(total=len(src))

        for similarity_index in sim_sorted:
            pbar.update(1)
            src_sentence = src[similarity_index].strip()
            tgt_sentence = tgt[similarity_index].strip()

            # Exclude almost identical sentences or too short sentence-pairs;
            # exclude sentences containing a lot of numbers
            if levenshtein_distance(src_sentence, tgt_sentence) < 30 or perc_numeric(src_sentence)>0.3:
                continue

            output.write('{0}\t{1}'.format(src[similarity_index].strip(), tgt[similarity_index]))

    output.close()


if __name__ == "__main__":
    main()
