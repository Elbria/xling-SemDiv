# -*- coding: utf-8 -*-

"""
Created on 29 March 2020

@author: eleftheria
"""

from nltk import ngrams
import numpy as np

bert_header='Quality\t#1ID\t#2ID\t#1String\t#2String\n'

def _recurse_all_hyponyms(synset, all_hyponyms):
    synset_hyponyms = synset.hyponyms()
    if synset_hyponyms:
        all_hyponyms += synset_hyponyms
        for hyponym in synset_hyponyms:
            _recurse_all_hyponyms(hyponym, all_hyponyms)


def _recurse_leaf_hyponyms(synset, leaf_hyponyms):
    synset_hyponyms = synset.hyponyms()
    if synset_hyponyms:
        for hyponym in synset_hyponyms:
            _recurse_all_hyponyms(hyponym, leaf_hyponyms)


def leaf_hyponyms(synset):
    """

    Get the set of leaf nodes from the tree of hyponyms under the synset

    """

    hyponyms = []
    _recurse_leaf_hyponyms(synset, hyponyms)
    return set(hyponyms)


def pos_phrases_ngrams(src, src_pos, list_ngrams):
    for n in range(2, 10):
        ngram_tok = [a for a in ngrams(src, n)]
        ngram_pos = [a for a in ngrams(src_pos, n)]
        for (tok, pos) in zip(ngram_tok, ngram_pos):
            list_ngrams[' '.join(pos)].append(' '.join(tok))
    return list_ngrams

def load_laser_embs(embs):
    """
    Load laser embeddings into numpy matrix
    :param embs:
    :return:
    """
    dim = 1024
    X = np.fromfile(embs, dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    return X

def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2

def levenshtein_distance(s1, s2):
    s1 = ' '.join(s1)
    s2 = ' '.join(s2)
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def perc_numeric(seq):
    '''

    Compute what percentage of a sentence contains numerical values
    :param seq: input sequence
    :return: percentage of sentence containing numbers

    '''
    seq = seq.split()
    counter = 0
    for word in seq:
        #word = str(word.encode('utf-8'))
        if word.isnumeric():
            counter += 1
    return float(counter/len(seq))

def alignments2dic(ali):
    '''

    Convert a list to alignments to dict
    :param ali: list of alignments (assumption src-tgt)
    :return: dictionary mapping each source to target
    '''
    alignment_mappings = {}

    for alignment in ali:

        alignment = alignment.split('-')
        print(ali)
        alignment_mappings[int(alignment[1])] = int(alignment[0])

    return alignment_mappings

def flatten_tree(tree):
    return ''.join([token.text_with_ws for token in list(tree)]).strip()

def sublist_indices(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll]==sl:
            return int(ind), int(ind+sll)


divergent_mappings = {
                        'u': 'uneven',
                        'i': 'insert',
                        'r': 'replace',
                        'd': 'delete',
                        'g': 'generalization',
                        'p': 'particularization'
                      }
