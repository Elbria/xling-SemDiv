#!/usr/bin/env python3

import argparse
import glob
import operator
import shutil
import os
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dict_dir", help='Folder containing dictionaries')
    parser.add_argument("--set_", help='input set for evaluation')
    args = parser.parse_args() 

    if args.set_ == 'test_synthetic' or args.set_ == 'test' or args.set_ == 'unrelated' \
	or args.set_ == 'some_meaning_difference':

        if args.set_ == 'test_synthetic':
            suffix='best_test_synthetic_preds_gt.txt'
        elif args.set_ == 'test':
            suffix='best_test_preds_gt.txt'
        elif args.set_ == 'unrelated':
            suffix ='best_unrelated_preds_gt.txt'
        elif args.set_ == 'some_meaning_difference':
            suffix ='best_some_meaning_difference_preds_gt.txt'
        
        preds, gold, res = [], [], []
            
        with open(args.dict_dir + suffix, 'r') as file_:
            file_ = file_.readlines()
            for line in file_:
            	line = line.strip().split('\t')
            	preds.append(int(line[0]))
            	gold.append(int(line[1]))
                
        precisions_per_class = precision_score(gold, preds, average=None)
        recall_per_class = recall_score(gold, preds, average=None)
        f1_per_class = f1_score(gold, preds, average=None)
        precision_weighted = precision_score(gold, preds, average='weighted')
        recall_weighted = recall_score(gold, preds, average='weighted')
        f1_micro = f1_score(gold, preds, average='micro')
        f1_weighted = f1_score(gold, preds, average='weighted')

    re =  [precisions_per_class[0], recall_per_class[0], f1_per_class[0], precisions_per_class[1], recall_per_class[1], f1_per_class[1], f1_weighted]      
    return re

if __name__ == "__main__":
   result = main()
   sys.exit(result)
