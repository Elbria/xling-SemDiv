#!/usr/bin/env bash

#############################################################################
#                                                                           #
#                                                                           #
#        Detecting Fine-Grained Cross-Lingual Semantic Divergences          #
#               without Supervision by Learning To Rank                     #
#                                                                           #
#                              eleftheria                                   #
#                                                                           #
#                          ====  Step 3  ====                               #
#                                                                           #
#                     Mimick synthetic divergences                          #
#                                                                           #
#                                                                           #
#############################################################################

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/clip-scratch/ebriakou/anaconda3/lib

# ==== Set directories
root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
data_dir=$root_dir/data
scripts_dir=$root_dir/source
seed_file=WikiMatrix.$1-$2.tsv.filtered_sample_50000.moses.seed
seeds=$data_dir/wikimatrix/$seed_file


# === Set dependencies
export NLTK_DATA=$data_dir/nltk_data
python -m spacy download 'en_core_web_sm'

echo $'\n> Generate synthetic divergences from seed equivalents'
for process in i u d r g p; do 

		echo $'\n--- Divergent type: '$process$' ---\n'

		python $scripts_dir/generate_divergent_data.py \
						--mode $process \
						--data $seeds \
						--output synthetic \
						--bert_local_cache pretrained_bert \
						--pretrained_bert "bert-base-cased" \
						#--debug

done		

cut -f 1-2 $seeds > ${seeds}_exclude_align

echo $'\n> Prepare divergence ranking for sentence-level divergentmBERT'
python $scripts_dir/build_bert_training_data.py \
                    --path_to_unlabeled ${seeds}_exclude_align \
                    --path_to_divergences $root_dir/synthetic/from_$seed_file \
                    --divergent_list rdpg \
                    --contrastive \
                    --learn-to-rank \
                    --divergence-ranking

echo $'\n> Prepare divergence ranking for multi-task divergentmBERT'
python $scripts_dir/build_bert_training_data.py \
                    --path_to_unlabeled ${seeds}_exclude_align \
                    --path_to_divergences $root_dir/synthetic/from_$seed_file \
                    --divergent_list rdpg \
                    --contrastive \
                    --learn-to-rank \
                    --divergence-ranking \
                    --multi-task

label_file=$root_dir/for_divergentmBERT/from_$seed_file/contrastive_multi_hard/rdpg/labels.txt
echo "O" > $label_file
echo "D" >> $label_file
