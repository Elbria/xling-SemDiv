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
#                 Fine-tune divergnetmBERT using CE (baseline)              #
#                                                                           #
#                                                                           #
#############################################################################

##############################################################################

corpus=WikiMatrix               #   Corpus from which seed equivalents are extracted
sampling_method=contrastive     #   Sampling method for extracting divergent examples from seeds
size=50000                      #   Number of seeds sampled from original corpus
src=$1                          #   Source language (language code)
tgt=$2                          #   Target language (language code)
divergent_list=rdpg             #   List of divergences (e.g, 'rd' if divergences include
                                # 				 phrase replacement and subtree deletion)
lr=2e-5                         #   Learning rate
batch_size=32                   #   Training batch size
epochs=3                        #   Number of training epochs

################################################################################

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
data_dir=$root_dir/data
scripts_dir=$root_dir/source

exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
data_dir=$root_dir/${exp_identifier}
output_dir=$root_dir/trained_bert/$exp_identifier
model=bert-base-multilingual-cased

################################################################################

# Create output directory if not exist
if [[ ! -e $output_dir ]]; then
    mkdir -p $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

python ${scripts_dir}/run_div_cross.py \
                    --node $SLURM_NODELIST \
                    --model_type bert \
                    --model_name_or_path ${model} \
                    --task_name SemDivCross \
                    --do_eval   \
                    --do_train \
                    --evaluate_during_training \
                    --evaluate_on_training \
                    --data_dir ${data_dir}/   \
                    --max_seq_length 128   \
                    --save_steps 100 \
                    --logging_steps 100 \
                    --per_gpu_train_batch_size=${batch_size}   \
                    --learning_rate ${lr}  \
                    --num_train_epochs ${epochs}  \
                    --output_dir ${output_dir} \
                    --overwrite_cache \
                    --synth_data_dir ${data_dir}/