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
#                 Fine-tune divergnetmBERT for token-tagging                #
#                                                                           #
#                                                                           #
#############################################################################

#############################################################################

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/clip-scratch/ebriakou/anaconda3/lib

corpus=WikiMatrix                           #   Corpus from which seed equivalents are extracted
sampling_method=contrastive_multi_hard      #   Sampling method for extracting divergent examples from seeds
size=50000                                  #   Number of seeds sampled from original corpus
src=$1                                      #   Source language (language code)
tgt=$2                                      #   Target language (language code)
divergent_list=rdpg                         #   List of divergences (e.g, 'rd' if divergences include
                                            #                        phrase replacement and subtree deletion)
lr=2e-5                                     #   Learning rate
batch_size=16                               #   Training batch size
epochs=5                                    #   Number of training epochs
margin=5                                    #   Margin Hyperparameter
alpha=1                                     #   Weight losses for mult-itask

############################################################################

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
data_dir=$root_dir/data
scripts_dir=$root_dir/source

exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
output_dir=$root_dir/trained_bert/$exp_identifier
model=bert-base-multilingual-cased

##############################################################################

if [[ ! -e $output_dir ]]; then
    mkdir -p $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

python ${scripts_dir}/run_div_multi.py \
                            --node $SLURM_NODELIST \
                            --model_type SemDivMulti \
                            --data_dir ${data_dir} \
                            --labels ${data_dir}/labels.txt \
                            --task_name SemDiv \
                            --model_name_or_path ${model} \
                            --margin ${margin} \
                            --max_seq_length 128   \
                            --output_dir $output_dir \
                            --num_train_epochs ${epochs} \
                            --per_gpu_train_batch_size $batch_size \
                            --save_steps 100 \
                            --logging_steps 100 \
                            --do_train \
                            --do_eval \
                            --alpha ${alpha} \
                            --learning_rate ${lr}  \
                            --do_predict \
                            --overwrite_cache \
                            --synth_data_dir ${data_dir}/ \
                            --overwrite_output_dir