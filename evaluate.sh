#!/usr/bin/env bash

#############################################################################
#                                                                           #
#                                                                           #
#        Detecting Fine-Grained Cross-Lingual Semantic Divergences          #
#               without Supervision by Learning To Rank                     #
#                                                                           #
#                              eleftheria                                   #
#                                                                           #
#                          ====  Step 4  ====                               #
#                                                                           #
#                         Evaluation on REFreSD                             #
#                                                                           #
#                                                                           #
#############################################################################

##############################################################################


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/clip-scratch/ebriakou/anaconda3/lib

corpus=WikiMatrix                                   #   Corpus from which seed equivalents are extracted
sampling_method=contrastive_divergence_ranking      #   Sampling method for extracting divergent examples from seeds
size=50000                                          #   Number of seeds sampled from original corpus
src=en                                              #   Source language (language code)
tgt=fr                                              #   Target language (language code)
divergent_list=rdpg                                 #   List of divergences (e.g, 'rd' if divergences include
                                                    #                   phrase replacement and subtree deletion)

#############################################################################

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
data_dir=$root_dir/data
scripts_dir=$root_dir/source

exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
output_dir=$root_dir/trained_bert/$exp_identifier
REFreSD_dir=$root_dir/REFreSD_no_normal/REFreSD_for_huggingface
model=bert-base-multilingual-cased

################################################################################
#                         Synthetic test evaluation                            #
################################################################################

set_=test_synthetic

python $scripts_dir/run_div_margin.py \
        	--node $SLURM_NODELIST \
        	--model_type bert_margin \
        	--model_name_or_path $model \
        	--task_name SemDiv \
        	--do_eval   \
        	--best_checkpoint \
        	--evaluation_set $set_ \
        	--data_dir $data_dir/   \
        	--output_dir $output_dir \
        	--synth_data_dir $data_dir/ \
        	--overwrite_cache

#################################################################################
#              REFreSD evaluation --- Divergence VS Equivalence                 #
#################################################################################

set_=test

python $scripts_dir/run_div_margin.py \
        	--node $SLURM_NODELIST \
        	--model_type bert_margin \
        	--model_name_or_path $model \
        	--task_name SemDiv \
        	--do_eval   \
        	--best_checkpoint \
        	--evaluation_set $set_ \
        	--data_dir $data_dir/   \
        	--output_dir $output_dir \
        	--synth_data_dir $REFreSD_dir/ \
        	--overwrite_cache

#####################################################################################
#          REFreSD evaluation --- Unrelated VS No meaning difference                #
#####################################################################################

set_=unrelated

python $scripts_dir/run_div_margin.py \
        	--node $SLURM_NODELIST \
        	--model_type bert_margin \
        	--model_name_or_path $model \
       		--task_name SemDiv \
        	--do_eval   \
        	--best_checkpoint \
        	--evaluation_set $set_ \
        	--data_dir $data_dir/   \
        	--output_dir $output_dir \
        	--synth_data_dir $REFreSD_dir/ \
        	--overwrite_cache

########################################################################################
#          REFreSD evaluation -- Some meaning difference VS No meaning difference      #
########################################################################################

set_=some_meaning_difference
python $scripts_dir/run_div_margin.py \
        	--node $SLURM_NODELIST \
        	--model_type bert_margin \
        	--model_name_or_path $model \
        	--task_name SemDiv \
        	--do_eval   \
        	--best_checkpoint \
        	--evaluation_set $set_ \
       	 	--data_dir $data_dir/   \
        	--output_dir $output_dir \
        	--synth_data_dir $REFreSD_dir/ \
        	--overwrite_cache

########################################################################################
#                           Print results                                              #
########################################################################################

echo '> Test synthetic:'
python $scripts_dir/sentence_evaluation.py --dict_dir $output_dir/ --set_ test_synthetic
echo '> REFreSD (Divergence vs Equivalence):'
python $scripts_dir/sentence_evaluation.py --dict_dir $output_dir/ --set_ test
echo '> REFreSD (Unrelated vs No meaning difference):'
python $scripts_dir/sentence_evaluation.py --dict_dir $output_dir/ --set_ unrelated
echo '> REFreSD (Some meaning difference vs No meaning difference):'
python $scripts_dir/sentence_evaluation.py --dict_dir $output_dir/ --set_ some_meaning_difference