#!/usr/bin/env bash

#############################################################################
#                                                                           #
#                                                                           #
#        Detecting Fine-Grained Cross-Lingual Semantic Divergences          #
#               without Supervision by Learning To Rank                     #
#                                                                           #
#                              eleftheria                                   #
#                                                                           #
#                          ====  Step 1  ====                               #
#                                                                           #
#                    Download and prepare software                          #
#                                                                           #
#                                                                           #
#############################################################################

# ==== Set directory
root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
software_dir=$root_dir/software

# Step A: Installing moses scripts
# http://www.statmt.org/moses/
# https://github.com/moses-smt/mosesdecoder

moses_scripts_path=$software_dir/moses-scripts
if [ ! -d $moses_scripts_path ]; then
	cd $software_dir
	git clone https://github.com/moses-smt/mosesdecoder.git
	cd mosesdecoder
	git checkout 06f519d
	cd $software_dir
	mv mosesdecoder/scripts moses-scripts
	rm -rf mosesdecoder
fi;

# Step B: Place berkeley aligner under software
# https://code.google.com/archive/p/berkeleyaligner/

berkeley_path=$software_dir/berkeleyaligner
if [ ! -d $berkeley_path ]; then
        mv berkeleyaligner $software_dir
fi;

# Step C: Install Huggingface transformers
# https://github.com/huggingface/transformers

transformers=$software_dir/transformers
if [ ! -d $transformers/src/transformers.egg-info ]; then
	cd $transformers
	pip install -e .
	cd $root_dir
	pip install pytorch-pretrained-bert==0.6.2 
fi;