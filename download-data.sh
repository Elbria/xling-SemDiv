#!/usr/bin/env bash

#############################################################################
#                                                                           #
#                                                                           #
#        Detecting Fine-Grained Cross-Lingual Semantic Divergences          #
#               without Supervision by Learning To Rank                     #
#                                                                           #
#                              eleftheria                                   #
#                                                                           #
#                          ====  Step 2  ====                               #
#                                                                           #
#                  Download and prepare WikiMatrix data                     #
#                                                                           #
#                                                                           #
#############################################################################

# ==== Set variables
sample=50000        # number of sentences (top laser score) sample
                    # this is different that number of seeds

if [ $1 = 'en' ]; then
    non_en=$2
else
    non_en=$1
fi

# ==== Set directories
root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
data_dir=$root_dir/data
scripts_dir=$root_dir/source
software_dir=$root_dir/software
moses_dir=$software_dir/moses-scripts/tokenizer

# Create output data directory
mkdir -p $data_dir

# ==> Step 1: Download Wikimatrix data for input language pair
wikimatrix_path=$data_dir/wikimatrix
if [ ! -f $wikimatrix_path/WikiMatrix.$1-$2.tsv ] ; then
    echo '> Downloading wikimatrix data from AWS'
    mkdir -p $wikimatrix_path
    cd $wikimatrix_path
    wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.$1-$2.tsv.gz
    cd $data_dir
    echo '> done'
fi;

# ==> Step 2: Unzip TSV file
wikimatrix_file=$wikimatrix_path/WikiMatrix.$1-$2.tsv
if [ ! -f "$wikimatrix_file" ]; then
    echo '> Unzip Wikimatrix TSV file'
    cd $wikimatrix_path
    gzip -d $wikimatrix_path/WikiMatrix.$1-$2.tsv.gz
    cd $data_dir
    echo '> done'
fi;

# ==> Step 3: Filter out Wikimatrix bitexts that are obviously noisy
wikimatrix_file=$wikimatrix_path/WikiMatrix.$1-$2.tsv
if [ ! -f "${wikimatrix_file}.filtered" ]; then
    echo '> Clean Wikimatrix TSV file (heuristic-based filtering)'
    cd $wikimatrix_path
    python $scripts_dir/filter_noise.py \
					--input-corpus $wikimatrix_file \
				        --output-corpus ${wikimatrix_file}.filtered \
				        --src $1 \
					--tgt $2 
    cd $data_dir
    echo '> done'
fi;

# ==> Step 4: Sample top-scored sentences from Wikimatrix
wikimatrix_file=$wikimatrix_path/WikiMatrix.$1-$2.tsv.filtered
if [ ! -f "${wikimatrix_file}_sample_${sample}" ]; then
    echo '> Extract top scored sentence-pairs from filtered Wikimatrix'
    cd $wikimatrix_path
    head -n $sample $wikimatrix_file > ${wikimatrix_file}_sample_${sample}
    cd $data_dir
    echo '> done'
fi;

# ==> Step 5: Moses-preprocessing
wikimatrix_file=$wikimatrix_path/WikiMatrix.$1-$2.tsv.filtered_sample_${sample}
if [ ! -f "${wikimatrix_file}.moses.$1" ]; then
    cd $wikimatrix_path

    cut -f1 $wikimatrix_file > ${wikimatrix_file}.en
    cut -f2 $wikimatrix_file > ${wikimatrix_file}.$non_en

    echo '> Pre-process Wikimatrix using moses-scripts'
    # English language preprocessing
    # CAUTION: enabling -no-escape is extremenly crucial;
    #          performing HTML escaping on apostrophy & quotes outputs
    #          OOV tokens for the pre-trained language model and  
    #          oversplits them in redundant subwords
    cat ${wikimatrix_file}.en | \
				$moses_dir/replace-unicode-punctuation.perl | \
                                $moses_dir/normalize-punctuation.perl -l en | \
				$moded_dir/remove-non-printing-char.perl | \
				$moses_dir/tokenizer.perl -l en -no-escape \
				> $wikimatrix_file.moses.en

    # Non-English language preprocessing
    cat ${wikimatrix_file}.${non_en} | \
                                $moses_dir/replace-unicode-punctuation.perl | \
                                $moses_dir/normalize-punctuation.perl -l $non_en | \
                                $moded_dir/remove-non-printing-char.perl | \
			        $moses_dir/tokenizer.perl -l $non_en -no-escape \
			        > ${wikimatrix_file}.moses.$non_en
    cd $data_dir
    echo '> done'
fi;

# ==> Step 7: Create aligner configuration 
aligner_configuration=$scripts_dir/aligner.en-${non_en}.conf
if [ ! -f "$software_dir/berkeleyaligner/aligner.$1-$2.conf" ]; then
    echo '> Create aligner configuration'
    cd $root_dir
    python $scripts_dir/aligner_configuration.py \
                                      --input-corpus wikimatrix_aligned/WikiMatrix.$1-$2.tsv.filtered_sample_${sample}.moses \
                                      --src en \
                                      --tgt $non_en
    mv aligner.en-${non_en}.conf $software_dir/berkeleyaligner
    echo '> done'
fi;

# ==> Step 8: Align sample sentences of Wikimatrix data
aligned_wikimatrix=${wikimatrix_file}.moses.align
berkeley_dir=$software_dir/berkeleyaligner
if [ ! -f "${aligned_wikimatrix}" ]; then
    echo '> Align wikimatrix data using unsupervised Berkeley aligner'
    cd $berkeley_dir

    # Prepare training data
    mkdir -p wikimatrix_aligned
    cp ${wikimatrix_file}.moses.$1 wikimatrix_aligned
    cp ${wikimatrix_file}.moses.$2 wikimatrix_aligned

    export CONF=aligner.en-${non_en}.conf
    bash align $CONF
  
    cp output.en-$non_en/training.align ${wikimatrix_file}.moses.align
    cd $root_dir
    echo '> done'
fi; 

# ==> Step 9: Extract seed equivalents
seed_equivalents=${wikimatrix_file}.moses.seed
if [ ! -f "${seed_equivalents}" ]; then
    echo '> Extract seed equivalents'
    cd $data_dir

    paste ${wikimatrix_file}.moses.en ${wikimatrix_file}.moses.$non_en ${wikimatrix_file}.moses.align > \
	  ${wikimatrix_file}.moses.seed
    cd $root_dir
    echo '> done'
fi;

# ==> Step 10: Download nltk data (stopwords)
nltk_corpora_dir=$data_dir/nltk_data/corpora
if [ ! -d "${nltk_corpora_dir}" ]; then
    echo '> Download stopwords corpora from nltk data'
    mkdir -p $nltk_corpora_dir
    cd $nltk_corpora_dir
    wget http://www.nltk.org/nltk_data/packages/corpora/stopwords.zip
    unzip stopwords.zip
    cd $root_dir
    echo '> done'
fi;

# ==> Step 11: Download nltk data (punctuation)
nltk_punkt_dir=$data_dir/nltk_data/tokenizers
if [ ! -d "${nltk_punkt_dir}" ]; then
    echo '> Download punctutation corpora from nltk data'
    mkdir -p $nltk_punkt_dir
    cd $nltk_punkt_dir
    wget http://www.nltk.org/nltk_data/packages/tokenizers/punkt.zip
    unzip punkt.zip
    cd $root_dir
    echo '> done'
fi;

# ==> Step 12: Download nltk data (taggers)
nltk_tag_dir=$data_dir/nltk_data/taggers
if [ ! -d "${nltk_tag_dir}" ]; then
    echo '> Download taggers from nltk data'
    mkdir -p $nltk_tag_dir
    cd $nltk_tag_dir
    wget http://www.nltk.org/nltk_data/packages/taggers/averaged_perceptron_tagger.zip
    unzip averaged_perceptron_tagger.zip
    cd $root_dir
    echo '> done'
fi;

# ==> Step 13: Download nltk data (wordnet)
nltk_corpora_dir=$data_dir/nltk_data/corpora
if [ ! -d "${nltk_corpora_dir}/wordnet" ]; then
    echo '> Download wordnetfrom nltk data'
    mkdir -p $nltk_corpora_dir
    cd $nltk_corpora_dir
    wget http://www.nltk.org/nltk_data/packages/corpora/wordnet.zip
    unzip wordnet.zip
    cd $root_dir
    echo '> done'
fi;