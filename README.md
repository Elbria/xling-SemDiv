# Detecting Fine-Grained Cross-Lingual Semantic Divergences without Supervision by Learning to Rank

This repository contains code and data from the EMNLP 2020 paper that can be found [here]()!

This work shows that explicitly considering diverse types of semantic divergences in bilingual text benefits both the
annotation and prediction of cross-lingual semantic divergences. We release the **R**ationalized
**E**nglish-**Fre**nch **S**emantic **D**ivergences corpus, **REF**re**SD**, based on a novel divergence annotation protocol that exploits
rationales to improve annotator agreement. We introduce Divergent mBERT, a BERT-based model that detects fine-grained
semantic divergences without supervision by learning to rank synthetic divergences of varying granularity.
Evaluation on REFRESD shows that our model distinguishes semantically equivalent from divergent examples much better
than the LASER baseline, and that unsupervised token-level divergence tagging offers promise to refine distinctions
among divergent instances.


## Table of contents

- [Setup](#setup)
- [Mimicking semantic divergences](#Mimicking-semantic-divergences)
- [Learning to rank semantic divergences](#divergence-ranking)
- [REFreSD evaluation](#refresd-evaluation)
- [Contact](#contact)

## Setup

1. Create a dedicated virtual environment (here we use [anaconda](https://anaconda.org)) for the project & install requirements:

    ```
    conda create -n semdiv python=3.6
    conda activate semdiv
    conda install --file requirements.txt
    ```

2. Download [Berkeley aligner](https://code.google.com/archive/p/berkeleyaligner/downloads) (**berkeleyaligner_unsupervised-2.1.tar.gz**); 
unzip the compressed file under the root directory of the project (a file named ```berkeleyaligner``` should appear under current root before you move to the next step).

3. Run the following script to download and install the required software: 

    ```bash
    bash setup.sh
    ```

## Mimicking semantic divergences

Download and preprocess WikiMatrix data for source-target language pair.
The scripts takes as arguments the ISO codes of the desired language pair.
The language pair must be in alphabetical order, e.g. "de-en" and not "en-de". 
The list of available bitexts and their sizes are given in the file [list_of_bitexts.txt](https://github.com/facebookresearch/LASER/blob/master/tasks/WikiMatrix/list_of_bitexts.txt). 
For example, to download the English-French parallel corpus run:

    
    bash download-data.sh en fr 
    

Generate synthetic divergences from seed equivalents:

    bash generate-divergences.sh en fr 
    
<p align="center">
    <img  src="static/sem_div_video_ele.gif" width="600" height="400" />
</p>

## Divergence ranking

Fine-tune divergentmBERT (implemented on top of the [HuggingFace transformers](https://github.com/huggingface/transformers) library) 
via learning to rank contrastive divergences of increased granularity. We learn to rank pairs of close divergence types: equivalent vs. lexical substitution (were we mimic both generalization **and** particularization), lexical substitution vs. phrase replacement or substree deletion yielding four contrastive pairs.


    bash train-margin.sh en fr 

Train token-tagger using multi-task learning:
minimize token-level cross entropy loss and divergence ranking
margin-based loss on synthetic contrastive divergences.

    bash train-multi.sh en fr  

## REFreSD evaluation

Sentence-level and token-level evaluation on REFreSD

    bash evaluate.sh
    bash predic-token-tags.sh
    
Predictions are found under the output directory of the trained models. For the token-tagger the output file consists of 8 tab-separated columns. An example of the divergentmBERT output is shown below:

```

Binary prediction: 1
Score: 0.08358259
English input   :  Complaints of this sort were so loud and frequent that the governors of Greenwich Hospital to whom the lighthouse belonged sent Sir John Thomson to examine and make arrangements on the subject .
French input:  Les plaintes restèrent fréquentes et les gouverneurs de l' hôpital de Greenwich , à qui appartenait le phare , envoyèrent sir John Thomson pour remédier au problème .
REFreSD (en) : O D D D D D D D O O O O O O O O O O O O O O O O D D D D D D D D D     
REFreSD (fr) : O O D O D O O O O O O O O O O O O O O O O O O D D D D D
Predict (en) : O O O D O O O O O O O O O O O O O O O O O O O O O D O D D O O D O  
Predict (fr) : O O O O O O O O O O O O O O O O O O O O O O O O D O D O
```

## Contact

If you use any contents of this repository, please cite us. For any questions, write to ebriakou@cs.umd.edu.

```
@inproceedings{briakou-carpuat-2020-detecting,
    title = "Detecting Fine-Grained Cross-Lingual Semantic Divergences without Supervision by Learning to Rank",
    author = "Briakou, Eleftheria and Carpuat, Marine",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.121",
    pages = "1563--1580",
}
```
