# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert). """

# Adapted from Hugging Face library to fine-tune on synthetic divergences using multi-task loss

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_processors as processors
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_multi_div import convert_examples_to_features, get_labels, read_examples_from_file

from transformers import AdamW, WarmupLinearSchedule
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceTokenClassificationMarginLoss, BertTokenizer

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )),
    ())

MODEL_CLASSES = {
    "semdivmulti": (BertConfig, BertForSequenceTokenClassificationMarginLoss, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids_eq'  : batch[0],
                      'input_mask_eq' : batch[1],
                      'input_ids_dv'  : batch[3],
                      'input_mask_dv' : batch[4],
                      'label_ids_dv'  : batch[6],
                      'label'         : batch[7]}
            if args.model_type != 'distilbert':
                inputs['segment_ids_eq'] = batch[2] 
                inputs['segment_ids_dv'] = batch[5] 
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, checkpoint, labels, pad_token_label_id, mode, sentence_eval=True, token_eval=False,  prefix=""):

    if mode != 'dev' or mode != 'test_synth':
        sentence_eval = True
        token_eval = True

    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_sent, preds_tok, sent_label_ids, tok_label_ids = None, None, None, None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if mode == 'dev':
            	inputs = {'input_ids_eq'  : batch[0],
                	  'input_mask_eq' : batch[1],
                      	  'input_ids_dv'  : batch[3],
                      	  'input_mask_dv' : batch[4],
                      	  'label_ids_dv'  : batch[6],
                      	  'label'         : batch[7]}
            	if args.model_type != 'distilbert':
                	inputs['segment_ids_eq'] = batch[2]
                	inputs['segment_ids_dv'] = batch[5]
            else:
            	inputs = {'input_ids_eq'  : None,
                          'input_mask_eq' : None,
                          'input_ids_dv'  : batch[0],
                          'input_mask_dv' : batch[1],
                          'label_ids_dv'  : batch[3],
                          'label'         : batch[4]}
            if args.model_type != 'distilbert':
                    inputs['segment_ids_eq'] = None 
                    inputs['segment_ids_dv'] = batch[2]

            outputs = model(**inputs)

            if mode == 'dev':
                tmp_eval_loss, logits_first, logits_second, logits_sec_seq = outputs[:4]
                eval_loss += tmp_eval_loss.mean().item()           
            else:
                logits_second, logits_sec_seq = outputs[:2]

        nb_eval_steps += 1

        if mode != 'dev':
            # Get sentence-level predictions
            if preds_sent is None:
                preds_sent = logits_second.detach().cpu().numpy()
                sent_label_ids = inputs["label"].detach().cpu().numpy()
            else:
                preds_sent = np.append(preds_sent, logits_second.detach().cpu().numpy(), axis=0)
                sent_label_ids = np.append(sent_label_ids, inputs["label"].detach().cpu().numpy(), axis=0)

            # Get token level predictions
            if preds_tok is None:
                preds_tok = logits_sec_seq.detach().cpu().numpy()
                tok_label_ids = inputs["label_ids_dv"].detach().cpu().numpy()
            else:
                preds_tok = np.append(preds_tok, logits_sec_seq.detach().cpu().numpy(), axis=0)
                tok_label_ids = np.append(tok_label_ids, inputs["label_ids_dv"].detach().cpu().numpy(), axis=0)

    label_map = {i: label for i, label in enumerate(labels)}

    if mode == 'dev':
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss,
        }
        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        return results

    if sentence_eval or token_eval:
        checkpoint=5
        output_pred_file = os.path.join(args.output_dir,
                                        str(checkpoint) + '_' + str(args.evaluation_set) + "_preds_gt.txt")
        if sentence_eval:
            # Convert logits to probabilities
            sigm = [1/(1+np.exp(-x)) for x in preds_sent]
            #sigm =[float(x[0]) for x in preds_sent]
            preds_sent = [1 for x in range(len(sigm))]
            for id_,x in enumerate(sigm):
                if x > 0.5:
                    preds_sent[id_] = 0

            with open(output_pred_file, "w") as writer:
                for (prob, (pr, gt)) in zip(sigm, zip(preds_sent, sent_label_ids)):
                    writer.write("%s\t %s\t %s\n" % (pr, gt, prob))
            result = compute_metrics('semdiv', preds_sent, sent_label_ids)

        else:
            sigm = None

        if token_eval:
            # Get token-level predictions
            preds_tok = np.argmax(preds_tok, axis=2)

            tok_label_list = [[] for _ in range(tok_label_ids.shape[0])]
            tok_preds_list = [[] for _ in range(tok_label_ids.shape[0])]

            for i in range(tok_label_ids.shape[0]):
                for j in range(tok_label_ids.shape[1]):
                    if tok_label_ids[i, j] != pad_token_label_id:
                        tok_label_list[i].append(label_map[tok_label_ids[i][j]])
                        tok_preds_list[i].append(label_map[preds_tok[i][j]])
        else:
            tok_preds_list = None

    return result, preds_sent, sigm, tok_preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}".format(mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == 'train' or mode == 'dev' or mode == 'test_synth':
            examples = read_examples_from_file(args.data_dir, mode)
        else:
            examples = read_examples_from_file(args.synth_data_dir, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, mode,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    if mode == 'train' or mode == 'dev':
        all_input_ids_eq = torch.tensor([f.input_ids_eq for f in features], dtype=torch.long)
        all_input_mask_eq = torch.tensor([f.input_mask_eq for f in features], dtype=torch.long)
        all_segment_ids_eq = torch.tensor([f.segment_ids_eq for f in features], dtype=torch.long)

    all_sentence_lbl = torch.tensor([int(f.label) for f in features], dtype=torch.long)
    all_input_ids_dv = torch.tensor([f.input_ids_dv for f in features], dtype=torch.long)
    all_input_mask_dv = torch.tensor([f.input_mask_dv for f in features], dtype=torch.long)
    all_segment_ids_dv = torch.tensor([f.segment_ids_dv for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids_dv for f in features], dtype=torch.long)

    if mode == 'train' or mode == 'dev':
        dataset = TensorDataset(all_input_ids_eq, all_input_mask_eq, all_segment_ids_eq, \
                                all_input_ids_dv,  all_input_mask_dv, all_segment_ids_dv, all_label_ids, all_sentence_lbl)

    else:
        # At test time we want to predict label and token level ids
        dataset = TensorDataset(all_input_ids_dv,  all_input_mask_dv, all_segment_ids_dv, all_label_ids, all_sentence_lbl)

    return dataset

def subword2token_labels(args, output_test_predictions_file, preds_sent, sigm, predictions, tokenizer, mode='test'):

    label_map = {'CHA': 'D', 'ADD': 'D', 'O': 'O', 'D': 'D', '1': 'D' ,'2': 'D', '3': 'D', '0': 'O'}
    binarize_map = {'CHA': 1, 'ADD': 1, 'D': 1, 'O': 0, '1': 1 ,'2':1, '3':1, '0':0}

    total_predictions_src, total_gold_src = [], []
    total_predictions_tgt, total_gold_tgt = [], []

    file_ = mode + '.tsv'
    if mode == 'train' or mode == 'dev' or mode == 'test_synth':
         out_ = args.data_dir
    else:
        out_ = args.synth_data_dir 


    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(out_, file_), "r") as f:
            f = f.readlines()
            example_id = -1

            for id_, line in enumerate(f):
                line = line.rstrip().split('\t')
                src_word_prediction, tgt_word_prediction = [], []
                if mode!='dev':
                    if id_ == 0:
                        continue
                    else:
                        example_id += 1
                else:
                    example_id+=1

                if mode != 'dev':
                    src_sent = line[3].split(' ')
                    tgt_sent = line[4].split(' ')
                else:
                    src_sent = line[2].split(' ')
                    tgt_sent = line[3].split(' ')

                prediction_length = len(predictions[example_id])
                # Get tokens of src and tgt and map them to labels
                src_toks = [tokenizer.tokenize(word) for word in src_sent]
                tgt_toks = [tokenizer.tokenize(word) for word in tgt_sent]


                # Extract word-level predictions from tokens
                for src_tok in src_toks:
                    preds = []
                    for subword in src_tok:
                        # Few sentences are > 128 assume 0 prediction
                        try:
                            preds.append(predictions[example_id].pop(0))
                        except:
                            preds.append('O')
                    if 'D' in preds:
                        total_predictions_src.append(1)
                        src_word_prediction.append('D')
                    else:
                        src_word_prediction.append('O')
                        total_predictions_src.append(0)

                for tgt_tok in tgt_toks:
                    preds = []
                    for subword in tgt_tok:
                        try:
                            preds.append(predictions[example_id].pop(0))
                        except:
                            continue

                    if 'D' in preds:
                        tgt_word_prediction.append('D')
                        total_predictions_tgt.append(1)

                    else:
                        tgt_word_prediction.append('O')
                        total_predictions_tgt.append(0)
                if mode != 'dev':
                    total_gold_src.extend([binarize_map[x] for x in line[5].split(' ')])
                    total_gold_tgt.extend([binarize_map[x] for x in line[6].rstrip().split(' ')])

                    src_spans = [label_map[x] for x in line[5].split(' ')]
                    tgt_spans = [label_map[x] for x in line[6].rstrip().split(' ')]
                else:
                    total_gold_src.extend([binarize_map[x] for x in line[4].split(' ')])
                    total_gold_tgt.extend([binarize_map[x] for x in line[5].rstrip().split(' ')])

                    src_spans = [label_map[x] for x in line[4].split(' ')]
                    tgt_spans = [label_map[x] for x in line[5].rstrip().split(' ')]

                assert len(src_word_prediction) == len(src_spans)
                assert len(tgt_word_prediction) == len(tgt_spans)

                writer.write( str(preds_sent[example_id]) + '\t' + str(sigm[example_id]) + '\t' +
                    ' '.join(src_sent) + '\t' + ' '.join(tgt_sent) + '\t' + ' '.join(src_spans) + '\t' + ' '.join(
                        tgt_spans) + '\t' + ' '.join(src_word_prediction) + '\t' + ' '.join(tgt_word_prediction) + '\n')
        writer.close()

    f1_src = f1_score(total_gold_src, total_predictions_src, average=None)
    f1_tgt = f1_score(total_gold_tgt, total_predictions_tgt, average=None)
    f1 = f1_score(total_gold_src+total_gold_tgt, total_predictions_src+total_predictions_tgt, average=None)


    results = {
        "SRC\tF1-OK": f1_src[0],
        "SRC\tF1-DIV": f1_src[1],
        "SRC\tF1-MUL": f1_src[0]*f1_src[1],
        "TGT\tF1-OK": f1_tgt[0],
        "TGT\tF1-DIV": f1_tgt[1],
        "TGT\tF1-MUL": f1_tgt[0]*f1_tgt[1],
        'AVG\tF1-OK': f1[0],
        'AVG\tF1-DIV': f1[1],
        'AVG\tF1-MUL': f1[0]*f1[1]
    }

    return results

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--node", default=None, type=str, required=True,
                        help="Node used for training.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir")
    parser.add_argument("--synth_data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--evaluation_set", type=str,
                        help="Evaluation set", default='dev')
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--best_checkpoint", action='store_true', 
                        help="Evaluate only one best checkpoint")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--margin', type=float, default=5,
                        help="margin for triplet loss")
    parser.add_argument('--alpha', type=float, default=5,
                        help="alpha for triplet loss")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument('--experiment_identifier', type=str, default='', help='experiment identifier name; this is used to control the name of the logs')
    args = parser.parse_args()
    args.start_training_time = time.time()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    config.margin = args.margin
    config.alpha = args.alpha
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


    if args.do_train:
        args.end_training_time = time.time()
        with open('runs/' + str(args.experiment_identifier + '/args'), 'w') as f:
            f.write('node\t' + str(args.node))
            f.write('data_dir\t' + str(args.data_dir))
            f.write('n_gpu\t' + str(args.n_gpu))
            f.write('start_training_time\t' + str(args.start_training_time))
            f.write('end_training_time\t' + str(args.end_training_time))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        prefix = 'checkpoint-' + str(args.best_checkpoint)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        if args.best_checkpoint:
            model = model_class.from_pretrained(args.output_dir)
            model.to(args.device)
            if args.evaluation_set!='dev':
                output_predictions_file = os.path.join(args.output_dir, args.evaluation_set + "_predictions.txt")
                _ , preds_sent, sigm, tok_preds_list = evaluate(args, model, tokenizer, prefix, labels, pad_token_label_id, args.evaluation_set, prefix=prefix)
                results = subword2token_labels(args, output_predictions_file, preds_sent, sigm, tok_preds_list, tokenizer, mode=args.evaluation_set)
                logger.info("***** Test results *****")
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))

            else:
                results = evaluate(args, model, tokenizer, prefix, labels, pad_token_label_id, args.evaluation_set, prefix=prefix)

            return results

        else:
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(os.path.dirname(c) for c in
                                   sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)

                if args.evaluation_set != 'dev':
                    output_predictions_file = os.path.join(str(checkpoint) + '_'+ args.evaluation_set + "_predictions.txt")
                    _, preds_sent, sigm, tok_preds_list = evaluate(args, model, tokenizer, prefix, labels,
                                                                   pad_token_label_id, args.evaluation_set,
                                                                   prefix=prefix)

                    results = subword2token_labels(args, output_predictions_file, preds_sent, sigm, tok_preds_list,
                                                  tokenizer, mode=args.evaluation_set)

                    logger.info("***** Test results *****")
                    for key in sorted(results.keys()):
                        logger.info("  %s = %s", key, str(results[key]))

                else:
                    evaluate(args, model, tokenizer, prefix, labels, pad_token_label_id, args.evaluation_set,
                                      prefix=prefix)


if __name__ == "__main__":
    main()
