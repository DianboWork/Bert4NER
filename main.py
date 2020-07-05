from bert_tagger.utils.data import Data
from bert_tagger.utils.batchify import batchify
from bert_tagger.utils.config import get_args
from bert_tagger.models.pretrain_base import Token_Classification
from bert_tagger.utils.metric import get_ner_fmeasure
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
import os
import numpy as np
import copy
import pickle
import torch
import torch.nn as nn
import time
import random
import sys
import gc

def data_initialization(args):
    data_stored_directory = args.data_stored_directory
    file = data_stored_directory + args.dataset_name + "_dataset.dset"
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(data_stored_directory, args.dataset_name)
    else:
        data = Data()
        data.generate_instance(args.train_doc, "train", args.bert_file)
        data.generate_instance(args.dev_doc, "dev", args.bert_file)
        data.generate_instance(args.test_doc, "test", args.bert_file)
        save_data_setting(data, args.dataset_name, data_stored_directory)
    return data


def save_data_setting(data, dataset_name, data_stored_directory):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(data_stored_directory):
        os.makedirs(data_stored_directory)
    dataset_saved_name = data_stored_directory + dataset_name +"_dataset.dset"
    with open(dataset_saved_name, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", dataset_saved_name)


def load_data_setting(data_stored_directory, name):
    dataset_saved_name = data_stored_directory + name + "_dataset.dset"
    with open(dataset_saved_name, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", dataset_saved_name)
    data.show_data_summary()
    return data

def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def evaluate(data, model, args, name):
    if name == "train":
        instances = data.train_ids
        texts = data.train_texts
    elif name == "dev":
        instances = data.dev_ids
        texts = data.dev_texts
    elif name == 'test':
        instances = data.test_ids
        texts = data.test_texts
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    model.eval()
    batch_size = args.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    o_label = data.label_alphabet.get_index("O")
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        text = texts[start:end]
        if not instance:
            continue
        input_ids, attention_mask, label_seq_tensor, loss_mask, crf_mask, scope = batchify(instance, args, o_label)
        tag_seq = model(input_ids, attention_mask, crf_mask, scope)
        pred_label, gold_label = recover_label(tag_seq, label_seq_tensor, attention_mask, data.label_alphabet)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagscheme)
    return speed, acc, p, r, f, pred_results


def train(data, model, args):
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(data.train_ids) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(data.train_ids) // args.batch_size) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(t_total*args.warmup_proportion), t_total=t_total)
    scheduler = WarmupConstantSchedule(optimizer, warmup_steps=int(t_total * args.warmup_proportion))
    best_dev = -1
    o_label = data.label_alphabet.get_index("O")
    for idx in range(args.num_train_epochs):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, args.num_train_epochs))
        instance_count = 0
        batch_loss = 0
        sample_loss = 0
        total_loss = 0
        random.shuffle(data.train_ids)
        model.train()
        model.zero_grad()
        batch_size = args.batch_size
        train_num = len(data.train_ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_ids[start:end]
            if not instance:
                continue
            model.zero_grad()
            input_ids, attention_mask, label_seq_tensor, loss_mask, crf_mask, scope = batchify(instance, args, o_label)
            loss, best_path = model.neg_log_likelihood(input_ids, attention_mask, label_seq_tensor, crf_mask, scope)
            instance_count += 1
            sample_loss += loss.item()
            total_loss += loss.item()
            batch_loss += loss
            loss.backward()
            if args.use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (batch_id + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

            if end % 100 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f" % (
                end, temp_cost, sample_loss))
                sys.stdout.flush()
                sample_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        speed, acc, p, r, f, _ = evaluate(data, model, args, "test")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_score = f
        print(
            "Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        if current_score > best_dev:
            print("Exceed previous best f score:", best_dev)
            if not os.path.exists(args.param_stored_directory + args.dataset_name + "_param"):
                os.makedirs(args.param_stored_directory + args.dataset_name + "_param")
            model_name = "{}epoch_{}_f1_{}.model".format(args.param_stored_directory + args.dataset_name + "_param/", idx, current_score)
            # torch.save(model.state_dict(), model_name)
            best_dev = current_score
        gc.collect()


if __name__ == '__main__':
    args, unparsed = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    data = data_initialization(args)
    model = Token_Classification(args, data)
    if args.use_gpu:
        model = model.cuda()
    train(data, model, args)


