# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
import os
import pickle as pkl

PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号, 未知字
START, END = '[START]', '[END]'
MAX_VOCAB_SIZE = 10000  # 词表长度限制


def build_dataset(config, tagging_schema):

    def dataset_split(data):
        n_train = len(data)
        n_val = int(n_train * 0.1)

        val_sample_ids = np.random.choice(n_train, n_val, replace=False)
        print("The first 10 val/test samples:", val_sample_ids[:10])
        val_set, tmp_train_set = [], []
        for i in range(n_train):
            record = data[i]
            if i in val_sample_ids:
                val_set.append(record)
            else:
                tmp_train_set.append(record)
        train_set = [r for r in tmp_train_set]
        return train_set, val_set

    def load_dataset(path, pad_size=32, test=False):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, tag_string = lin.split('####')

                word_tag_pairs = tag_string.split(' ')
                ote_tags = []
                for item in word_tag_pairs:
                    eles = item.split('=')
                    if len(eles) == 2:
                        word, tag = eles
                    elif len(eles) > 2:
                        # 句子中有"="需要进行特殊处理
                        tag = eles[-1]
                        word = (len(eles) - 2) * "="
                    ote_tags.append(tag)

                token = config.tokenizer.tokenize(content)
                token = [CLS] + token

                seq_len = len(token)
                mask = []
                ner_mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                ote_tags.insert(0, '[CLS]')
                if tagging_schema == 'OT':
                    ote_tags = ot(ote_tags)
                else:
                    ote_tags = ot2bio(ote_tags)
                ote_labels = set_labels(ote_tags, tagging_schema)

                if test == False:
                    with open("data/ote/token_label.txt", "a", encoding='utf-8') as f:
                        f.write(str(token) + '\n')
                        f.write(str(ote_tags) + '\n')

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                        ote_labels += ([0] * (pad_size - len(token)))
                        ner_mask = [1] * len(token) + [0] * (pad_size - len(token))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                        ner_mask = [1] * pad_size
                # print(content)
                assert len(ote_labels) == pad_size
                contents.append((token_ids, seq_len, mask, ote_labels, ner_mask))

        return contents

    if 'CRF' in config.model_name:
        tagging_schema = 'BIO'
    else:
        tagging_schema = 'OT'
    train = load_dataset(config.train_path, config.pad_size, test=True)
    dev = load_dataset(config.dev_path, config.pad_size, test=True)
    test = load_dataset(config.test_path, config.pad_size)
    # train, dev = dataset_split(train)
    # train, test = dataset_split(train)

    return train, dev, test


def ot(ote_tag_sequence):
    new_ote_sequence = []
    n_tag = len(ote_tag_sequence)
    for i in range(n_tag):
        cur_ote_tag = ote_tag_sequence[i]
        if cur_ote_tag == '[CLS]':
            new_ote_sequence.append(cur_ote_tag)
        elif cur_ote_tag == 'O':
            new_ote_sequence.append(cur_ote_tag)
        else:
            new_ote_sequence.append('T')
    return new_ote_sequence

def ot2bio(ote_tag_sequence):
    new_ote_sequence = ot2bio_ote(ote_tag_sequence=ote_tag_sequence)
    assert len(new_ote_sequence) == len(ote_tag_sequence)
    return new_ote_sequence

def ot2bio_ote(ote_tag_sequence):
    new_ote_sequence = []
    n_tag = len(ote_tag_sequence)
    prev_ote_tag = '$$$'
    for i in range(n_tag):
        cur_ote_tag = ote_tag_sequence[i]
        if cur_ote_tag == '[CLS]':
            new_ote_sequence.append(cur_ote_tag)
        elif cur_ote_tag == 'O':
            new_ote_sequence.append(cur_ote_tag)
        else:
            # cur_ote_tag 中有 'T'
            if str(cur_ote_tag).find('T') != -1:
                if prev_ote_tag == cur_ote_tag:
                    new_ote_sequence.append('I')
                else:
                    new_ote_sequence.append('B')
            else:
                # cur tag is at the beginning of the opinion target
                new_ote_sequence.append('B')
        prev_ote_tag = cur_ote_tag
    return new_ote_sequence

def set_labels(ote_tags, tagging_schema='OT'):
    if tagging_schema == 'OT':
        ote_tag_vocab = {'[SEP]': 0, '[CLS]': 1, 'O': 2, 'T': 3}
        ts_tag_vocab = {'O': 0, 'T-POS': 1, 'T-NEG': 2}
    elif tagging_schema == 'BIO':
        ote_tag_vocab = {'[SEP]': 0, '[CLS]': 1, 'O': 2, 'B': 3, 'I': 4}
        ts_tag_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'B-NEG': 3, 'I-NEG': 4}
    elif tagging_schema == 'BIEOS':
        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
        ts_tag_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'E-POS': 3, 'S-POS': 4,
                        'B-NEG': 5, 'I-NEG': 6, 'E-NEG': 7, 'S-NEG': 8}
    else:
        raise Exception("Invalid tagging schema %s" % tagging_schema)
    n_records = len(ote_tags)
    for i in range(n_records):
        ote_labels = [ote_tag_vocab[t] for t in ote_tags]
    return ote_labels


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        ner_mask = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        return (x, seq_len, mask, ner_mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
