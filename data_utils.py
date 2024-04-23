import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from collections import Counter
from copy import deepcopy
from glob import glob
import math
import json
from random import sample
import pandas as pd
import argparse


def build_tokenizer_old(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def split_punc(text):
    processed_text = ""
    for c in text:
        if c in [',', '.', '!', '?', '/', '#', '@', '(', ')', '{', '}']:
            processed_text += ' '
        processed_text += c
    return processed_text


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines)):
                line = lines[i].strip()
                line_data = line.split("\t")
                assert len(line_data) == 2
                text_raw = split_punc(line_data[0])
                aspect = fname.split("/")[-1].lower()
                aspect.replace("_", " ")

                text_raw = text_raw + " " + aspect.replace("#", " ") + " "
                text += text_raw
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else '../glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {'[mask]': 1}
        self.idx2word = {1: '[mask]'}
        self.idx = 2

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', max_seq_len=None):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        return pad_and_truncate(sequence, max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', max_seq_len=None):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        return pad_and_truncate(sequence, max_seq_len, padding=padding, truncating=truncating)


class TaskDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TraditionDataset():
    def __init__(self, data_dir, tasks, tokenizer, opt):
        self.tasks = tasks
        self.all_targets = self.tasks
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.opt = opt
        self.polarities = opt.polarities
        self.get_all_aspect_indices()
        self.all_data = []
        self.all_data = self._get_all_data()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def get_all_aspect_indices(self, ):
        self.all_aspect_indices = []
        for aspect in self.all_targets:
            aspect = aspect.replace("#", ' ')
            aspect = aspect.replace("_", ' ')
            aspect_indices = self.tokenizer.text_to_sequence(aspect, max_seq_len=4)
            assert not all(aspect_indices == 0), "aspect error"
            self.all_aspect_indices.append(aspect_indices)
        assert len(self.all_aspect_indices) == len(self.tasks)

    def _get_all_data(self):
        self.polarity2label = {polarity: idx for idx, polarity in enumerate(self.opt.polarities)}
        self.label2polarity = {idx: polarity for idx, polarity in enumerate(self.opt.polarities)}
        out_masked_features = "./masked_feature/{}".format(self.opt.dataset)
        if not os.path.exists(out_masked_features):
            os.makedirs(out_masked_features)
        for task_id, task in enumerate(self.tasks):
            fname = os.path.join(self.data_dir, task)
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            fin.close()
            aspect = task.split(".")[0].replace("#", ' ').lower()
            aspect = aspect.replace("_", " ")
            for i in range(0, len(lines)):
                line = lines[i].strip()
                line_data = line.split("\t")
                assert len(line_data) == 2, "line data error"
                text = split_punc(line_data[0])
                polarity = line_data[1]
                text_indices = self.tokenizer.text_to_sequence(text)
                aspect_indices = self.tokenizer.text_to_sequence(aspect, max_seq_len=4)
                aspect_len = np.sum(aspect_indices != 0)
                if self.polarity2label.get(polarity) is not None:
                    label = self.polarity2label[polarity]
                else:
                    print("error polarity: {}\n{}".format(len(polarity), self.polarity2label))
                    label = None
                assert type(label) == int
                text_len = np.sum(text_indices != 0)
                concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")
                concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)
                text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
                data = {
                    'concat_bert_indices': concat_bert_indices,
                    'concat_segments_indices': concat_segments_indices,
                    'text_bert_indices': text_bert_indices,
                    'text_indices': text_indices,
                    'aspect_indices': aspect_indices,
                    'task_id': task_id,
                    'polarity': label,
                }
                self.all_data.append(data)

        return self.all_data


def _get_tasks(task_path):
    tasks = []
    with open(task_path) as file:
        for line in file.readlines():
            line = line.strip()
            tasks.append(line)
    return tasks


def _get_file_names(data_dir, tasks):
    return [os.path.join(data_dir, task) for task in tasks]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ZeroshotDataset():
    def __init__(self, data_dir, tokenizer, opt, data_type):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.opt = opt
        self.polarities = opt.polarities
        # self.get_all_aspect_indices()
        self.all_data = []
        self.data_type = data_type
        self.type = opt.type
        self.all_data = self._get_all_data()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def _get_all_data(self):
        out_masked_features = "./masked_feature/{}".format(self.opt.dataset)
        if not os.path.exists(out_masked_features):
            os.makedirs(out_masked_features)

        data_file = pd.read_csv(self.data_dir, encoding='unicode_escape')
        if self.data_type == 'train' or self.type == 2:
            pass
        else:
            data_file = data_file[data_file['seen?'] == self.type]
        topics = data_file['topic_str'].tolist()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        label_list = data_file.label.to_list()
        tweets = data_file['text_s'].tolist()
        wiki_summaries = data_file['knowledge'].astype(str).tolist()
        tweets_targets = [f'text: {x} target: {y}' for x, y in zip(tweets, topics)]
        print(len(tweets_targets))
        encodings = tokenizer(tweets_targets, wiki_summaries, padding=True, truncation=True)
        # input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        # attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        # token_type_ids = torch.tensor(encodings['token_type_ids'], dtype=torch.long)
        for i in range(len(encodings['input_ids'])):
            data = {
                'concat_bert_indices': torch.tensor(encodings['input_ids'][i]),
                'concat_segments_indices': torch.tensor(encodings['token_type_ids'][i]),
                'polarity': label_list[i],

            }
            self.all_data.append(data)

        return self.all_data


from tqdm import tqdm


class ZSSDDataset(Dataset):

    def __init__(self, fname, tokenizer, ):

        self.tokenizer = tokenizer
        self.all_data = []
        self.all_data = self._get_all_data(fname, tokenizer)

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def _get_all_data(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

        self.fname = fname
        self.tokenizer = tokenizer
        # self.opt = opt
        # self.polarities = opt.polarities

        lines = fin.readlines()
        fin.close()

        index = 0
        text_target_list = []
        polarity_list = []
        knowledge_list = []
        # topic2index = {topic: idx for idx, topic in enumerate(topics)}
        # progress = tqdm(total=len(lines)/4)
        for i in range(0, len(lines), 4):
            text = lines[i].lower().strip()
            target = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            knowledge = lines[i + 3].strip()
            text_target = 'target: {} text: {} '.format(text, target)
            text_target_list.append(text_target)
            polarity_list.append(int(polarity))
            knowledge_list.append(knowledge)

            # progress.update(1)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # print(len(text_target_list))
        encodings = tokenizer(text_target_list, knowledge_list, padding=True, truncation=True)
        for i in range(len(encodings['input_ids'])):
            data = {
                'concat_bert_indices': torch.tensor(encodings['input_ids'][i]),
                'concat_segments_indices': torch.tensor(encodings['token_type_ids'][i]),
                'polarity': polarity_list[i],

            }
            self.all_data.append(data)
        return self.all_data


from tqdm import tqdm


class zeroshotDataset(Dataset):

    def __init__(self, data_dir, tokenizer, opt, data_type, target):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.opt = opt
        self.polarities = opt.polarities
        # self.get_all_aspect_indices()
        self.all_data = []
        self.data_type = data_type
        self.type = opt.type
        self.all_data = self._get_all_data()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def _get_all_data(self):
        out_masked_features = "./masked_feature/{}".format(self.opt.dataset)
        if not os.path.exists(out_masked_features):
            os.makedirs(out_masked_features)

        data_file = pd.read_csv(self.data_dir, encoding='utf-8')
        if self.data_type == 'train':
            data_file = data_file[~(data_file['target'] == self.opt.target)]
        else:
            data_file = data_file[data_file['target'] == self.opt.target]

        data_file = data_file.reset_index()
        topics = data_file['target'].tolist()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # 0 for Against, 1 for Favor, 2 for none
        data_file['label'] = data_file['label'].replace({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2})
        label_list = data_file.label.to_list()

        tweets = data_file['text'].tolist()
        wiki_summaries = data_file['background'].astype(str).tolist()
        tweets_targets = [f'text: {x} target: {y}' for x, y in zip(tweets, topics)]
        print(len(tweets_targets))
        encodings = tokenizer(tweets_targets, wiki_summaries, padding=True, truncation=True)
        # input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        # attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        # token_type_ids = torch.tensor(encodings['token_type_ids'], dtype=torch.long)
        for i in range(len(encodings['input_ids'])):
            data = {
                'concat_bert_indices': torch.tensor(encodings['input_ids'][i]),
                'concat_segments_indices': torch.tensor(encodings['token_type_ids'][i]),
                'polarity': label_list[i],

            }
            self.all_data.append(data)

        return self.all_data
