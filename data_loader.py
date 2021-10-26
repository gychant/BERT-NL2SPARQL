"""Data loader"""
import random
import numpy as np
import os
import sys
from tqdm import tqdm
import torch
from bert import BertTokenizer
import json
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

# torch.set_printoptions(threshold=sys.maxsize)
# WHITESPACE_PLACEHOLDER = ' □ '
qmark_placeholder = '[unused21]'
uri_placeholder = '[unused22]'

def spo_list2triplets_dict(spo_list):
    '''get sub: [pre, obj] dict'''
    triplets_dict = {}
    for spo in spo_list:
        sub = spo['subject'].strip().lower()
        pre = spo['predicate']
        obj = spo['object'].strip().lower()
        if sub in triplets_dict:
            triplets_dict[sub].append((pre, obj))
        else:
            triplets_dict[sub] = [(pre, obj)]
    return triplets_dict

def spo_list2triplets(spo_list):
    triplets = set()
    for spo in spo_list:
        sub = spo['subject'].strip().lower()
        pre = spo['predicate']
        obj = spo['object'].strip().lower()
        triplets.add((sub, pre, obj))
    return triplets

def replace_placeholder(text):
    return text.replace('?x', qmark_placeholder).replace('?uri', uri_placeholder)   # uri, for the first 3 dataset uri, ?y --- >uri
    # return text


def replace_placeholder_rnn(text):
    return text.replace("?x", "xxxxx").replace("?uri", "yyyyy")


import unicodedata
from nltk.tokenize.treebank import TreebankWordTokenizer
class NLTKTokenizer(object):
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}

    def tokenize(self, text):
        return self.processed_text(text).split()

    def set_vocab(self, vocab_size, word2idx, idx2word):
        self.vocab_size = vocab_size
        self.word2idx = word2idx
        self.idx2word = idx2word

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.word2idx:
                ids.append(self.word2idx[tok])
            else:
                ids.append(self.word2idx["<unk>"])
        return ids

    def processed_text(self, text, to_strip_accents=False):
        text = text.replace('\\\\', '')
        if to_strip_accents:
            stripped = strip_accents(text.lower())
        else:
            stripped = text.lower()   # .lower()
        toks = self.tokenizer.tokenize(stripped)
        return " ".join(toks)

    def strip_accents(self, text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if unicodedata.category(c) != 'Mn')


class DataLoader(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.pre2idx, self.idx2pre = self.load_predicates()
        args.idx2pre = self.idx2pre
        self.encoder_type = args.encoder_type
        if self.encoder_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=True)
        elif self.encoder_type == "rnn":
            self.tokenizer = NLTKTokenizer()
        self.max_len = args.max_len
        self.device = args.device
        self.batch_size = args.batch_size

        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        self.build_vocab(["train", "test"])

    def load_predicates(self):
        pres = ['Nan']
        #todo
        with open(os.path.join(self.data_dir, 'schemas'), 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                pre = json.loads(line)['predicate']
                if pre not in pres:
                    pres.append(pre)
        pre2idx = {pre:idx for idx, pre in enumerate(pres)}
        idx2pre = {idx:pre for idx, pre in enumerate(pres)}
        return pre2idx, idx2pre

    def build_vocab(self, data_types):
        print("building vocabulary ...")
        vocab_size = 0
        word2idx = {}
        idx2word = {}

        # add special tokens
        special_tokens = ["<pad>", "<unk>", "[CLS]", "[SEP]", "xxxxx", "yyyyy"]
        for word in special_tokens:
            if word not in word2idx:
                word2idx[word] = vocab_size
                idx2word[vocab_size] = word
                vocab_size += 1

        for data_type in data_types:
            with open(os.path.join(self.data_dir, "{}_data.json".format(data_type)), 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    sample = json.loads(line)
                    for word in self.tokenizer.tokenize(sample['text'].lower()):
                        if word not in word2idx:
                            word2idx[word] = vocab_size
                            idx2word[vocab_size] = word
                            vocab_size += 1
        self.vocab_size = vocab_size
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.tokenizer.set_vocab(vocab_size, word2idx, idx2word)
        print("Done.")

    def load_data(self, data_type, max_len=300, repeat_multi_sub=False, encoder_type="bert"):
        data = []
        with open(os.path.join(self.data_dir, '%s_data.json'%data_type), 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data_sample = {'sub': {'tokens': [], 'spans': ([], []), 'weight': 0}, 'obj': {}, 'gold_triplets': set(), 'sent': None, 'id': None}
                sample = json.loads(line)
                data_sample['sent'] = sample['text']

                data_sample['id'] = sample['id']
                #print("sample:", sample)
                #print("max_len:", max_len)
                text = sample['text'].lower()[:max_len]
                spo_list = sample['spo_list'] if data_type not in ['test', 'test_debug', 'test_final', 'corrected_questions-verb-copy', 'corrected_questions-corrected-copy'] else []
                triplets_dict = spo_list2triplets_dict(spo_list)
                data_sample['gold_triplets'] = spo_list2triplets(spo_list)
                if encoder_type == "bert":
                    tokens = ['[CLS]'] + self.tokenizer.tokenize(replace_placeholder(text), inference=True) + ['[SEP]']
                    print("tokens:", tokens)
                    print(hello)
                elif encoder_type == "rnn":
                    tokens = ['[CLS]'] + self.tokenizer.tokenize(replace_placeholder_rnn(text)) + ['[SEP]']
                    # print('text:',text, "tokens:", tokens)
                data_sample['sub']['tokens'] = tokens
                data_sample['sub']['weight'] = len(spo_list)
                used_spans = set()

                if len(spo_list) == 0 and data_type not in ['test', 'test_debug', 'test_final','corrected_questions-verb-copy', 'corrected_questions-corrected-copy']:
                    continue
                else:
                    for sub in triplets_dict:
                        if encoder_type == "bert":
                            sub_tokens = self.tokenizer.tokenize(replace_placeholder(sub), inference=True)
                        elif encoder_type == "rnn":
                            sub_tokens = self.tokenizer.tokenize(replace_placeholder_rnn(sub))
                        used_spans, span = self._find_span(used_spans, tokens, sub_tokens)
                        if span:
                            # print(data_sample, span[0], span[1], len(tokens))
                            assert span[0] < len(tokens) and span[1] > 0
                            data_sample['sub']['spans'][0].append(span[0])
                            data_sample['sub']['spans'][1].append(span[1]-1)
                    for sub in triplets_dict:
                        if repeat_multi_sub:
                            data_sample['obj'] = {}
                        data_sample['obj'][sub] = {'query_tokens': [], 'token_types': [], 'spans': ([], []), 'weight': 0}
                        if encoder_type == "bert":
                            query_tokens_sub = self.tokenizer.tokenize(replace_placeholder(sub), inference=True)
                        elif encoder_type == "rnn":
                            query_tokens_sub = self.tokenizer.tokenize(replace_placeholder_rnn(sub))
                        query_tokens = ['[CLS]'] + query_tokens_sub + ['[SEP]'] + tokens[1:]
                        token_types = [0] + [0]*len(query_tokens_sub) + [0] + [1]*len(tokens[1:])
                        assert len(query_tokens) == len(token_types)
                        data_sample['obj'][sub]['query_tokens'] = query_tokens
                        data_sample['obj'][sub]['weight'] = len(triplets_dict[sub])
                        data_sample['obj'][sub]['token_types'] = token_types

                        used_spans = set()    #todo reset used_spans( are used)
                        for pre, obj in triplets_dict[sub]:
                            if encoder_type == "bert":
                                obj_tokens = self.tokenizer.tokenize(replace_placeholder(obj), inference=True)
                            elif encoder_type == "rnn":
                                obj_tokens = self.tokenizer.tokenize(replace_placeholder_rnn(obj))
                            # if obj == 'anaheim, california':
                            #     print(used_spans)
                            used_spans, span = self._find_span(used_spans, query_tokens, obj_tokens)
                            # if obj == 'anaheim, california':
                            #     print(used_spans)
                            #     print(obj_tokens, query_tokens, span)
                            if span:
                                assert span[0] < len(query_tokens) and span[1] > 0
                                data_sample['obj'][sub]['spans'][0].append((span[0], self.pre2idx[pre]))
                                data_sample['obj'][sub]['spans'][1].append(span[1]-1)
                        if repeat_multi_sub:
                            data.append(data_sample)
                    if not repeat_multi_sub:
                        data.append(data_sample)
                    # print('data', data_sample)
        return data

    def data_iterator(self, data, batch_size, seed=None, is_train=False, shuffle=False):
        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        data_size = len(data)
        order = list(range(data_size))
        if shuffle:
            random.seed(seed)
            random.shuffle(order)
        # print(data_size , batch_size)
        for i in range(data_size // batch_size):
            batch_data = [data[idx] for idx in order[i*batch_size: (i+1)*batch_size]]
            # subject task
            ## max_len of sub task
            batch_sub_lengths = [len(data_sample['sub']['tokens']) for data_sample in batch_data]
            max_len_sub_tokens = max(batch_sub_lengths)
            ## subtask data
            batch_tokens = np.zeros((batch_size, max_len_sub_tokens))
            if is_train:
                batch_sub_heads = np.zeros((batch_size, max_len_sub_tokens))
                batch_sub_tails = np.zeros((batch_size, max_len_sub_tokens))
                batch_sub_weights = np.zeros(batch_size)
                # object task
                ## max_len of obj task
                batch_subs = [random.choice(list(data_sample['obj'].keys())) for data_sample in batch_data] # random pick a subj key
                batch_obj_lengths = [len(data_sample['obj'][batch_subs[i]]['query_tokens']) for i, data_sample in enumerate(batch_data)]
                # print('batch subjs', batch_subs)
                max_len_obj_tokens = max(batch_obj_lengths)
                ## objtask data
                batch_query_tokens = np.zeros((batch_size, max_len_obj_tokens))
                batch_token_types = np.zeros((batch_size, max_len_obj_tokens))
                batch_obj_heads = np.zeros((batch_size, max_len_obj_tokens, len(self.pre2idx)))
                batch_obj_tails = np.zeros((batch_size, max_len_obj_tokens))
                batch_obj_weights = np.zeros(batch_size) # gai
            for i, data_sample in enumerate(batch_data):
                print('\n', data_sample)
                batch_tokens[i, :batch_sub_lengths[i]] = self.tokenizer.convert_tokens_to_ids(data_sample['sub']['tokens'])
                print("tokens:", batch_tokens[i, :batch_sub_lengths[i]])
                input()
                if is_train:
                    batch_sub_heads[i, data_sample['sub']['spans'][0]] = 1 # todo check if add subjs not sampled -> yes
                    batch_sub_tails[i, data_sample['sub']['spans'][1]] = 1
                    # add other subject terms which are not sampled
                    # print('batch sub heads', batch_sub_heads)
                    batch_sub_weights[i] = data_sample['sub']['weight']
                    # object predicate task, todo check if add objs and predicates from the same subjs
                    sub = batch_subs[i]
                    batch_query_tokens[i, :batch_obj_lengths[i]] = self.tokenizer.convert_tokens_to_ids(data_sample['obj'][sub]['query_tokens'])
                    batch_token_types[i, :batch_obj_lengths[i]] = data_sample['obj'][sub]['token_types']
                    #print("data_sample:", data_sample)
                    # print(hello)
                    batch_obj_heads[i, [tup[0] for tup in data_sample['obj'][sub]['spans'][0]], [tup[1] for tup in data_sample['obj'][sub]['spans'][0]]] = 1
                    batch_obj_tails[i, data_sample['obj'][sub]['spans'][1]] = 1
                    batch_obj_weights[i] = data_sample['obj'][sub]['weight']
                    # print('batch obj tails', batch_obj_tails)

            # to tensor
            batch_tokens = torch.tensor(batch_tokens, dtype=torch.long).to(self.device)
            if is_train:
                batch_sub_heads = torch.tensor(batch_sub_heads, dtype=torch.float).to(self.device)
                batch_sub_tails = torch.tensor(batch_sub_tails, dtype=torch.float).to(self.device)
                batch_sub_weights = torch.tensor(batch_sub_weights, dtype=torch.float).to(self.device)
                # to tensor
                batch_query_tokens = torch.tensor(batch_query_tokens, dtype=torch.long).to(self.device)
                batch_token_types = torch.tensor(batch_token_types, dtype=torch.long).to(self.device)
                batch_obj_heads = torch.tensor(batch_obj_heads, dtype=torch.float).to(self.device)
                batch_obj_tails = torch.tensor(batch_obj_tails, dtype=torch.float).to(self.device)
                batch_obj_weights = torch.tensor(batch_obj_weights, dtype=torch.float).to(self.device)
                yield (batch_tokens, batch_sub_heads, batch_sub_tails, batch_sub_weights), \
                    (batch_query_tokens, batch_token_types, batch_obj_heads, batch_obj_tails, batch_obj_weights)
            else:
                yield batch_tokens

    def _find_span(self, used_spans, tokens, entity_tokens):
        spans = self._find_all_spans(tokens, entity_tokens)
        for span in spans:
            if not self._has_intersection(used_spans, span):
                used_spans.add(span)
                return used_spans, span
        return used_spans, None

    def _has_intersection(self, used_spans, span):
        for used_span in used_spans:
            used_span_set = set(range(used_span[0], used_span[1]))
            span_set = set(range(span[0], span[1]))
            if 0 < len(used_span_set.intersection(span_set)) < max(len(span_set), len(used_span_set)):
                return True
        return False

    def _find_all_spans(self, tokens, entity_tokens):
        res = []
        for i in range(len(tokens)-len(entity_tokens)+1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                res.append((i, i+len(entity_tokens)))
        return res

def data_loader_test():
    class ARGS:
        data_dir = 'lic-corrected-70-semantic-embedd-v4' # lic-corrected-70-semantic-embedd-v4; WebQSP_0824_67
        # data_dir = 'data'
        max_len = 100
        device = 'cpu'
        bert_model_dir = 'uncased_L-12_H-768_A-12'
        batch_size = 10
        encoder_type = 'rnn'
    dl = DataLoader(ARGS())

    dev_data = dl.load_data('test_re_pp', repeat_multi_sub=False, encoder_type='rnn')
    # dev_data = dl.load_data('dev')
    dg = dl.data_iterator(dev_data, 1, 5, is_train=True, shuffle=False)
    for i, (batch_sub, batch_obj) in enumerate(dg):
        input()
        print(i, dev_data[1*i])
        # for tmp in batch_sub:
        #     print(tmp[0], tmp.size())
        # for tmp in batch_obj:
        #     print(tmp[0], tmp.size())
    # dg = dl.data_iterator(dev_data, 1, 6, is_train=True, shuffle=False)
    # for i, (batch_sub, batch_obj) in enumerate(dg):
    #     input()
    #     print(dev_data[1 * i])

if __name__ == "__main__":
    data_loader_test()
# tokens = self.tokenizer.tokenize(replace_placeholder(text), inference=True)