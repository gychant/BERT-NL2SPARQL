import random
import numpy as np
import os
import sys
from tqdm import tqdm
import json
from copy import deepcopy

import re
from fuzzywuzzy import process, fuzz
from nltk.tokenize.treebank import TreebankWordTokenizer

# todo: how to re-organize/format the dataset
tokenizer = TreebankWordTokenizer()

zhuanyou_dict = {'u.s .':'u.s.', 'u.k .':'u.k.', 'u.n.c.l.e .': 'u.n.c.l.e.', 'ltd .':'ltd.',
                 '` zee \'':'\'zee \'', 'f.c .': 'f.c.', 'r.e.m .': 'r.e.m.', 'l.h.o.o.q .':'l.h.o.o.q.'
                 ,'e.n.d .':'e.n.d.', 'e.t .':'e.t.','tiffany \'s':'tiffany\'s', 'ocean ` s': 'ocean`s',
                 's.s .':'s.s.', 'inc .': 'inc.', 'jr .': 'jr.', 'j.f.k .':'j.f.k.'}

def processed_text(text):
    '''use nltk treebank tokenizer to tokenize the sentence/words'''
    text = text.replace('\\\\', '')
    stripped = text.lower()
    toks = tokenizer.tokenize(stripped)
    return " ".join(toks)

def replace_zhuanyou(tokens):
    '''replace some zhuanyou mingci'''
    for dic_element in zhuanyou_dict.keys():
        if dic_element in tokens:
            tokens = tokens.replace(dic_element, zhuanyou_dict[dic_element])
    return tokens

def open_fbqa_data_file(data_dir, data_type):
    '''the format is a bit different'''
    with open(os.path.join(data_dir, '%s.json' % data_type), 'r', encoding='utf-8') as fbw:
        temp_data_dic = json.load(fbw)
    print('total lines in %s:'% data_type, len(temp_data_dic['Questions']))

    return temp_data_dic

def fuzzy_score(query_list, candidate_list):
    '''compute the score between potentialtopicentitymention and topicentityname'''
    assert len(query_list) == len(candidate_list)
    score_list = []
    for i in range(len(query_list)):
        tmp_score = fuzz.partial_ratio(query_list[i], candidate_list[i])
        score_list.append(tmp_score)

    return score_list


def reverse_linking(data_dir, data_type, saved_data_type):

    data_dic = open_fbqa_data_file(data_dir, data_type)

    with open(os.path.join(data_dir, '%s.json' % saved_data_type), 'w', encoding='utf-8') as fs:
        for temp_question in data_dic['Questions']:
            saved_one_line_dic = {}  # the one-line data_dic to be saved to the saved data type
            temp_q_id = temp_question['Question-ID']
            temp_p_question = processed_text(temp_question['ProcessedQuestion'])
            temp_pharse_ids = []
            temp_pharse_ptems = []
            temp_pharse_tems = []
            temp_pharse_ics = []
            temp_mids = []
            for temp_pharse in temp_question['Parses']:  #every question has multiple Parses (multi-paths)
                temp_pharse_ids.append(temp_pharse['Parse-Id'])
                temp_p_ptem = processed_text(temp_pharse['PotentialTopicEntityMention']
                                             .replace('?', '\'').replace('""', '').replace('``', ''))
                temp_p_ptem = replace_zhuanyou(temp_p_ptem)
                temp_pharse_ptems.append(temp_p_ptem)
                temp_pharse_tems.append(temp_pharse['TopicEntityName'])
                temp_pharse_ics.append(temp_pharse['InferentialChain'])
                temp_mids.append(temp_pharse['TopicEntityMid'])

            # ensure ptem is in the question
            for temp_pharse_ptem in temp_pharse_ptems:
                if temp_pharse_ptem not in temp_p_question:
                    print('question_id',temp_q_id, 'processed_text', temp_pharse_ptem, 'processed_q', temp_p_question)
                    raise ValueError('potential topic entity mention not in processed question')

            score_list = fuzzy_score(temp_pharse_ptems, temp_pharse_tems)
            highest_score_index = np.argmax(score_list)

            # find all selected items
            selected_pharse_id = temp_pharse_ids[highest_score_index]
            selected_pharse_ptem = temp_pharse_ptems[highest_score_index]
            selected_pharse_tem = temp_pharse_tems[highest_score_index]
            selected_pharse_ic = temp_pharse_ics[highest_score_index]
            selected_pharse_mid = temp_mids[highest_score_index]
            selected_pharse_ic_list = selected_pharse_ic.split('..')   #split the inferential chain

            saved_one_line_dic['id'] = temp_q_id
            saved_one_line_dic['text'] = temp_p_question

            saved_one_line_dic['Parse-Id'] = selected_pharse_id
            saved_one_line_dic['TopicEntityMention'] = selected_pharse_tem

            saved_one_line_dic['spo_list'] = []

            if len(selected_pharse_ic_list) == 1:
                tmp_spo1 = {"subject": selected_pharse_ptem, "subject_mid": selected_pharse_mid,
                            "predicate": selected_pharse_ic_list[0], "object": "?uri"}
                saved_one_line_dic['spo_list'].append(tmp_spo1)
            elif len(selected_pharse_ic_list) == 2:
                tmp_spo1 = {"subject": selected_pharse_ptem, "subject_mid": selected_pharse_mid,
                            "predicate": selected_pharse_ic_list[0], "object": "?x"}
                tmp_spo2 = {"subject": '?x', "subject_mid": None,
                            "predicate": selected_pharse_ic_list[1], "object": "?uri"}
                saved_one_line_dic['spo_list'].append(tmp_spo1)
                saved_one_line_dic['spo_list'].append(tmp_spo2)

            else:
                raise ValueError('pharse inferential chain can not be longer than 2')

            fs.write(json.dumps(saved_one_line_dic, ensure_ascii=False)+'\n')

def create_schemas(data_dir, data_type, saved_file):
    '''count # schemas in data file'''
    schemas = []
    count = 0  #total lines
    count_2 = 0 #total number of schemas (s-p-o)
    count_3 = 0  #total number of predicates only
    preds = []
    # tmp_schema = {"object_type": None, "predicate": None, "subject_type": None}
    with open(os.path.join(data_dir, '%s_data.json' % data_type), 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            dic = json.loads(line)
            for spo in dic['spo_list']:
                sub_type = None
                obj_type = None
                pre = spo['predicate']

                if [obj_type, pre, sub_type] not in schemas:
                    schemas.append([obj_type, pre, sub_type])
                    count_2 +=1
                if pre not in preds:
                    preds.append(pre)
                    count_3 +=1

    print('total line of data', count)
    print('total num of schemas ', count_2)
    print('total num of preds', count_3)

    with open(os.path.join(data_dir, saved_file), 'w', encoding='utf-8') as sf:
        for schema in schemas:
            # print(schema)
            tmp_schema = {'object_type': schema[0], 'predicate': schema[1], 'subject_type': schema[2]}   #三个至少一个不同
            sf.write(json.dumps(tmp_schema, ensure_ascii=False) + "\n")

def assure_predicates(data_dir, data_type):
    '''count # predicates in data file'''
    predicates = set()
    with open(os.path.join(data_dir, '%s_data.json' % data_type), 'r', encoding='utf-8') as f:
        for line in f:
            dic = json.loads(line)
            for spo in dic['spo_list']:
                predicates.add(spo['predicate'])

    return predicates

def replace_subj_obj_by_candi(data_dir, data_type, saved_data_type):
    '''count # predicates in data file'''
    with open(os.path.join(data_dir, '%s_data.json' % saved_data_type), 'w', encoding='utf-8') as fw:
        counter=0
        for line in open(os.path.join(data_dir, '%s_data.json' % data_type), 'r', encoding='utf-8'):
            tmp_dic = json.loads(line)
            counter += 1
            # print(counter, tmp_dic)
            # for spo in tmp_dic['spo_list']:
            #
            #     #to add the candidates to it
            #     if spo['subject'] not in ['?x', '?uri']:
            #         temp_subj = spo['subject']
            #         temp_sub_candidate = spo['subject_candidate']
            #         spo['subject'] = temp_sub_candidate[0]
            #         spo['subject_candidate'].append(temp_subj)
            #
            #     if spo['object'] not in ['?x', '?uri']:
            #         temp_obj = spo['object']
            #         temp_obj_candidate = spo['object_candidate']
            #         spo['object'] = temp_obj_candidate[0]
            #         spo['object_candidate'].append(temp_obj)

            fw.write(json.dumps(tmp_dic, ensure_ascii=False) + "\n")

def check_subj_obj(data_dir, data_type):
    count0 = 0  #total number of subs or objs not in tmp_text
    for line in open(os.path.join(data_dir, '%s_data.json' % data_type), 'r', encoding='utf-8'):
        tmp_dic = json.loads(line)
        for spo in tmp_dic['spo_list']:
            if len(spo['subject']) == 0 or spo['subject'] not in tmp_dic['text']:
                print(spo['subject'], tmp_dic['text'],tmp_dic['id'])
                count0 += 1

            if len(spo['object']) == 0 or spo['object'] not in tmp_dic['text']:
                print(spo['object'], tmp_dic['text'], tmp_dic['id'])
                count0 += 1

    print('total number of subjs or objs not in text', count0)

def pre_process_train(data_dir, pre_train, pre_dev, pre_test, data_type1, data_type2):
    '''hard-coding, add unseen predicates/text（in dev and test) to the pre in train, and return the corresponding predicates for train'''
    tmp_text_for_train=[]
    with open(os.path.join(data_dir, '%s_data.json' % data_type1), 'r', encoding='utf-8') as f:
        for line in f:
            tmp_dic = json.loads(line)
            for spo in tmp_dic['spo_list']:
                if spo['predicate'] in pre_dev and spo['predicate'] not in pre_train:
                    tmp_text_for_train.append(tmp_dic)
                    pre_train.add(spo['predicate'])
                    break

    with open(os.path.join(data_dir, '%s_data.json' % data_type2), 'r', encoding='utf-8') as f:
        for line in f:
            tmp_dic = json.loads(line)
            for spo in tmp_dic['spo_list']:
                if spo['predicate'] in pre_test and spo['predicate'] not in pre_train:
                    tmp_text_for_train.append(tmp_dic)
                    pre_train.add(spo['predicate'])
                    break

    return tmp_text_for_train

def replace_longblank(text):
    return text.replace('  ', ' ')

def text_append(data_dir, data_type, saved_data_type, text_to_write_in=None):
    '''append ?x, ?uri to text, and write some more text to train data if exists'''
    with open(os.path.join(data_dir, '%s_data.json' % saved_data_type), 'w', encoding='utf-8') as fw:
        for line in open(os.path.join(data_dir, '%s_data.json'% data_type), 'r', encoding='utf-8'):
            tmp_dic = json.loads(line)
            tmp_text = tmp_dic['text']
            # tmp_text = replace_longblank(tmp_text)

            tmp_dic['text'] = tmp_text + ' ' + '?x' + ' ' + '?uri'

            # if tmp_text['predicate:'] not in predicates_in:
            #     text_in.append(tmp_text)
            #     predicates_in.add(tmp_text['predicate:'])
            #     print('predicates not in train', tmp_text['predicate:'])
            #     continue

            fw.write(json.dumps(tmp_dic, ensure_ascii=False) + "\n")

        if text_to_write_in is not None:
            print('which data_type gets append', data_type)
            for text in text_to_write_in:
                tmp_text = text['text']
                # tmp_text = replace_longblank(tmp_text)

                text['text'] = tmp_text + ' ' + '?x' + ' ' + '?uri'
                fw.write(json.dumps(text, ensure_ascii=False) + "\n")

        # if text_to_write is not None:
        #     print('data_type', data_type)
        #     with open(os.path.join(data_dir, '%s_data.json' % saved_data_type), 'a', encoding='utf-8') as fw:
        #         for text in text_to_write:
        #             fw.write(str(text))
        #             fw.write('\n')

def put_labels_in_predicates(data_dir, data_type, saved_data_type, reversed_predicate_mapper):
    with open(os.path.join(data_dir, '%s_data.json' % saved_data_type), 'w', encoding='utf-8') as fw:
        for line in open(os.path.join(data_dir, '%s_data.json' % data_type), 'r', encoding='utf-8'):
            tmp_dic = json.loads(line)

            for spo in tmp_dic['spo_list']:
                tmp_predicate = spo['predicate']
                spo['predicate'] = reversed_new_case_predicate_uri_mapper[tmp_predicate]
                spo['real_predicate'] = tmp_predicate

            fw.write(json.dumps(tmp_dic, ensure_ascii=False) + "\n")

def save_to_file(data_dir, split_dir, saved_data_type, data_list):

    if not os.path.isdir(os.path.join(data_dir, split_dir)):
        os.makedirs(os.path.join(data_dir, split_dir))

    with open(os.path.join(data_dir, split_dir, '%s_data.json' % saved_data_type), 'w', encoding='utf-8') as fs:
       for tmp_list in data_list:
           fs.write(json.dumps(tmp_list, ensure_ascii=False) + "\n")
       print(len(data_list))

if __name__ == "__main__":
    data_dir = '/Users/Zixuan/Downloads/FreebaseQA/'
    #data_type = 'FreebaseQA-train'
    #saved_data_type ='trial'
    #reverse_linking(data_dir, data_type, saved_data_type)

    reverse_linking(data_dir, data_type='FreebaseQA-train', saved_data_type='train_data')
    reverse_linking(data_dir, data_type='FreebaseQA-dev', saved_data_type='dev_data')
    reverse_linking(data_dir, data_type='FreebaseQA-eval', saved_data_type='test_data')

    print(data_dir)

    new_schemas = 'schemas'

    replace_subj_obj_by_candi(data_dir, 'train', 'train_re') # train_dev_2_part_verb_2, combine train, train2, dev, dev2
    replace_subj_obj_by_candi(data_dir, 'dev', 'dev_re')  # dev_2
    replace_subj_obj_by_candi(data_dir, 'test', 'test_re')  # test_2

    #create the schemas
    create_schemas(data_dir, 'train_re', new_schemas)
    #read the predicates
    predicates_in_train = assure_predicates(data_dir, 'train_re')
    predicates_in_dev = assure_predicates(data_dir, 'dev_re')
    predicates_in_test = assure_predicates(data_dir, 'test_re')

    print('# predicates in train, dev, test', len(predicates_in_train), len(predicates_in_dev), len(predicates_in_test))
    #find unseen predicates for train
    tmp_text_for_train = pre_process_train(data_dir, predicates_in_train, predicates_in_dev, predicates_in_test,
                                           data_type1='dev_re', data_type2='test_re')
    print('unseen predicates in dev/test not in train', len(tmp_text_for_train))

    #append ?x and ?uri
    text_append(data_dir, 'train_re', 'train_re_pp', tmp_text_for_train)
    text_append(data_dir, 'dev_re', 'dev_re_pp')
    text_append(data_dir, 'test_re', 'test_re_pp')

    create_schemas(data_dir, 'train_re_pp', new_schemas)
    # create_schemas(data_dir, 'train_re_pp', new_schemas)

    predicates_in_train = assure_predicates(data_dir, 'train_re_pp')
    print('new predicates in train', len(predicates_in_train))

    check_subj_obj(data_dir, 'train_re_pp')