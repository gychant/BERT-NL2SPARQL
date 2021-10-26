import random
import numpy as np
import os
import sys
from tqdm import tqdm
import json
from copy import deepcopy

from schema_chill import how_many_schema, ontology_property_interchange, plural_problem, case_problem, labels2uri

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
                sub_type = spo['subject_type']
                obj_type = spo['object_type']
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
            counter+=1
            # print(counter, tmp_dic)
            for spo in tmp_dic['spo_list']:

                #to add the candidates to it
                if spo['subject'] not in ['?x', '?uri']:
                    temp_subj = spo['subject']
                    temp_sub_candidate = spo['subject_candidate']
                    spo['subject'] = temp_sub_candidate[0]
                    spo['subject_candidate'].append(temp_subj)

                if spo['object'] not in ['?x', '?uri']:
                    temp_obj = spo['object']
                    temp_obj_candidate = spo['object_candidate']
                    spo['object'] = temp_obj_candidate[0]
                    spo['object_candidate'].append(temp_obj)

            fw.write(json.dumps(tmp_dic, ensure_ascii=False) + "\n")

def check_subj_obj(data_dir, data_type):
    count0 = 0  #total number of subs or objs not in tmp_text
    for line in open(os.path.join(data_dir, '%s_data.json' % data_type), 'r', encoding='utf-8'):
        tmp_dic = json.loads(line)
        for spo in tmp_dic['spo_list']:
            if spo['subject'] not in tmp_dic['text']:
                print(spo['subject'], tmp_dic['text'],tmp_dic['id'])
                count0 += 1

            if spo['object'] not in tmp_dic['text']:
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

def cross_validation_n_fold(data_dir, split_dir='revised',data_type='full_pp', seed=0, num_folds=5):

    count_total=0
    full_list = []
    for line in open(os.path.join(data_dir, split_dir, '%s_data.json'% data_type), 'r', encoding='utf-8'):
        tmp_dic = json.loads(line)
        count_total+=1
        full_list.append(tmp_dic)
    print('total num of lines in data', count_total)
    list_num = list(range(count_total))

    random.seed(seed)
    random.shuffle(list_num)
    # print(full_list[:5])

    split_threshold =[count_total*0.8, count_total*0.9] # % in train, dev, test

    for i in range(5, 10):
        train_list_num = list_num[:int(split_threshold[0])+1]
        dev_list_num = list_num[int(split_threshold[0])+1:int(split_threshold[1])+1]
        test_list_num = list_num[int(split_threshold[1])+1:]

        train_data_list = [full_list[idx] for idx in train_list_num]
        save_to_file(data_dir, split_dir + '/' + str(i), 'train_pp', train_data_list)

        dev_data_list = [full_list[idx] for idx in dev_list_num]
        save_to_file(data_dir, split_dir + '/' + str(i), 'dev_pp', dev_data_list)

        test_data_list = [full_list[idx] for idx in test_list_num]
        save_to_file(data_dir, split_dir + '/' + str(i), 'test_pp', test_data_list)

        random.shuffle(list_num)

def unify_fk_blank(data_dir, data_type, saved_data_type):
    '''unify the rt blankspace'''
    # this is no longer neeeded
    with open(os.path.join(data_dir, '%s_data.json' % saved_data_type), 'w', encoding='utf-8') as fw:
        for line in open(os.path.join(data_dir, '%s_data.json'% data_type), 'r', encoding='utf-8'):
            tmp_dic = json.loads(line)

            tmp_text = tmp_dic['text']
            tmp_dic['text'] = tmp_text.replace(' - ', '-').replace(' – ', '–')
            for spo in tmp_dic['spo_list']:
                tmp_sub = spo['subject']
                tmp_obj = spo['object']

                spo['subject'] = tmp_sub.replace(' - ', '-').replace(' – ', '–')
                spo['object'] = tmp_obj.replace(' - ', '-').replace(' – ', '–')
            fw.write(json.dumps(tmp_dic, ensure_ascii=False) + "\n")

def add_nstype(data_dir1, data_type1, data_dir2, data_type2, saved_data_type):
    '''add nstype back to the sentence'''  #先统计一下不出现的概率, 再  # add back, 约7-9%ns#type不出现在句子中
    data_file1 = open(os.path.join(data_dir1, '%s_data.json' % data_type1), 'r', encoding='utf-8') #to be processed
    data_file2 = open(os.path.join(data_dir2, '%s_data.json' % data_type2), 'r', encoding='utf-8') #contains ns#type

    data1_list=[]
    for line in data_file1:
        tmp_dic = json.loads(line)
        data1_list.append(tmp_dic)
    data2_list=[]
    for line in data_file2:
        tmp_dic = json.loads(line)
        data2_list.append(tmp_dic)
    print(len(data1_list), len(data2_list))
    count = 0
    count2 = 0
    with open(os.path.join(data_dir1, '%s_data.json' % saved_data_type), 'w', encoding='utf-8') as fw:
        for i in range(len(data1_list)):
            #print(data1_list[i])
            for spo in data2_list[i]['spo_list']:
                if 'ns#type' in spo['predicate']:   #包含ns#type的spo
                    if spo['object'] in data1_list[i]['text']:   #同时在原文中有
                        data1_list[i]['spo_list'].append(spo)
                        count+=1
                        # print(count)

        for tmp_list in data1_list:
            fw.write(json.dumps(tmp_list, ensure_ascii=False) + "\n")
            count2+=1
    print(count, count2)

if __name__ == "__main__":
    # 1. normal pre-process
    # data_dir = '/root/ie4sparql/lic-verb-70-threshold-label'
    data_dir = '/Users/Zixuan/IE4SPARQL/lic-corrected-70-semantic-embedd-v4/revised/'

    print(data_dir)

    old_schemas = 'schemas_old'
    new_schemas = 'schemas'

    replace_subj_obj_by_candi(data_dir, 'train_dev_2_part_verb_2', 'train_re') # train_dev_2_part_verb_2, combine train, train2, dev, dev2
    replace_subj_obj_by_candi(data_dir, 'dev_2', 'dev_re')  # dev_2
    replace_subj_obj_by_candi(data_dir, 'test_2', 'test_re')  # test_2

    #create the schemas
    create_schemas(data_dir, 'train_re', old_schemas)
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

    create_schemas(data_dir, 'train_re_pp', old_schemas)
    # create_schemas(data_dir, 'train_re_pp', new_schemas)

    predicates_in_train = assure_predicates(data_dir, 'train_re_pp')
    print('new predicates in train', len(predicates_in_train))

    check_subj_obj(data_dir, 'train_re_pp')

    # merge label test
    combine_labels = True
    if combine_labels:
        print('Combining schema test......................................................................')
        schema_set = how_many_schema(data_dir, schema_file=old_schemas)
        predicate_uri_mapper = ontology_property_interchange(schema_set)  # inter-changable ontology <->property
        new_predicate_uri_mapper = plural_problem(predicate_uri_mapper) # combine plural
        new_case_predicate_uri_mapper = case_problem(new_predicate_uri_mapper) # combine cases
        reversed_new_case_predicate_uri_mapper = labels2uri(new_case_predicate_uri_mapper)

        #save forward case predicate uri mapper
        with open(os.path.join(data_dir, 'uri_label_mapper'), 'w', encoding='utf-8') as sfz:
            sfz.write(json.dumps(new_case_predicate_uri_mapper, ensure_ascii=False))
        print('saved uri_label_mapper data file')

        put_labels_in_predicates(data_dir, 'train_re_pp', 'train_re_pp_l', reversed_new_case_predicate_uri_mapper)
        put_labels_in_predicates(data_dir, 'dev_re_pp', 'dev_re_pp_l', reversed_new_case_predicate_uri_mapper)
        put_labels_in_predicates(data_dir, 'test_re_pp', 'test_re_pp_l', reversed_new_case_predicate_uri_mapper)

        predicates_in_train = assure_predicates(data_dir, 'train_re_pp_l')
        print('new predicates in train', len(predicates_in_train))

        create_schemas(data_dir, 'train_re_pp_l', new_schemas)


