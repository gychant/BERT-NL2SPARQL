import random
import numpy as np
import os
import sys
from tqdm import tqdm
import json
import copy
import inflect

#predefined
plural_engine = inflect.engine()
plural_engine.defnoun("headquarter", "headquarters")  #add the special case
plural_engine.defnoun("specy", "species")  #add the special case
plural_engine.defnoun("sery", "series")  #add the special case

def how_many_schema(data_dir, schema_file):

    schema_set = set()  # i just wanna to know how many schemas
    schema_list = []
    with open(os.path.join(data_dir, schema_file), 'r', encoding='utf-8') as schema:
        for schema_line in schema:
            tmp_schema_line = json.loads(schema_line)
            schema_list.append(
                [tmp_schema_line['object_type'], tmp_schema_line['predicate'], tmp_schema_line['subject_type']])
            schema_set.add(tmp_schema_line['predicate'])
        print('total num of triples', len(schema_list), 'total num of predicates', len(schema_set))

    return schema_set

def ontology_property_interchange(schema_set):
    '''find the predicate label(exclude the uri part, know ontology<->property'''

    # ontology <-> property
    predicate_collectors = []  # get all labels
    for predicate_uri in schema_set:
        predicate = ''
        for char in predicate_uri[::-1]:
            if char == '/':
                break
            else:
                predicate += char
        predicate = predicate[::-1]  # predicate=predicate[::-1].lower()
        predicate_collectors.append(predicate)

    predicate_collectors = set(predicate_collectors)  # remove duplicate

    # create predicate uri mapper e.g. {battle: ['http://dbpedia.org/ontology/battle',
    # 'http://dbpedia.org/property/battle']}
    predicate_uri_mapper = {}
    for p_c in predicate_collectors:
        predicate_uri_mapper[p_c] = []

    for predicate_uri in schema_set:
        tmp_char = ''
        for char in predicate_uri[::-1]:
            if char == '/':
                break
            else:
                tmp_char += char
        tmp_char = tmp_char[::-1]   #the label

        for p_c in predicate_collectors:
            if p_c == tmp_char:
                predicate_uri_mapper[p_c].append(predicate_uri)

    return predicate_uri_mapper

def plural_problem(predicate_uri_mapper):
    '''create dict resolve plural problem
    e.g. {battle: ['http://dbpedia.org/ontology/battle', http://dbpedia.org/property/battles']}'''

    new_predicate_uri_mapper = copy.deepcopy(predicate_uri_mapper)

    for key1, value1 in predicate_uri_mapper.items():
        tmp_word = plural_engine.singular_noun(key1)  # return false if it is a single
        if tmp_word:
            for key2, value2 in predicate_uri_mapper.items():
                if key2 == tmp_word:
                    #                 print(key2)
                    for tmp_value1 in value1:
                        new_predicate_uri_mapper[key2].append(tmp_value1)
                    # print(key2, new_predicate_uri_mapper[key2])
                    del new_predicate_uri_mapper[key1]
        else:
            pass
    print('revised plural', len(new_predicate_uri_mapper))

    return new_predicate_uri_mapper

def case_problem(new_predicate_uri_mapper):
    '''create dict resolve case problem,
    e.g. {battle ['http://dbpedia.org/ontology/battle', 'http://dbpedia.org/property/Battle']}'''

    new_case_predicate_uri_mapper = copy.deepcopy(new_predicate_uri_mapper)
    for key3, value3 in new_predicate_uri_mapper.items():
        for key4, value4 in new_predicate_uri_mapper.items():
            if key3 == key4:
                pass
            else:
                if key3.lower() == key4.lower() and key4 in new_case_predicate_uri_mapper.keys():
                    for tmp_value3 in value3:
                        new_case_predicate_uri_mapper[key4].append(tmp_value3)
                    # print(key4, new_case_predicate_uri_mapper[key4])
                    del new_case_predicate_uri_mapper[key3]
    print('revised case', len(new_case_predicate_uri_mapper))
    return new_case_predicate_uri_mapper

def uri2labels(reversed_new_case_predicate_uri_mapper):
    '''{label: uri}'''
    forward_new_case_predicate_uri_mapper = {}
    for key, value in reversed_new_case_predicate_uri_mapper.items():
        if value not in forward_new_case_predicate_uri_mapper.keys():
            forward_new_case_predicate_uri_mapper[value] = [key]
        else:
            forward_new_case_predicate_uri_mapper[value].append(key)
    print('uri to labels', len(forward_new_case_predicate_uri_mapper))

    return forward_new_case_predicate_uri_mapper

def labels2uri(new_case_predicate_uri_mapper):
    '''{uri: label}'''
    reversed_new_case_predicate_uri_mapper = {}
    for key, value in new_case_predicate_uri_mapper.items():
        for val in value:
            if val not in reversed_new_case_predicate_uri_mapper.keys():
                reversed_new_case_predicate_uri_mapper[val] = key
            else:
                raise ValueError('bad dict val goes into key error')
    print('labels to uri', len(reversed_new_case_predicate_uri_mapper))

    return reversed_new_case_predicate_uri_mapper

# test
if __name__ == "__main__":
    data_dir = 'data/lcquad/'
    schema_file = 'schemas'
    schema_set = how_many_schema(data_dir, schema_file)
    predicate_uri_mapper = ontology_property_interchange(schema_set)
    new_predicate_uri_mapper = plural_problem(predicate_uri_mapper)
    new_case_predicate_uri_mapper = case_problem(new_predicate_uri_mapper)

    reversed_new_case_predicate_uri_mapper = labels2uri(new_case_predicate_uri_mapper)
    forward_new_case_predicate_uri_mapper = uri2labels(reversed_new_case_predicate_uri_mapper)

    # for key, value in new_case_predicate_uri_mapper.items():
    #     print(key, value)