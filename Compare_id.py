from tqdm import tqdm
import json
import pprint

def spo_list2triples(spo_list):
    triples = set()
    for spo in spo_list:

        # interchange between ontology and property
        if 'http://dbpedia.org/property' in spo['predicate']:
            spo['predicate'] = spo['predicate'].replace('http://dbpedia.org/property', 'http://dbpedia.org/ontology')

        triples.add((spo['subject'].lower().strip(), spo['predicate'].lower(), spo['object'].lower().strip()))
    return triples

def compare_entites(old_gold_sample, new_sample):
    old_entities_set = set()
    new_entities_set = set()
    for old_goal_triple in old_gold_sample['spo_list']:
        old_entities_set.add(old_goal_triple['subject'])
        old_entities_set.add(old_goal_triple['object'])
    for new_triple in new_sample['spo_list']:
        new_entities_set.add(new_triple['subject'])
        new_entities_set.add(new_triple['object'])

    correct = old_entities_set.intersection(new_entities_set)

    return correct, old_entities_set

def compare_predicates(old_gold_sample, new_sample):
    old_pred_set = set()
    new_pred_set = set()
    for old_goal_triple in old_gold_sample['spo_list']:
        old_pred_set.add(old_goal_triple['predicate'])
    for new_triple in new_sample['spo_list']:
        new_pred_set.add(new_triple['predicate'])

    correct = old_pred_set.intersection(new_pred_set)

    return correct, old_pred_set


def compare_predicates_wo_entities(old_gold_sample, new_sample):
    old_entities_set = set()
    new_entities_set = set()
    for old_goal_triple in old_gold_sample['spo_list']:
        old_entities_set.add(old_goal_triple['subject'])
        old_entities_set.add(old_goal_triple['object'])
    for new_triple in new_sample['spo_list']:
        new_entities_set.add(new_triple['subject'])
        new_entities_set.add(new_triple['object'])

    if old_entities_set != new_entities_set:
        old_pred_set = set()
        new_pred_set = set()
        for old_goal_triple in old_gold_sample['spo_list']:
            old_pred_set.add(old_goal_triple['predicate'])
        for new_triple in new_sample['spo_list']:
            new_pred_set.add(new_triple['predicate'])

        if new_pred_set == old_pred_set:
            return 1
    return 0


def compare_entities_wo_predicates(old_gold_sample, new_sample):
    old_pred_set = set()
    new_pred_set = set()
    for old_goal_triple in old_gold_sample['spo_list']:
        old_pred_set.add(old_goal_triple['predicate'])
    for new_triple in new_sample['spo_list']:
        new_pred_set.add(new_triple['predicate'])

    if new_pred_set != old_pred_set:
        old_entities_set = set()
        new_entities_set = set()
        for old_goal_triple in old_gold_sample['spo_list']:
            old_entities_set.add(old_goal_triple['subject'])
            old_entities_set.add(old_goal_triple['object'])
        for new_triple in new_sample['spo_list']:
            new_entities_set.add(new_triple['subject'])
            new_entities_set.add(new_triple['object'])

        # if '?x' in old_entities_set:
        #     old_entities_set.remove('?x')
        # if '?uri' in old_entities_set:
        #     old_entities_set.remove('?uri')
        #
        # if '?x' in new_entities_set:
        #     new_entities_set.remove('?x')
        # if '?uri' in new_entities_set:
        #     new_entities_set.remove('?uri')

        if new_entities_set == old_entities_set:
            return 1
    return 0


class CompareAnalyze:
    def compare(self, old_file, new_file, log_file='/root/compare.txt'):
        c_empty = 0
        count = 0
        count_2 = 0
        count_acc = 0
        count_acc_plus = 0
        num_0 = 0
        count_entites_pre = 0
        count_entites_gold = 0
        count_predicates_pre = 0
        count_predicates_gold = 0
        count_ent_wo_pred = 0
        count_pre_wo_ent = 0

        # temp_old_data = open(temp_old_file, 'r', encoding='utf-8')
        compare_file = open(log_file, 'w', encoding='utf-8')
        # old_goal_data = open('/root/lic_ie/novel_tagging/bert-chinese-ner-master/output-5.0-32-5e-05-128-raw_data/json_result/result.json', 'r', encoding='utf-8')
        old_goal_data = open(old_file, 'r', encoding='utf-8')
        # old_predict_data = open('/root/data/train_data_v0/predict_data.json', 'r')
        # with open('/root/lic_ie/novel_tagging/bert-chinese-ner-master/output-5.0-32-5e-05-128-raw_data/json_result/result_post.json', 'r', encoding='utf-8') as new_file:
        with open(new_file, 'r', encoding='utf-8') as new_file:
            for new_line in tqdm(new_file):
                new_sample = json.loads(new_line)
                old_goal_sample = json.loads(old_goal_data.readline())
                # old_predict_sample = json.loads(old_predict_data.readline())
                old_goal_triples = spo_list2triples(old_goal_sample['spo_list'])

                # temp
                # while len(old_goal_triples) == 0:
                #     c_empty += 1
                #     old_goal_sample = json.loads(old_goal_data.readline())
                #     old_goal_triples = spo_list2triples(old_goal_sample['spo_list'])

                # old_goal_triples = set()
                # old_predict_triples = tta.spo_list2triples(old_predict_sample['spo_list'])
                new_triples = spo_list2triples(new_sample['spo_list'])

                # temp_old_sample = json.loads(temp_old_data.readline())

                count += len(new_triples)
                # print(old_goal_sample['text'])
                # assert new_sample['text'] == old_goal_sample['text']
                # if len(set(old_goal_triples).intersection(set(old_predict_triples))) < len(old_predict_triples):
                if old_goal_triples == new_triples:
                    count_acc += 1
                if old_goal_triples != new_triples:

                    # count specific errors
                    # temp_ent_cor, tem_ent_old_pred = compare_entites(old_goal_sample, new_sample)
                    # count_entites_pre += len(temp_ent_cor)
                    # count_entites_gold += len(tem_ent_old_pred)
                    # # compute predicates
                    # temp_pred_cor, tem_pred_old_pred = compare_predicates(old_goal_sample, new_sample)
                    # count_predicates_pre += len(temp_pred_cor)
                    # count_predicates_gold += len(tem_pred_old_pred)

                    # tmp_count_ent_wo_pred = compare_predicates_wo_entities(old_goal_sample, new_sample)
                    # count_ent_wo_pred += tmp_count_ent_wo_pred
                    # tmp_count_pre_wo_ent = compare_entities_wo_predicates(old_goal_sample, new_sample)
                    # count_pre_wo_ent += tmp_count_pre_wo_ent


                    #count the include
                    # for old_goal_triple in old_goal_triples:  # 1
                    #     if old_goal_triple in new_triples:
                    #         for new_triple in new_triples:
                    #             if old_goal_triple == new_triple:
                    #                 count_acc_plus += 1
                    #             # break
                    #         print(old_goal_triples, ' ', new_triples, ' ', new_sample['text'])

                    count_2 += 1
                    # if tmp_count_ent_wo_pred:
                    #     compare_file.write('entity error' + '\n')
                    # elif tmp_count_pre_wo_ent:
                    #     compare_file.write('pred error' + '\n')
                    # else:
                    #     compare_file.write('both error' + '\n')
                    # id
                    compare_file.write(pprint.pformat(new_sample['id']) + '\n')

                    compare_file.write(pprint.pformat(num_0) + '\n')
                    num_0+=1
                    # compare_file.write(pprint.pformat(temp_old_sample['text'])+'\n')

                    compare_file.write(pprint.pformat(old_goal_sample['text'])+'\n')
                    compare_file.write(pprint.pformat(new_sample['text'])+'\n')
                    compare_file.write(pprint.pformat(sorted(old_goal_triples))+'\n')
                    # compare_file.write(print_triplets(old_predict_triples, if_return=True)+'\n')
                    compare_file.write(pprint.pformat(sorted(new_triples))+'\n')
                    compare_file.write('\n')
            print('total num of new triples', count)
            print('total difference between new and old', count_2)
            print('total corrected ', count_acc)
            print('count # includes', count_acc_plus)  #pred more than 1 and includes the correct one
            print('accuracy', count_acc / (count_2+count_acc))
            print('c_empty', c_empty)    #without uri labels
            # print('entities problem', count_entites_pre, count_entites_gold,1-count_entites_pre/(count_entites_gold+1e-10))
            # print('predicates problem', count_predicates_pre, count_predicates_gold, 1-count_predicates_pre/(count_predicates_gold+1e-10))
            # print('count_ent_wo_pred', count_ent_wo_pred)
            # print('count_pre_wo_ent', count_pre_wo_ent)

        compare_file.close()
        old_goal_data.close()

if __name__ == "__main__":
    ca = CompareAnalyze()
    ca.compare(
               # old_file='/Users/Zixuan/IE4SPARQL/Compare/test_2_re_pp_data.json',
               # new_file='/Users/Zixuan/IE4SPARQL/lic-corrected-70-semantic-embedd-v4/revised/test_re_pp_data.json',
               # log_file='/Users/Zixuan/IE4SPARQL/Compare/compare_between_pure_rl_and_with_hannotated/log_w_wo_annotated_4.txt'
               old_file = '/Users/Zixuan/IE4SPARQL/Compare/test_re_pp_data.json',
               new_file = '/Users/Zixuan/IE4SPARQL/Compare/110_result.json',
               log_file = '/Users/Zixuan/IE4SPARQL/Compare/log_110.txt'
               )
