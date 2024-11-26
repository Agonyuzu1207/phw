import time

import numpy as np
from collections import defaultdict
import pickle
import os
import timeit
import argparse
import json
from tqdm import tqdm


# =========== NOTES ======================
# This riginal version should work if you
# put it inside the dataset dictionary
# ========================================

def readfile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        quadrupleList = []
        r_erot = defaultdict(list)
        eo_rt = defaultdict(set)
        ort_e = defaultdict(set)
        e_ort = defaultdict(set)
        ert_o = defaultdict(set)
        ero_t = defaultdict(set)
        er_ot = defaultdict(set)
        for line in f:
            try:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                r_erot[rel].append((head, rel, tail, time))
                eo_rt[(head, tail)].add((rel, time))
                ort_e[(tail, rel, time)].add(head)
                e_ort[head].add((tail, rel, time))
                ert_o[(head, rel, time)].add(tail)
                ero_t[(head, rel, tail)].add(time)
                er_ot[(head, rel)].add((tail, time))
            except:
                print(line)
    return quadrupleList, r_erot, eo_rt, ort_e, e_ort, ert_o, ero_t, er_ot


def cal_conf_support(triple_data, rules, type, min_conf=0.2, min_bodysupport=20, min_time=2 * 60):
    quadrupleList, r_erot, eo_rt, ort_e, e_ort, ert_o, ero_t, er_ot = triple_data
    rule_list = []
    if type == "composition":
        for r1, r2, r3 in tqdm(rules, desc='composition'):
            rule = {}
            conf = 0
            body_support = 0
            rule_support = 0

            flag = 1
            start = time.time()

            for e1, r1, e2, t1 in r_erot[r1]:

                now_time = time.time()
                if now_time - start >= min_time:
                    flag = 0
                    break

                for e3, t2 in er_ot[(e2, r2)]:
                    body_support += 1
                    temp = 0

                    for t3 in ero_t[(e1, r3, e3)]:
                        if t3 >= t1 and t3 >= t1:
                            rule_support += 1
                            if temp == 0:
                                temp = 1
                            if temp == 1:
                                body_support += 1
            conf = 1.0 * rule_support / body_support
            if flag == 0:
                print("跳过该关系，太久了")
            if flag and conf >= min_conf and body_support >= min_bodysupport:
                rule["head_rel"] = r3
                rule["body_rels"] = [r1, r2]
                rule["var_constraints"] = [[1, 5], [2, 3], [4, 6]]
                rule["conf"] = conf
                rule["rule_supp"] = rule_support
                rule["body_supp"] = body_support
                rule_list.append(rule)

    return sorted(rule_list, key=sort_data)


def sort_data(item):
    # 返回一个元组，首先按照head_rel排序，如果head_rel相同则按照conf降序排序
    return (item["head_rel"], -item["conf"])  # 注意这里用-item["conf"]来实现降序


def renn_data(rule_dict, save_file):
    endlist = []
    grouped_data = {}
    for k, v in rule_dict.items():
        endlist += v

    sorted_data = sorted(endlist, key=sort_data)
    for data in sorted_data:
        head_rel = data['head_rel']
        if head_rel not in grouped_data.keys():
            grouped_data[head_rel] = []
        grouped_data[head_rel].append(data)

    print('开始保存renn数据，保存地址为:' + 'renn' + save_file)
    with open('renn' + save_file, 'w') as f:
        json.dump(grouped_data, f)


def save_json(data, filename):
    print('开始保存数据，保存地址为:' + filename)
    with open(filename, 'w+') as f:
        json.dump(data, f)


# input: triples
# output: possible axioms for each type
# Axioms(dict): {inverse:{...},symmetric: {...}, ... }
def generateAxioms(triple_data, p, g, save_file, batch_size=1000):
    num_samples = 100
    quadrupleList, r_erot, eo_rt, ort_e, e_ort, ert_o, ero_t, er_ot = triple_data
    num_axiom_types = 1
    composition = set()
    count = 0

    for rel in tqdm(r_erot.keys(), desc='pre_axiom'):
        # the number of triples about r to generate axioms
        N = len(r_erot[rel])
        pN = p * N
        num_samples = round(N - N * pow(1 - g, 1 / pN))
        np.random.shuffle(r_erot[rel])
        num_triples = min(num_samples, len(r_erot[rel]))
        # print("num_triples", num_triples)
        erots = r_erot[rel][:num_triples]

        count_triples = 0
        for e1, r1, e2, t1 in erots:
            # 5 composition
            for e3, r2, t2 in e_ort[e2]:
                for r3, t3 in eo_rt[(e1, e3)]:
                    if t3 >= t1 and t3 >= t1:
                        composition.add((r1, r2, r3))
        count += 1
        '''
        if count > 3:
            break
        '''
    ############################验证是否合理 计算conf和support#################################
    start_sta = 0
    while True:
        end_sta = start_sta + batch_size if start_sta + batch_size < len(composition) else len(composition)
        temp_composition = list(composition)[start_sta:end_sta]
        rul_composition = cal_conf_support(triple_data, temp_composition, "composition")
        save_json(rul_composition, 'composition_' + str(start_sta) + save_file)
        if start_sta + batch_size >= len(composition):
            break
        start_sta = start_sta + batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment setup')
    # misc(short for miscellaneous)
    parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default='../data/ICEWS18')
    parser.add_argument('--train_file', dest='train_file', type=str, default='train.txt')
    parser.add_argument('--save_file', dest='save_file', type=str, default='ICEWS18.json')
    parser.add_argument('--save', dest='save', type=str, default='/ICEWS18')
    parser.add_argument('--axiom_probability', dest='axiom_probability', type=float, default=0.5)
    parser.add_argument('--axiom_proportion', dest='axiom_proportion', type=float, default=0.95)
    # dest for parser
    option = parser.parse_args()

    file_train = os.path.join(option.dataset_dir, option.train_file)
    start = timeit.default_timer()
    # keep the axioms with probability larger than p
    p = option.axiom_probability
    # the probability of keep axioms when generating
    g = option.axiom_proportion
    # triples: [(head, rel, tail), ...]
    triple_data = readfile(file_train)  # 读取的数据里面是编号，不是具体的关系
    generateAxioms(triple_data, p, g, option.save_file)
    end = timeit.default_timer()
    print('cost time:', end - start)
