import json
import pickle
import logging
import random
import time
from tqdm import tqdm

import numpy as np
import os
from collections import defaultdict
from scipy.sparse import coo_matrix


def load_stat(path):
    with open(path, 'r') as f:
        num_entity, num_relation, _ = f.read().split()
    return int(num_entity), int(num_relation)


def load_rule(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class Generate_new_truples():
    def __init__(self, datadir, ruledir, axiom_weight=1.0, max_entialments=5):
        self.sparsity = 0.995
        self.axiom_weight = axiom_weight
        self.max_entialments = max_entialments
        self.time_higher = 5

        self.entity = set()
        self.relation = set()
        self.timestamp = set()

        self.train = self.load_quadruples(os.path.join(datadir, 'train.txt'))
        self.rules = load_rule(ruledir)
        self.num_entity, self.num_relation = load_stat(os.path.join(datadir, 'stat.txt'))

        self.entity2frequency, self.entity2sparsity = self._entity2frequency(self.train)

        self.p_sot, self.spt_o, self.sp_ot = self._generate()

    def load_quadruples(self, path):
        with open(path, 'r') as f:
            quadrupleList = []
            for line in f:
                try:
                    line_split = line.split()
                    head = int(line_split[0])
                    rel = int(line_split[1])
                    tail = int(line_split[2])
                    time = int(line_split[3])
                    quadrupleList.append([head, rel, tail, time])
                    self.entity.add(head)
                    self.entity.add(tail)
                    self.relation.add(rel)
                    self.timestamp.add(time)
                except:
                    print(line)
        return quadrupleList

    def _entity2frequency(self, examples):
        ent2freq = {ent: 0 for ent in range(self.num_entity)}
        ent2sparsity = {ent: -1 for ent in range(self.num_entity)}
        for h, r, t, _ in examples:
            ent2freq[h] += 1
            ent2freq[t] += 1

        max_freq = max(list(ent2freq))
        min_freq = min(list(ent2freq))
        for ent, freq in ent2freq.items():
            sparsity = 1 - (freq - min_freq) / (max_freq - min_freq)
            ent2sparsity[ent] = sparsity
        return ent2freq, ent2sparsity

    def _generate(self):
        p_sot = defaultdict(set)
        spt_o = defaultdict(set)
        sp_ot = defaultdict(set)

        for s, p, o, t in self.train:
            p_sot[p].add((s, p, t))
            spt_o[(s, p, t)].add(o)
            sp_ot[(s, p)].add((o, t))
        return p_sot, spt_o, sp_ot

    def random_data(self, data, lengths):
        selected_data = list(data)  # 将集合转换为列表
        # 检查数据是否足够100个，如果不足就获取全部
        if len(selected_data) > lengths:
            # 随机抽取100个元素
            sample_data = random.sample(selected_data, lengths)
        else:
            # 如果少于100个，直接返回所有元素
            sample_data = selected_data
        return sample_data

    def materialize_sparse(self, axioms, length=100):
        inference = []  # 表示所有的满足规则的三元组
        # axiom2entailment is a dict
        # with the all axioms in the axiom pool as keys
        # and all the entailments for each axiom as values
        axiom_list = axioms
        max_entailments = self.max_entialments

        for axiom in tqdm(axiom_list, desc='新的四元组生成中'):
            inference_tmp = []
            if axiom['conf'] < 0.5:
                continue
            # print(axiom['type'])

            if axiom['type'] == 'symmetric':
                body_rels = axiom['body_rels'][0]
                for (s, o, t) in self.random_data(self.p_sot[body_rels], length):
                    if len(inference_tmp) > max_entailments:
                        break
                    if (o, s, t) not in self.p_sot[body_rels] and (
                            self.entity2sparsity[s] > self.sparsity or self.entity2sparsity[o] > self.sparsity
                    ):
                        inference_tmp.append([(o, body_rels, s, t), axiom['id'], axiom['conf']])

            if axiom['type'] == 'inverse':
                body_rels = axiom['body_rels'][0]
                head_rel = axiom['head_rel']
                for (s, o, t1) in self.random_data(self.p_sot[body_rels], length):
                    if len(inference_tmp) > max_entailments:
                        break
                    for i in range(self.time_higher):
                        if (o, s, t1 + i) not in self.p_sot[head_rel] and (
                                self.entity2sparsity[s] > self.sparsity or self.entity2sparsity[o] > self.sparsity
                        ):
                            inference_tmp.append([(o, head_rel, s, t1 + i), axiom['id'], axiom['conf']])
                            break

            if axiom['type'] == 'equivalent':
                body_rels = axiom['body_rels'][0]
                head_rel = axiom['head_rel']
                for (s, o, t) in self.random_data(self.p_sot[body_rels], length):
                    if len(inference_tmp) > max_entailments:
                        break
                    for i in range(self.time_higher):
                        if (s, o, t + i) not in self.p_sot[head_rel] and (
                                self.entity2sparsity[s] > self.sparsity or self.entity2sparsity[o] > self.sparsity
                        ):
                            inference_tmp.append([(s, head_rel, o, t + i), axiom['id'], axiom['conf']])
                            break

            if axiom['type'] == 'transitive':
                p1, p2 = axiom['body_rels']
                head_rel = axiom['head_rel']
                for (s, o, t) in self.random_data(self.p_sot[p1], length):
                    if len(inference_tmp) > max_entailments:
                        break
                    for e in self.random_data(self.spt_o[(o, p1, t)] - self.spt_o[(s, p1, t)], length):
                        if e != s and (
                                self.entity2sparsity[s] > self.sparsity or self.entity2sparsity[e] > self.sparsity
                        ):
                            inference_tmp.append([(s, p1, e, t), axiom['id'], axiom['conf']])

            if axiom['type'] == 'composition':
                p1, p2 = axiom['body_rels']
                head_rel = axiom['head_rel']
                for (s, o, t1) in self.random_data(self.p_sot[p1], length):
                    if len(inference_tmp) > max_entailments:
                        break
                    for (e, t2) in self.random_data(self.sp_ot[(o, p2)], length):
                        for i in range(self.time_higher):
                            t = max(t1, t2) + i
                            if (s, e, t) not in self.p_sot[head_rel] and (
                                    self.entity2sparsity[s] > self.sparsity or self.entity2sparsity[e] > self.sparsity
                            ):
                                inference_tmp.append([(s, head_rel, e, t), axiom['id'], axiom['conf']])
                                break

            inference += inference_tmp

        return inference, len(inference)


if __name__ == '__main__':
    datadir = 'data/GDELT'
    rule_dir = 'Rule/GDELT/GDELT.json'
    # save_file = os.path.join(datadir, 'train_new.json')
    generator = Generate_new_truples(datadir, rule_dir)
    rule_triples, rule_num = generator.materialize_sparse(generator.rules)

    new_triple_path = os.path.join(datadir, 'train_new.txt')
    new_triple_label_path = os.path.join(datadir, 'train_label.pickle')

    triple_label = defaultdict(int)

    new_triples = []
    for data in generator.train:
        new_triples.append([data[0], data[1], data[2], data[3], 1])
        # triple_label[(data[0], data[1], data[2], data[3])] = -1

    for rule_triple in rule_triples:
        (s, p, o, t), axiom_id, axiom_conf = rule_triple
        new_triples.append([s, p, o, t, axiom_conf])
        # triple_label[(s, p, o, t)] = axiom_id

    with open(new_triple_path, 'w') as file:
        for sublist in new_triples:
            line = ' '.join(map(str, sublist))
            file.write(line + '\n')

    # pickle.dump(triple_label, open(new_triple_label_path, 'wb'))
