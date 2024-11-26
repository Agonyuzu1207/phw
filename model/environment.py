import networkx as nx
from collections import defaultdict
import numpy as np
import torch, numpy, pickle, random, time, argparse
from tqdm import tqdm


class Env(object):
    def __init__(self, examples, config, padding, jump, maxn, transformer_space=None):
        """Temporal Knowledge Graph Environment.
        examples: quadruples (subject, relation, object, timestamps);
        config: config dict;
        state_action_space: Pre-processed action space;
        """
        self.config = config
        self.num_rel = config['num_rel']
        self.label2nodes, self.neighbors = self.prepare_data(examples)
        self.nebor_relation = self.built_nebor_relation(examples)
        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2  # Padding relation.
        self.tPAD = 0  # Padding time
        self.confPAD = 1  # Padding time
        self.padding = padding
        self.jump = jump
        self.maxn = maxn
        # self.state_action_space = state_action_space  # Pre-processed action space
        self.transformer_space = transformer_space
        self.transformer_space_key = []
        if transformer_space:
            self.transformer_space_key = self.transformer_space.keys()

    def prepare_data(self, examples):
        label2nodes = defaultdict(set)
        neighbors = defaultdict(dict)
        examples.sort(key=lambda x: x[3], reverse=True)  # Reverse chronological order
        for example in tqdm(examples, desc="开始built_graph"):
            src = example[0]
            rel = example[1]
            dst = example[2]
            time = example[3]
            conf = example[4]

            src_node = (src, time)
            dst_node = (dst, time)
            src_node_conf = (src, time, conf)
            dst_node_conf = (dst, time, conf)

            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)

            # 为transformer做准备
            try:
                neighbors[src_node][rel].add(dst_node_conf)
            except KeyError:
                neighbors[src_node][rel] = set([dst_node_conf])
            try:
                neighbors[dst_node][rel + self.num_rel].add(src_node_conf)
            except KeyError:
                neighbors[dst_node][rel + self.num_rel] = set([src_node_conf])

        """需要对neighbors里面的tail根据conf值进行排序，由大到小排序"""
        for h, t in neighbors:
            # neighbors[(h, t)] = {r: list(ts) for r, ts in neighbors[(h, t)].items()}  # 生成邻居信息

            for r, ts_tuples in neighbors[(h, t)].items():
                # 将元组列表转换为列表的列表（每个内部列表包含一个元组转换成的列表）
                ts_lists = [t for t in ts_tuples]

                # 根据每个内部列表的第三个值（索引为2）进行排序（由大到小）
                sorted_ts_lists = sorted(ts_lists, key=lambda x: x[-1], reverse=True)

                # 更新 neighbors[(h, t)] 中的值
                neighbors[(h, t)][r] = sorted_ts_lists
            # print(neighbors[(h, t)])

        return label2nodes, neighbors

    def built_nebor_relation(self, examples):
        """The graph node is represented as (entity, time), and the edges are directed and labeled relation.
        return:
            graph: nx.MultiDiGraph;
            label2nodes: a dict [keys -> entities, value-> nodes in the graph (entity, time)]
        """
        nebor_relation = torch.ones(self.config['num_ent'], 2 * self.num_rel + 1)
        for head, relation, tail, timestamp, conf in tqdm(examples, desc='正在生成nebor_relation'):
            nebor_relation[head][relation] += 1
            nebor_relation[head][relation + self.num_rel] += 1

        first_elemnt = {key[0]: key for key in self.neighbors.keys()}
        for e in range(self.config['num_ent']):  # 由于此处使用的是所有的数据，不存在找不到的情况
            if e not in first_elemnt.keys():  # 不在train训练集中
                nebor_relation[e][2 * self.num_rel] += 1
        nebor_relation = torch.log(nebor_relation)
        nebor_relation /= nebor_relation.sum(1).unsqueeze(1)

        return nebor_relation

    def extract_without_token(self, head, timestamp):
        MAXN = self.maxn
        PADDING = self.padding
        JUMP = self.jump
        subgraph_entity = [head]
        subgraph_time = [timestamp]
        subgraph_relation = [self.rPAD]
        subgraph_conf = [1]

        relation = []
        length = [0]
        for jump in range(JUMP):
            length.append(len(subgraph_entity))
            for parent in range(length[jump], length[jump + 1]):
                now_entity = subgraph_entity[parent]
                now_time = subgraph_time[parent]
                nodes = self.label2nodes[now_entity].copy()
                nodes = list(filter((lambda x: x[1] <= now_time), nodes))
                nodes.sort(key=lambda x: x[1], reverse=True)
                # print(nodes)

                for node in nodes:
                    if len(subgraph_entity) > PADDING:
                        break
                    for r in self.neighbors[node]:
                        if len(self.neighbors[node][r]) > MAXN:
                            # print(f'J{node}-{r}-{len(self.neighbors[node][r])}', end=' ')
                            # continue
                            random.shuffle(self.neighbors[node][r])
                        # print(self.neighbors[node][r])
                        for ent, t, conf in self.neighbors[node][r][:MAXN]:
                            try:
                                pos = subgraph_entity.index(ent)  # 在subgraph中找到t的索引  防止重复加入list中
                                relation.append((parent, pos, r))
                            except ValueError:
                                subgraph_entity.append(ent)
                                subgraph_time.append(t)
                                subgraph_relation.append(r)
                                subgraph_conf.append(conf)
                                pos = subgraph_entity.index(ent)
                                relation.append((parent, pos, r))
                    if (len(subgraph_entity) != len(subgraph_time) or len(subgraph_entity) != len(subgraph_relation) or
                            len(subgraph_entity) != len(subgraph_conf)):
                        print("error")

        length.append(len(subgraph_entity))
        if length[-1] > PADDING or not relation:  # subgraph is too big
            subgraph_entity = subgraph_entity[:PADDING]
            subgraph_time = subgraph_time[:PADDING]
            subgraph_relation = subgraph_relation[:PADDING]
            subgraph_conf = subgraph_conf[:PADDING]
            length[-1] = len(subgraph_entity)
        RELA = set()
        # 如果在上面的subgraph可能多余padding个，在上面的if中已经取出来实体部分，现在需要对relation也进行去除
        for i, j, r in relation:
            if i < PADDING and j < PADDING:
                RELA.add((i, j, r))
                inv_r = r + self.num_rel * ((r < self.num_rel) * 2 - 1)
                RELA.add((j, i, inv_r))
        # rela_mat = numpy.array(list(RELA)).T

        indices = torch.tensor(list(RELA)).t()
        values = torch.ones(len(RELA))
        size = (self.padding, self.padding, 2 * self.num_rel + 1)
        if len(RELA) == 0:
            rela_mat = torch.zeros(self.padding, self.padding, 2 * self.num_rel + 1)
        else:
            rela_mat = torch.sparse_coo_tensor(indices, values, size)
            rela_mat = rela_mat.coalesce()

        return (torch.LongTensor(subgraph_entity), torch.LongTensor(subgraph_time), torch.LongTensor(subgraph_relation),
                torch.FloatTensor(subgraph_conf),
                rela_mat, torch.LongTensor(length))

    def getsubgraph(self, H, timestamp):
        if (H, timestamp) not in self.transformer_space_key:
            # print(H, timestamp, '不是key')  # 在代码正确的时候依旧可能出现不是key的情况，这里就直接再次生成就好了
            subgraph_entity, subgraph_time, subgraph_relation, subgraph_conf, relations, length = self.extract_without_token(
                H, timestamp)
        else:
            subgraph_entity, subgraph_time, subgraph_relation, subgraph_conf, relations, length = \
                self.transformer_space[(H, timestamp)]


        return subgraph_entity, subgraph_time, subgraph_relation, subgraph_conf, relations, length


    def get_subgraphs_transformer(self, heads, timestamps, rels, mode):
        if self.config['cuda']:
            heads = heads.cpu()
            timestamps = timestamps.cpu()
            rels = rels.cpu()

        trgs = torch.stack([heads, timestamps, rels], dim=1).to(torch.long).to('cuda')
        heads = heads.numpy()
        timestamps = timestamps.numpy()

        indices_list = []
        values_list = []
        batch_size = heads.shape[0]

        subgraph_entitys = torch.full((batch_size, self.padding), self.ePAD, dtype=torch.long, device='cuda')
        subgraph_times = torch.full((batch_size, self.padding), self.tPAD, dtype=torch.long, device='cuda')
        subgraph_rels = torch.full((batch_size, self.padding), self.rPAD, dtype=torch.long, device='cuda')
        subgraph_confs = torch.full((batch_size, self.padding), self.confPAD, dtype=torch.float, device='cuda')
        # if mode == "transformer" or "noRL":
        #     rela_mats = torch.full((batch_size, self.padding, self.padding, 2 * self.num_rel + 1), 1, dtype=torch.float, device='cuda')
        # else:
        #     rela_mats = None
        rela_mats = None
        lengths = torch.full((batch_size, self.jump+2), 1, dtype=torch.long, device='cuda')

        # time2 = time.time()
        for i in range(heads.shape[0]):
            # time1 = time.time()
            subgraph_entity, subgraph_time, subgraph_rel, subgraph_conf, rela_mat, length = self.getsubgraph(heads[i], timestamps[i])
            # print("获取一次的时间为", time.time() - time1)                  # 获取一次的时间为 0.002287626266479492
            # time2 = time.time()
            subgraph_entitys[i, 0:subgraph_entity.shape[0]] = subgraph_entity
            subgraph_times[i, 0:subgraph_time.shape[0]] = subgraph_time
            subgraph_rels[i, 0:subgraph_rel.shape[0]] = subgraph_rel
            subgraph_confs[i, 0:subgraph_conf.shape[0]] = subgraph_conf
            lengths[i, :] = length
            if mode == 'transformer':
                if not rela_mat.is_sparse:
                    rela_mat = rela_mat.to_sparse()
                rela_indices = rela_mat.coalesce().indices()  # 稀疏矩阵的索引
                rela_values = rela_mat.coalesce().values()  # 稀疏矩阵的值
                # 在索引中添加批量维度
                batch_indices = torch.cat([torch.full((1, rela_indices.shape[1]), i, dtype=torch.long), rela_indices],
                                          dim=0)

                indices_list.append(batch_indices)
                values_list.append(rela_values)

            # rela_mats[i, :, :, :] = rela_mat.to_dense() if rela_mat.is_sparse else rela_mat      # 为什么不在这里做to_densea因为不在cuda里面
            # print("获取一次的时间为", time.time()-time2)           # 获取一次的时间为 0.005005359649658203
            # print("获取一次的时间为", time.time()-time1)         # 0.008993148803710938
        if mode == 'transformer':
            indices = torch.cat(indices_list, dim=1) if indices_list else torch.empty((4, 0), dtype=torch.long,
                                                                                      device='cuda')
            values = torch.cat(values_list) if values_list else torch.empty((0,), dtype=torch.float, device='cuda')

            rela_mats = torch.sparse_coo_tensor(indices, values,
                                                (batch_size, self.padding, self.padding, 2 * self.num_rel + 1),
                                                device='cuda').to_dense()

        return subgraph_entitys, subgraph_times, subgraph_rels, subgraph_confs, rela_mats, trgs, lengths

