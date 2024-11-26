import json
import time

import torch
import pickle
import torch.nn as nn
from collections import defaultdict


class Episode(nn.Module):
    def __init__(self, env, agent, config):
        super(Episode, self).__init__()
        self.config = config
        self.env = env
        self.agent = agent
        self.path_length = config['path_length']
        self.num_rel = config['num_rel']

    def forward(self, query_entities, query_timestamps, query_relations):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            all_loss: list
            all_logits: list
            all_actions_idx: list
            current_entities: torch.tensor, [batch_size]
            current_timestamps: torch.tensor, [batch_size]
        """
        query_entities_embeds = None
        query_relations_embeds = None
        prev_relations = torch.ones_like(query_relations) * self.num_rel * 2
        self.agent.policy_step.set_hiddenx(query_relations.shape[0])

        if self.config["train_mode"] == 'lstm':
            query_entities_embeds = self.agent.ent_embs(query_entities,torch.zeros_like(query_timestamps))
            query_relations_embeds = self.agent.rel_embs(query_relations)
            prev_relations = torch.ones_like(query_relations) * self.num_rel

        heads = query_entities
        relations = query_relations
        timestamps = query_timestamps

        all_loss = []
        all_logits = []
        all_actions_idx = []
        all_reward = []

        for t in range(self.path_length):
            # start_time = time.time()
            # 使用transformer进行编码
            if self.config['train_mode'] == "lstm":
                subgraph_entitys, subgraph_times, subgraph_rels, subgraph_confs, rela_mats, trgs, lengths = self.env.get_subgraphs_transformer(heads,timestamps,query_timestamps)
            else:
                subgraph_entitys, subgraph_times, subgraph_rels, subgraph_confs, rela_mats, trgs, lengths = self.env.get_subgraphs_transformer(
                    heads, timestamps, relations, self.config['train_mode']
                )


            # print(t,'次的获取子图的时间',time.time()-start_time)         # 0 次的获取子图的时间 0.08743619918823242
            # 0次的获取子图的时间0.014414072036743164       0.4896383285522461   0.07900142669677734     0 次的获取子图的时间 0.08699870109558105
            # 0次的获取子图的时间0.03599953651428223
            # time2 = time.time()

            """写一个translator的获取规则矩阵的方法"""
            if self.config['train_mode'] == "lstm" or self.config["train_mode"] == 'MLP':
                loss, logits, action_id = self.agent(
                    prev_relations,
                    heads,
                    timestamps,
                    query_relations_embeds,
                    query_entities_embeds,
                    query_timestamps,
                    subgraph_entitys,
                    subgraph_times,
                    subgraph_rels
                )
            else:
                loss, logits, action_id = self.agent(
                    query_timestamps,
                    subgraph_entitys,
                    subgraph_times,
                    prev_relations,
                    query_entities,
                    query_relations,
                    trgs,
                    rela_mats,
                    lengths)

            chosen_relation = torch.gather(subgraph_rels, dim=1, index=action_id).reshape(
                subgraph_rels.shape[0])
            prev_relations = chosen_relation
            chosen_entity = torch.gather(subgraph_entitys, dim=1, index=action_id).reshape(
                subgraph_entitys.shape[0])
            chosen_entity_timestamps = torch.gather(subgraph_times, dim=1, index=action_id).reshape(
                subgraph_times.shape[0])
            if self.config['train_mode'] != "lstm":
                chosen_rewards = torch.gather(subgraph_confs, dim=1, index=action_id).reshape(
                    subgraph_times.shape[0])
            # print(action_id.shape)                                               # torch.Size([batch_size, 1])
            # print(heads.shape,chosen_entity.shape,chosen_relation.shape,chosen_entity_timestamps.shape) # [batch_size]
            """heads,chosen_relation,chosen_entity,chosen_entity_timestamps 表示选择走的四元组，那么该元组的评分需要判断"""

            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)
            if self.config['train_mode'] != "lstm":
                all_reward.append(chosen_rewards)

            heads = chosen_entity
            timestamps = chosen_entity_timestamps
            relations = chosen_relation
            # print(t,'model运行',time.time()-time2)
            # 0次的模型花的时间0.017003774642944336
            # 0次的花的时间0.051766395568847656
            # 1次的花的时间0.05309605598449707
            # 2次的花的时间0.05301189422607422

        return all_loss, all_logits, all_actions_idx, all_reward, heads, timestamps

    def beam_search(self, query_entities, query_timestamps, query_relations):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            current_entites: [batch_size, test_rollouts_num]
            beam_prob: [batch_size, test_rollouts_num]
        """
        batch_size = query_entities.shape[0]
        query_entities_embeds = None
        query_relations_embeds = None
        prev_relations = torch.ones_like(query_relations) * self.num_rel * 2
        self.agent.policy_step.set_hiddenx(query_relations.shape[0])
        if self.config['train_mode'] == "lstm":
            query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
            query_relations_embeds = self.agent.rel_embs(query_relations)
            prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP

        # In the first step, if rollouts_num is greater than the maximum number of actions, select all actions
        heads = query_entities
        relations = query_relations
        timestamps = query_timestamps
        if self.config['train_mode'] == "lstm":
            subgraph_entitys, subgraph_times, subgraph_rels, subgraph_confs, rela_mats, trgs, lengths = self.env.get_subgraphs_transformer(
                heads, timestamps, query_timestamps)
        else:
            subgraph_entitys, subgraph_times, subgraph_rels, subgraph_confs, rela_mats, trgs, lengths = self.env.get_subgraphs_transformer(
                heads, timestamps, relations, self.config['train_mode']
            )

        if self.config['train_mode'] == "lstm" or self.config["train_mode"] == 'MLP':
            loss, logits, action_id = self.agent(
                prev_relations,
                heads,
                timestamps,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                subgraph_entitys,
                subgraph_times,
                subgraph_rels
            )
        else:
            loss, logits, action_id = self.agent(
                query_timestamps,
                subgraph_entitys,
                subgraph_times,
                prev_relations,
                query_entities,
                query_relations,
                trgs,
                rela_mats,
                lengths)

        action_space_size = subgraph_entitys.shape[-1]
        #
        # target = torch.full_like(subgraph_entitys, self.config["num_ent"],
        #                          device='cuda:0')  # [batch_size,action_num]
        # matches = torch.eq(subgraph_entitys, target)
        # count = torch.sum(matches)
        # count = count.cpu().item()

        if self.config['beam_size'] > action_space_size:
            beam_size = action_space_size
        else:
            beam_size = self.config['beam_size']

        # if action_space_size - count < beam_size:
        #     beam_size = action_space_size - count

        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size,
                                                    dim=1)  # beam_log_prob.shape [batch_size, beam_size]
        beam_log_prob = beam_log_prob.reshape(-1)  # [batch_size * beam_size] # 将所有的概率全部整合到一个维度里面

        heads = torch.gather(subgraph_entitys, dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]

        timestamps = torch.gather(subgraph_times, dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]
        relations = torch.gather(subgraph_rels, dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]
        if self.config["train_mode"] == 'MLP':
            prev_relations = relations
        else:
            self.agent.policy_step.hx = self.agent.policy_step.hx.repeat(1, 1, beam_size).reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]
            self.agent.policy_step.cx = self.agent.policy_step.cx.repeat(1, 1, beam_size).reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]
            prev_relations = relations

        beam_tmp = logits.reshape(batch_size, -1)  # [batch_size, beam_size]
        # print("查看一下第一次的beam_tmp",beam_tmp.size())
        # print("第一次的subgraph", subgraph_entitys.reshape(batch_size, -1).size())
        for t in range(1, self.path_length):
            query_timestamps_roll = query_timestamps.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_entities_roll = query_entities.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_relations_roll = query_relations.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_relations_embeds_roll = None
            query_entities_embeds_roll = None
            if self.config['train_mode'] == "lstm":
                query_entities_embeds_roll = query_entities_embeds.repeat(1, 1, beam_size)
                query_entities_embeds_roll = query_entities_embeds_roll.reshape(
                    [batch_size * beam_size, -1])  # [batch_size * beam_size, ent_dim]
                query_relations_embeds_roll = query_relations_embeds.repeat(1, 1, beam_size)
                query_relations_embeds_roll = query_relations_embeds_roll.reshape(
                    [batch_size * beam_size, -1])  # [batch_size * beam_size, rel_dim]

            if self.config['train_mode'] == "lstm":
                subgraph_entitys, subgraph_times, subgraph_rels, subgraph_confs, rela_mats, trgs, lengths = self.env.get_subgraphs_transformer(heads,timestamps,query_timestamps_roll)
            else:
                subgraph_entitys, subgraph_times, subgraph_rels, subgraph_confs, rela_mats, trgs, lengths = self.env.get_subgraphs_transformer(
                    heads, timestamps, relations, self.config['train_mode']
                )

            if self.config["train_mode"] == "lstm":
                loss, logits, action_id = self.agent(
                    prev_relations,
                    heads,
                    timestamps,
                    query_relations_embeds_roll,
                    query_entities_embeds_roll,
                    query_timestamps_roll,
                    subgraph_entitys,
                    subgraph_times,
                    subgraph_rels
                )
                hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
                cx_tmp = self.agent.policy_step.cx.reshape(batch_size, beam_size, -1)
            elif self.config["train_mode"] == 'MLP':
                loss, logits, action_id = self.agent(
                    prev_relations,
                    heads,
                    timestamps,
                    query_relations_embeds_roll,
                    query_entities_embeds_roll,
                    query_timestamps_roll,
                    subgraph_entitys,
                    subgraph_times,
                    subgraph_rels
                )
            else:
                loss, logits, action_id = self.agent(
                    query_timestamps_roll,
                    subgraph_entitys,
                    subgraph_times,
                    prev_relations,
                    query_entities_roll,
                    query_relations_roll,
                    trgs,
                    rela_mats,
                    lengths)
                hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
                cx_tmp = self.agent.policy_step.cx.reshape(batch_size, beam_size, -1)
            # logits.shape [bs * rollouts_num, max_action_num]

            beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0)  # [batch_size * beam_size, max_action_num]

            beam_tmp += logits
            beam_tmp = beam_tmp.reshape(batch_size, -1)  # [batch_size, beam_size * max_actions_num]

            if action_space_size * beam_size >= self.config['beam_size']:
                beam_size = self.config['beam_size']
            else:
                beam_size = action_space_size * beam_size

            # if beam_tmp.shape[1] - count < beam_size:
            #     beam_size = beam_tmp.shape[1] - count

            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, beam_size, dim=1)  # [batch_size, beam_size]
            if self.config['train_mode'] == 'lstm' or self.config['train_mode'] == 'transformer':
                offset = top_k_action_id // action_space_size  # [batch_size, beam_size]
                offset = offset.unsqueeze(-1).repeat(1, 1, self.config['state_dim'])  # [batch_size, beam_size]
                self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset)
                self.agent.policy_step.hx = self.agent.policy_step.hx.reshape([batch_size * beam_size, -1])
                self.agent.policy_step.cx = torch.gather(cx_tmp, dim=1, index=offset)
                self.agent.policy_step.cx = self.agent.policy_step.cx.reshape([batch_size * beam_size, -1])

            heads = torch.gather(subgraph_entitys.reshape(batch_size, -1), dim=1,
                                 index=top_k_action_id).reshape(-1)
            timestamps = torch.gather(subgraph_times.reshape(batch_size, -1), dim=1,
                                      index=top_k_action_id).reshape(-1)
            relations = torch.gather(subgraph_rels.reshape(batch_size, -1), dim=1,
                                     index=top_k_action_id).reshape(-1)
            prev_relations = relations
            beam_log_prob = top_k_log_prob.reshape(-1)  # [batch_size * beam_size]
        # print("最后的beam_tmp的情况", beam_tmp.size)
        # print("最后的subgraph",subgraph_entitys.reshape(batch_size, -1).size())
        return subgraph_entitys.reshape(batch_size, -1), beam_tmp
