import torch
import json
import os
import tqdm

from collections import defaultdict
import pickle
import time

class Trainer(object):
    def __init__(self, model, pg, optimizer, args, gradient_accumulation_steps=10, distribution=None, rule_label='data/ICEWS14/train_label.pickle',
                 rule_path='Rule/ICEWS14/ICEWS14.json'):
        self.model = model
        self.pg = pg
        self.optimizer = optimizer
        self.args = args
        self.distribution = distribution
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.train_label = pickle.load(open(rule_label, 'rb'))
        self.rules = json.load(open(rule_path, 'r'))

    def train_epoch(self, dataloader, ntriple):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        counter = 0
        # print("使用的是transfomer编码" if self.args.transformer else "使用agent编码")
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:
            bar.set_description('Train')

            # rule_score = self.generate_rule_score()

            for step, (src_batch, rel_batch, dst_batch, time_batch) in enumerate(dataloader):

                if self.args.cuda:
                    src_batch = src_batch.cuda()
                    rel_batch = rel_batch.cuda()
                    dst_batch = dst_batch.cuda()
                    time_batch = time_batch.cuda()
                """在这里需要将所有的评分全部算出来"""

                # time1 = time.time()
                all_loss, all_logits, _, reward_conf, current_entities, current_time = self.model(src_batch, time_batch, rel_batch)
                # print('model使用时间', time.time()-time1) # model使用时间 0.09123778343200684
                # time2 = time.time()
                reward = self.pg.get_reward(current_entities, dst_batch, src_batch, rel_batch, time_batch)

                cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward, reward_conf)

                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)

                self.pg.baseline.update(torch.mean(cum_discounted_reward))
                self.pg.now_epoch += 1

                # if self.gradient_accumulation_steps > 1:
                #     reinfore_loss = reinfore_loss / self.gradient_accumulation_steps
                #
                # reinfore_loss.backward()  # 计算梯度
                # if step % self.gradient_accumulation_steps == 0 or step == ntriple:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()

                self.optimizer.zero_grad()
                reinfore_loss.backward()
                if self.args.clip_gradient:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                self.optimizer.step()

                # print("反向传播时间",time.time()-time3)
                # print("总共时间", time.time()-time2)    # 计算损失 0.15115761756896973
                reinfore_loss = reinfore_loss.detach()                              # 想要看loss，必须进行detach，或者items，不能直接看，否则内存会炸
                reward = reward.detach()
                total_loss += reinfore_loss
                total_reward += torch.mean(reward)
                counter += 1
                bar.update(self.args.batch_size)
                bar.set_postfix(loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(reward).item())
        return total_loss / counter, total_reward / counter

    # def generate_rule_score(self):
    #     score = defaultdict(list)
    #     end_score = defaultdict(list)
    #     for rule in self.rules:
    #         # rule_type = rule['type']
    #         # if rule_type in ["symmetric", "inverse", "equivalent", "transitive", "composition"]:
    #         #     index = torch.tensor(rule['body_rels'], device='cuda')
    #         #     embeds = self.agent.rel_embs(index)
    #         if rule['type'] == "symmetric":
    #             index = torch.tensor(rule['body_rels'], device='cuda')
    #             embeds = self.model.agent.rel_embs(index)
    #             identity = torch.ones_like(embeds)
    #             score[rule['type']].append((rule['id'], torch.norm(embeds - identity, p=2)))
    #         if rule['type'] == "inverse":
    #             body_index = torch.tensor(rule['body_rels'], device='cuda')
    #             head_index = torch.tensor(rule['head_rel'], device='cuda')
    #             body_embed = self.model.agent.rel_embs(body_index)
    #             head_embed = self.model.agent.rel_embs(head_index)
    #             identity = torch.ones_like(head_embed)
    #             score[rule['type']].append((rule['id'], torch.norm(body_embed * head_embed - identity, p=2)))
    #         if rule['type'] == "equivalent":
    #             body_index = torch.tensor(rule['body_rels'], device='cuda')
    #             head_index = torch.tensor(rule['head_rel'], device='cuda')
    #             body_embed = self.model.agent.rel_embs(body_index)
    #             head_embed = self.model.agent.rel_embs(head_index)
    #             score[rule['type']].append((rule['id'], torch.norm(body_embed - head_embed, p=2)))
    #         if rule['type'] == "transitive":
    #             body_index = torch.tensor(rule['body_rels'], device='cuda')
    #             head_index = torch.tensor(rule['head_rel'], device='cuda')
    #             body_embed = self.model.agent.rel_embs(body_index)
    #             head_embed = self.model.agent.rel_embs(head_index)
    #             score[rule['type']].append((rule['id'], torch.norm(body_embed[0] * body_embed[-1] - head_embed, p=2)))
    #         if rule['type'] == "composition":
    #             body_index = torch.tensor(rule['body_rels'], device='cuda')
    #             head_index = torch.tensor(rule['head_rel'], device='cuda')
    #             body_embed = self.model.agent.rel_embs(body_index)
    #             head_embed = self.model.agent.rel_embs(head_index)
    #             score[rule['type']].append((rule['id'], torch.norm(body_embed[0] * body_embed[-1] - head_embed, p=2)))
    #
    #     for k, v in score.items():
    #         sorted_data = sorted(v, key=lambda x: x[1], reverse=True)
    #         _, smax = sorted_data[0]
    #         _, smin = sorted_data[-1]
    #         for data in v:
    #             if smax == smin:
    #                 sc = data[1]
    #             else:
    #                 sc = (smax - data[1]) / (smax - smin)
    #             end_score[data[0]] = sc
    #
    #     return end_score

    def save_model(self, checkpoint_path='checkpoint.pth'):
        """Save the parameters of the model and the optimizer,"""
        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.args.save_path, checkpoint_path)
        )
