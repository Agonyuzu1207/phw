import json

import torch
import pickle
import numpy as np
import math
from model.baseline import ReactiveBaseline


def load_rule(rule_label, rule_path):
    rule_label = pickle.load(open(rule_label, 'rb'))
    rules = json.load(open(rule_path, 'r'))
    return rule_label, rules


def entropy_reg_loss(all_logits):
    all_logits = torch.stack(all_logits, dim=2)
    entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))
    return entropy_loss


class PG(object):
    def __init__(self, config, rule_label='data/ICEWS14/train_label.pickle',rule_path='Rule/ICEWS14/ICEWS14.json'):
        self.config = config
        self.positive_reward = 1.0
        self.negative_reward = 0.0
        self.baseline = ReactiveBaseline(config, config['lambda'])
        self.now_epoch = 0
        if self.config['data_new']:
            self.train_label, self.rules = load_rule(rule_label, rule_path)

    def get_reward(self, current_entites, answers, src_entitys, rel_batch, time_batch):
        positive = torch.ones_like(current_entites, dtype=torch.float32) * self.positive_reward
        negative = torch.ones_like(current_entites, dtype=torch.float32) * self.negative_reward
        reward = torch.where(current_entites == answers, positive, negative)

        return reward

    def calc_cum_discounted_reward(self, rewards, reward_conf):
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.config['path_length']])
        if self.config['cuda']:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config['path_length'] - 1] = rewards
        for t in reversed(range(self.config['path_length'])):
            running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add

        return cum_disc_reward

    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward):
        loss = torch.stack(all_loss, dim=1)
        base_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - base_value

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)
        entropy_loss = self.config['ita'] * math.pow(self.config['zita'], self.now_epoch) * entropy_reg_loss(
            all_logits)
        # 理解entropy是正则化的损失
        total_loss = torch.mean(loss) - entropy_loss
        return total_loss
