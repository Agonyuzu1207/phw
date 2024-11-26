''' This module will handle the text generation with beam search. '''
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask
from collections import defaultdict
import numpy, time



class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super(HistoryEncoder, self).__init__()
        self.config = config
        self.lstm_cell = torch.nn.LSTMCell(input_size=config['action_dim'],
                                           hidden_size=config['state_dim'])

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx

class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        self.mlp_l1 = nn.Linear(config['mlp_input_dim'], config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['ent_dim'], bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = self.mlp_l2(hidden).unsqueeze(1)
        return output

class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(self, model, opt=None, config=None, device=None, gama=0.7):

        super(Translator, self).__init__()

        self.gama = 0.7
        self.max_seq_len = opt.jump + 1
        self.src_pad_idx = opt.src_pad_idx
        self.n_trg_vocab = opt.trg_vocab_size
        self.n_src_vocab = opt.src_vocab_size  # 实体的个数
        self.jump = opt.jump
        self.padding = opt.padding
        self.device = device
        self.config = config
        self.r_PAD = config['num_rel'] * 2

        self.model = model
        self.model.train()

        self.rel_mlp = nn.Linear(self.jump * self.n_trg_vocab, self.padding, bias=True)
        self.policy_step = HistoryEncoder(config)
        self.policy_mlp = PolicyMLP(config)
        self.score_weighted_fc = nn.Linear(
            self.config['ent_dim'] * 2 + self.config['rel_dim'] + self.config['state_dim'],
            1, bias=True)
        self.trg_word_prj = nn.Linear(opt.d_model + self.config['ent_dim'], self.n_trg_vocab, bias=False)

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = None
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # print("dec_output",dec_output.shape)    # dec_output torch.Size([8, 1, 200])
        # print("self.agent_embds",self.agent_embds.shape)    # self.agent_embds torch.Size([8, 1, 200])

        dec_output = torch.cat([dec_output, self.agent_embds.repeat(1,dec_output.shape[1],1)], dim=-1) # batch_size,step,d_model+ent_dim
        return F.softmax(self.trg_word_prj(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_time_seq, src_mask, link=None, length=None):
        enc_output = None

        enc_output, *_ = self.model.encoder(src_seq, src_time_seq, src_mask, link=link, length=length)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(1)

        scores = torch.log(best_k_probs).view(self.batch_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx.squeeze()

        return enc_output, gen_seq, scores, dec_output

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step, link=None):
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(1)
        assert dec_output.size(1) == step
        gen_seq[:, step] = best_k2_idx.squeeze()

        return gen_seq, scores, dec_output

    def forward(self, query_timestamps, src_seq, src_time_seq, current_relation, query_entity, query_relation, trg=None, link=None, length=None):
        batch_size = trg.size(0)
        self.batch_size = batch_size
        self.init_seq = trg[:, 2].unsqueeze(-1).clone().detach()
        self.blank_seqs = trg[:, 2].unsqueeze(-1).repeat(1, self.max_seq_len).clone().detach()

        # embeddings
        current_entities = src_seq[:, 0]
        current_time = src_time_seq[:, 0]
        current_delta_time = query_timestamps - current_time
        current_embds = self.model.entityE(current_entities, current_delta_time)  # [batch_size, ent_dim]
        prev_relation_embs = self.model.relationE(current_relation)
        query_entity_embds = self.model.entityE(query_entity, torch.zeros_like(query_timestamps))
        query_relation_embds = self.model.relationE(query_relation)

        # History Encode
        NO_OP_mask = torch.eq(current_relation, torch.ones_like(current_relation) * self.r_PAD)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        prev_action_embedding = torch.cat([prev_relation_embs, current_embds],
                                          dim=-1)  # [batch_size, rel_dim + ent_dim]
        lstm_output = self.policy_step(prev_action_embedding, NO_OP_mask)  # [batch_size, state_dim]

        # agent state representation
        agent_state = torch.cat([lstm_output, query_entity_embds, query_relation_embds],
                                dim=-1)  # [batch_size, state_dim + ent_dim + rel_dim]
        self.agent_embds = self.policy_mlp(agent_state)  # [batch_size, ent_dim]

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        current_delta_times = query_timestamps.unsqueeze(-1).expand(src_time_seq.shape) - src_time_seq
        enc_output, gen_seq, scores, dec_output = self._get_init_state(src_seq, current_delta_times, src_mask, link=link, length=length)

        for step in range(2, self.max_seq_len):
            dec_output = self._model_decode(gen_seq[:, :step].clone().detach(), enc_output, src_mask)
            # print("查看dec",dec_output.size())  # 查看dec torch.Size([16, 2, 461])
            gen_seq, scores, dec_output = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step, link=link)
        # print(dec_output.shape)

        rel_mlp_input = dec_output.reshape(dec_output.shape[0], dec_output.shape[1] * dec_output.shape[2])
        scores_rel = self.rel_mlp(rel_mlp_input)

        temp = self.model.entityE(trg[:, 0], query_timestamps - trg[:, -1]).unsqueeze(1).expand(
            enc_output.shape)
        scores_ent = F.cosine_similarity(enc_output, temp, dim=-1)

        # scoring
        # entitis_output = output[:, :, self.config['rel_dim']:]
        # relation_ouput = output[:, :, :self.config['rel_dim']]
        # print("查看enc_output",enc_output.shape)
        # print("查看relation_output",relation_ouput.shape)
        # print("查看entity_output",entitis_output.shape)
        # scores_all = torch.sum(torch.mul(enc_output, output), dim=2)
        # entities_score = torch.sum(torch.mul(enc_output, entitis_output), dim=2)  # [batch_size, action_number]
        # actions = enc_output  # [batch_size, action_number, action_dim]

        # agent_state_repeats = agent_state.unsqueeze(1).repeat(1, enc_output.shape[1], 1)
        # score_attention_input = torch.cat([enc_output, agent_state_repeats], dim=-1)
        # a = self.score_weighted_fc(score_attention_input)
        # a = torch.sigmoid(a).squeeze()  # [batch_size, action_number]

        # scores_all = (1 - a) * relation_score + a * entities_score
        scores_all = self.gama * scores_ent + (1 - self.gama) * scores_rel  # torch.Size([1, 140])

        pad_mask = torch.ones_like(src_seq) * self.src_pad_idx  # [batch_size, action_number]
        pad_mask = torch.eq(src_seq, pad_mask)  # [batch_size, action_number]
        scores_all = scores_all.masked_fill(pad_mask, -1e10)
        action_prob = torch.softmax(scores_all, dim=-1)
        action_id = torch.multinomial(action_prob, 1)  # Randomly select an action. [batch_size, 1]
        logits = torch.nn.functional.log_softmax(scores_all, dim=1)  # [batch_size, action_number]
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)

        return loss, logits, action_id


