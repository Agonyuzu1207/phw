import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.baseDataset import baseDataset, QuadruplesDataset, baseDataset_new
from model.agent import Agent
from model.MLPencode import MLPAgent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
from model.env2 import Env2
# from model.dirichlet import Dirichlet
from transformer.Models import Transformer
from transformer.Translator import Translator

import os
import pickle
import time

"""
--data_path data/ICEWS14 --cuda --do_train --do_test --jump=2 --padding=140  --savestep=5 --desc=ICEWS14
--data_path data/ICEWS18 --cuda --do_train --do_test --jump=2 --padding=140  --savestep=5 --desc=ICEWS18
--data_path data/YAGO --cuda --do_train --do_test --jump=5 --padding=140 --maxN=20  --n_head=6 --d_v=64  --savestep=5 --desc=YAGO
--data_path data/ICEWS05-15 --cuda --do_train --do_test --jump=2 --padding=140  --savestep=5 --desc=ICEWS05-15
--data_path data/GDELT --cuda --do_train --do_test --jump=2 --padding=140  --savestep=5 --desc=GDELT
"""

"""
当模型运行时间过长，大概率是模型太大了，可以将模型的embedding设置小一点，padding设置小一点
"""

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Forecasting Models',
        usage='main.py [<args>] [-h | --help]'
    )


    parser.add_argument('--cuda', action='store_true', help='whether to use GPU or not.')
    parser.add_argument('--data_path', type=str, default='data/ICEWS18', help='Path to data.')
    parser.add_argument('--do_train', action='store_true', help='whether to train.')
    parser.add_argument('--do_test', action='store_true', help='whether to test.')
    parser.add_argument('--save_path', default='logs', type=str, help='log and model save path.')
    parser.add_argument('--load_model_path', default='logs', type=str, help='trained model checkpoint path.')

    # Train Params
    parser.add_argument('--batch_size', default=512, type=int, help='training batch size.')
    parser.add_argument('--max_epochs', default=400, type=int, help='max training epochs.')
    # parser.add_argument('--num_workers', default=2, type=int, help='workers number used for dataloader.')
    parser.add_argument('--valid_epoch', default=30, type=int, help='validation frequency.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate.')
    parser.add_argument('--save_epoch', default=30, type=int, help='model saving frequency.')
    parser.add_argument('--clip_gradient', default=10.0, type=float, help='for gradient crop.')

    # Test Params
    parser.add_argument('--test_batch_size', default=256, type=int,
                        help='test batch size, it needs to be set to 1 when using IM module.')
    parser.add_argument('--beam_size', default=1, type=int, help='the beam number of the beam search.')
    parser.add_argument('--test_inductive', action='store_true',
                        help='whether to verify inductive inference performance.')

    # Environment Params
    parser.add_argument('--transformer_space_path', default='subgraph', type=str,
                        help='the file stores preprocessed candidate action array.')
    parser.add_argument('--state_actions_path', default='state_actions_space.pkl', type=str,
                        help='the file stores preprocessed candidate action array.')

    # Episode Params
    parser.add_argument('--path_length', default=3, type=int, help='the agent search path length.')

    # Policy Gradient Params
    parser.add_argument('--Lambda', default=0.0, type=float, help='update rate of baseline.')
    parser.add_argument('--Gamma', default=0.95, type=float, help='discount factor of Bellman Eq.')
    parser.add_argument('--Ita', default=0.01, type=float, help='regular proportionality constant.')
    parser.add_argument('--Zita', default=0.9, type=float, help='attenuation factor of entropy regular term.')

    # reward shaping params
    parser.add_argument('--k', default=300, type=int, help='statistics recent K historical snapshots.')
    parser.add_argument('--data_new', default=False, type=bool)

    # preprocess_data
    parser.add_argument('--maxN', default=40, type=int)
    parser.add_argument('--jump', default=2, type=int)
    parser.add_argument('--padding', default=50, type=int)

    # train_mode
    parser.add_argument('--transformer', default=True, type=bool)

    # transformer param setting
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--exps', type=str, default='EXPS/')
    parser.add_argument('--subgraph', type=str, default='')
    parser.add_argument('--savestep', type=int)

    # Transformers parameter
    parser.add_argument('--d_v', type=int, default=25)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)

    parser.add_argument('--warmup', '--n_warmup_steps', type=int, default=400)
    parser.add_argument('--lr_mul', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=31)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embs_share_weight', action='store_true')
    parser.add_argument('--proj_share_weight', action='store_true')
    parser.add_argument('--scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--desc', type=str, required=True)

    parser.add_argument('--output', default='pred.txt',
                        help="""Path to output the predictions (each line will be the decoded sequence""")

    # mode
    parser.add_argument('--train_mode', type=str, default='lstm')

    opt = parser.parse_args(args)
    opt.d_k = opt.d_v
    opt.d_inner_hid = opt.d_model = opt.d_k * opt.n_head
    opt.desc += f'/{opt.path_length}_{opt.train_mode}/j{opt.jump}'
    opt.desc += time.strftime("_%Y%m%d_%H_%M_%S", time.localtime())
    # opt.exps = os.path.join(opt.exps, opt.desc)
    opt.data = opt.data_path
    opt.subgraph = opt.data + f'/subgraph{opt.jump}' if opt.subgraph == '' else opt.data + f'/subgraph{opt.subgraph}'
    # os.mkdir(opt.exps)
    # with open(opt.exps + '/options.txt', 'w') as option:
    #     for k, v in sorted(opt.__dict__.items(), key=lambda x: x[0]):
    #         option.write(f'{k} = {v}\n')
    # logfile = opt.exps + '/log.txt'
    opt.ent_dim = opt.d_model
    opt.rel_dim = opt.d_model
    opt.state_dim = opt.d_model
    opt.hidden_dim = opt.d_model
    opt.d_word_vec = opt.d_model

    return opt


def get_model_config(args, num_ent, num_rel, num_time, device):
    config = {
        'cuda': args.cuda,  # whether to use GPU or not.
        'batch_size': args.batch_size,  # training batch size.
        'num_ent': num_ent,  # number of entities
        'num_rel': num_rel,  # number of relations
        'num_time': num_time,  # number of relations
        'ent_dim': args.ent_dim,  # Embedding dimension of the entities
        'rel_dim': args.rel_dim,  # Embedding dimension of the relations
        'state_dim': args.state_dim,  # dimension of the LSTM hidden state
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions
        'time_dim': args.ent_dim // 5,
        'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer
        'path_length': args.path_length,  # agent search path length
        # 'max_action_num': args.max_action_num,  # max candidate action number
        'lambda': args.Lambda,  # update rate of baseline
        'gamma': args.Gamma,  # discount factor of Bellman Eq.
        'ita': args.Ita,  # regular proportionality constant
        'zita': args.Zita,  # attenuation factor of entropy regular term
        'beam_size': args.beam_size,  # beam size for beam search
        # 'entities_embeds_method': args.entities_embeds_method,
        # default: 'dynamic', otherwise static encoder will be used
        'data_new': args.data_new,
        'device': device,
        'train_mode': args.train_mode
    }
    return config


def load_model(opt, device, nebor_relation):
    model = Transformer(
        opt.src_vocab_size,  # 实体的数目
        opt.trg_vocab_size,  # 关系的数目
        opt.src_pad_idx,  # pad的编号    该文章中为0，titer中为num_rel
        -1,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,  # arg中的值
        emb_src_trg_weight_sharing=opt.embs_share_weight,  # arg中的值
        d_k=opt.d_k,  # d_k = d_v 表示transformer中的参数 50
        d_v=opt.d_v,
        d_model=opt.d_model,  # opt.d_model = opt.d_k * opt.n_head 应该是词向量的维度
        d_word_vec=opt.d_word_vec,  # opt.d_word_vec = opt.d_model
        d_inner=opt.d_inner_hid,  # opt.d_inner_hid = opt.d_model = opt.d_k * opt.n_head
        n_layers=opt.n_layers,  # 表示encode层数 2
        n_head=opt.n_head,  # 多头注意力机制中的头数 2
        dropout=opt.dropout,  # dropout 遗忘率 0.1
        scale_emb_or_prj=opt.scale_emb_or_prj,  # arg中的值
        n_position=opt.padding,  # arg中的值  140表示action的长度
        data=opt.data,  # 数据文件的路径
        opt=opt,
        mode=opt.train_mode,
    ).to(device)  # base_data.nebor_relation basedata中的参数
    return model


def main(args):
    #######################Set Logger#################################
    args.save_path = os.path.join(args.save_path, args.desc)
    print(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    #######################Create DataLoader#################################
    train_path = os.path.join(args.data_path, 'train.txt')
    test_path = os.path.join(args.data_path, 'test.txt')
    stat_path = os.path.join(args.data_path, 'stat.txt')
    valid_path = os.path.join(args.data_path, 'valid.txt')

    baseData = baseDataset(train_path, test_path, stat_path, valid_path)

    trainDataset = QuadruplesDataset(baseData.trainQuadruples, baseData.num_r)
    train_dataloader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=args.num_workers,
    )

    validDataset = QuadruplesDataset(baseData.validQuadruples, baseData.num_r)
    valid_dataloader = DataLoader(
        validDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
    )

    testDataset = QuadruplesDataset(baseData.testQuadruples, baseData.num_r)
    test_dataloader = DataLoader(
        testDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
    )

    device = torch.device('cuda')
    ######################Creat the agent and the environment###########################
    config = get_model_config(args, baseData.num_e, baseData.num_r, baseData.num_time, device)
    args.src_vocab_size = baseData.num_e + 1
    args.trg_vocab_size = 2 * baseData.num_r + 1
    args.src_pad_idx = baseData.num_e

    logging.info(config)
    logging.info(args)

    transformer_space_path = os.path.join(args.data_path, args.transformer_space_path +str(1) + str(args.jump))
    transformer_space = None
    # if not os.path.exists(transformer_space_path):
    #     transformer_space = None
    # else:
    #     transformer_space = pickle.load(open(transformer_space_path, 'rb'))
    #     print('导入transformer数据成功，数据路径为：', transformer_space_path)

    state_action_space = None
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(open(os.path.join(args.data_path, args.state_actions_path), 'rb'))

    train_new_F = os.path.join(args.data_path, 'train_new.txt')
    test_new_F = os.path.join(args.data_path, 'test_new.txt')
    valid_new_F = os.path.join(args.data_path, 'valid_new.txt')
    dataset_new = baseDataset_new(train_new_F, test_new_F, valid_new_F)


    if args.train_mode == 'lstm':
        env = Env2(dataset_new.trainQuadruples, config, state_action_space)
    else:
        print("采用的是transformer的环境")
        env = Env(dataset_new.trainQuadruples, config, args.padding, args.jump, args.maxN, transformer_space)

    # creat the environment,agent
    if args.train_mode == 'lstm':
        print("采用的是lstm编码")
        agent = Agent(config)
    elif args.train_mode == 'MLP':
        print("采用MLP编码")
        agent = MLPAgent(config)
    else:
        print("采用的是transformer编码")
        agent = Translator(
            model=load_model(args, device, env.nebor_relation),
            opt=args,
            config=config,
            device=device,
        ).to(device)

    episode = Episode(env, agent, config)

    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    # Load the model parameters
    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        episode.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    ######################Training and Testing###########################
    distributions = None
    gradient_accumulation_steps = 10
    trainer = Trainer(episode, pg, optimizer, args, gradient_accumulation_steps, distributions)
    tester = Tester(episode, args, baseData.train_entities)
    if args.do_train:
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            loss, reward = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

            if i % args.save_epoch == 0 and i != 0:
            # if i % args.save_epoch == 0:
                trainer.save_model('checkpoint_{}.pth'.format(i))
                logging.info('Save Model in {}'.format(args.save_path))

            if i % args.valid_epoch == 0 and i != 0:
            # if i % args.valid_epoch == 0:
                logging.info('Start Val......')
                metrics = tester.test(valid_dataloader,
                                      validDataset.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))

        trainer.save_model()
        logging.info('Save Model in {}'.format(args.save_path))

    if args.do_test:
        logging.info('Start Testing......')
        metrics = tester.test(test_dataloader,
                              testDataset.__len__(),
                              baseData.skip_dict,
                              config['num_ent'])
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_args()
    main(args)

