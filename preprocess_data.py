import pickle
import os
import argparse
from model.environment import Env
from dataset.baseDataset import baseDataset,baseDataset_new
from tqdm import tqdm


"""
ICEWS14 padding 70 maxn 40
ICEWS18 padding 70 maxn 40
YAGO  padding 30 maxn 20 jump=5

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess', usage='preprocess_data.py [<args>] [-h | --help]')
    parser.add_argument('--data_dir', default='data/ICEWS14', type=str, help='Path to data.')
    parser.add_argument('--outfile', default='subgraph_2', type=str,
                        help='file to save the preprocessed data.')
    parser.add_argument('--data_new', default=False, type=bool)
    parser.add_argument('-maxN', default=40, type=int)
    parser.add_argument('-jump', default=2, type=int)
    parser.add_argument('-padding', default=140, type=int)
    args = parser.parse_args()

    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None
    dataset = baseDataset(trainF, testF, statF, validF)

    train_new_F = os.path.join(args.data_dir, 'train_new.txt')
    test_new_F = os.path.join(args.data_dir, 'test_new.txt')
    valid_new_F = os.path.join(args.data_dir, 'valid_new.txt')
    dataset_new = baseDataset_new(train_new_F, test_new_F, valid_new_F)

    config = {
        'num_rel': dataset.num_r,
        'num_ent': dataset.num_e,
    }
    env = Env(dataset_new.allQuadruples, config, args.padding, args.jump, args.maxN)
    state_actions_space = {}
    subgraph = dict()
    false = []

    # timestamps = list(dataset.get_all_timestamps())
    print(args)
    with tqdm(total=len(dataset_new.allQuadruples)) as bar:
        for (head, rel, tail, t, _) in dataset_new.allQuadruples:
            if (head, t) not in subgraph.keys():
                res = env.extract_without_token(head, t)
                try:
                    subgraph[(head, t)] = res
                except Exception:
                    print("出现问题")
            if (tail, t) not in subgraph.keys():
                res = env.extract_without_token(tail, t)
                try:
                    subgraph[(tail, t)] = res
                except Exception:
                    print("出现问题")
            bar.update(1)
    print("保存的文件为：",os.path.join(args.data_dir, f'subgraph{args.jump}'))
    with open(os.path.join(args.data_dir, f'subgraph1{args.jump}'), 'wb') as db:
        pickle.dump(subgraph, db)

