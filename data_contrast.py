import os
import ujson as json
import random
import math
from typing import List, Tuple

import torch

from common import *


# Please note that, the datasets have been already normalized.
length_max_line = 1500
time_max_line = 1000
code_max_line = 720
code_length_max_line = 3 * 1500


code_length_void_val = 0
time_pos_void_val = 0
void_seq_feature_ls = [code_length_void_val, time_pos_void_val]


@time_log
def load_data_contrast_remake(data_tag:str, 
                            data_path:str, 
                            clean_traffic_tag:str,
                            clean_traffic_path:str,
                            shuffle_seed=None, 
                            segment_len=100,
                            pre_train_ratio=0.2,
                            contrast_batch_size=50,
                            train_ratio=0.1,
                            train_number=-1,
                            test_samp_ratio=0,
                            benign_attack_parse_num=(-1, -1),
                            # benign_attack_parse_num=(100000, 100000),
                            benign_attack_samp_num=(-1, -1),
                            # benign_attack_samp_num=(10000, 10000),
            ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, # pre-train batch <seq, non-seq, interaction> 
                       torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, # train data <seq, non-seq, interaction>
                       torch.IntTensor, torch.IntTensor,                        # Label and size of test data
                       torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, # test data <seq, non-seq, interaction>
                       torch.IntTensor, torch.IntTensor]:                       # Label and size of test data
    
    data_target = f'{data_path}/{data_tag}.json'
    clean_data_target = f'{clean_traffic_path}/{clean_traffic_tag}.json'
    if not os.path.exists(data_target):
        logging.warn("Target dataset not found.")
        exit(-1)

    if clean_traffic_tag is None and clean_traffic_path is None:
        logging.info(f'Read attack/benign from {data_target}')
        with open(data_target, 'r') as f:
            d = json.load(f)
            if d['benign'] is None:
                d['benign'] = []
            if d['attack'] is None:
                d['attack'] = []
            logging.info(f"[{data_tag}] All Benign Flow: {len(d['benign'])}, All Attack Flow: {len(d['attack'])}")
            benign_flows = d['benign'][:benign_attack_parse_num[0]]
            attack_flows = d['attack'][:benign_attack_parse_num[1]]

    elif clean_traffic_tag is not None and clean_traffic_path is not None:
        logging.info(f'Read attack from {data_target}, benign from {clean_data_target}')
        with open(data_target, 'r') as f:
            da = json.load(f)
            attack_flows = da['attack'][:benign_attack_parse_num[1]]
        
        with open(clean_data_target, 'r') as f:
            db = json.load(f)
            benign_flows = db['benign'][:benign_attack_parse_num[0]]
            logging.info(f"[{data_tag}] All Benign Flow: {len(db['benign'])}, All Attack Flow: {len(da['attack'])}")

    else:
        logging.error('Invalid clean traffic tag and path.')
        assert False

    if shuffle_seed is not None:
        # random.seed(shuffle_seed)
        random.shuffle(attack_flows)
        random.shuffle(benign_flows)

    attack_flows = attack_flows[:benign_attack_samp_num[1]] if benign_attack_samp_num[1] != -1 else attack_flows
    benign_flows = benign_flows[:benign_attack_samp_num[0]] if benign_attack_samp_num[0] != -1 else benign_flows

    # For fixed length input
    def _normalize_flow(flow_collect):
        for i in range(len(flow_collect)):
            if len(flow_collect[i]) < segment_len:
                flow_collect[i] += [void_seq_feature_ls] * (segment_len - len(flow_collect[i]))
            else:
                flow_collect[i] = flow_collect[i][:segment_len]
        return flow_collect
    
    attack_seq_flows_features = _normalize_flow([x['seq_feature'] for x in attack_flows])
    benign_seq_flows_features = _normalize_flow([x['seq_feature'] for x in benign_flows])
    
    attack_non_seq_flows_features = [x['nonseq_feature'] for x in attack_flows]
    benign_non_seq_flows_features = [x['nonseq_feature'] for x in benign_flows]

    attack_interact_features = [x['interact_feature'] for x in attack_flows]
    benign_interact_features = [x['interact_feature'] for x in benign_flows]

    benign_flows_size = [x['len'] for x in benign_flows]
    attack_flows_size = [x['len'] for x in attack_flows]

    benign_flows_ids = [x['id'] for x in benign_flows]

    n_benign, n_attack = len(benign_flows_size), len(attack_flows_size)

    pre_train_line = math.floor(pre_train_ratio * n_benign)
    pre_train_data = list(zip(
        benign_seq_flows_features[:pre_train_line], 
        benign_non_seq_flows_features[:pre_train_line], 
        benign_interact_features[:pre_train_line]
    ))
    pre_train_id = benign_flows_ids[:pre_train_line]

    if train_number == -1:
        benign_train_line = pre_train_line + math.floor(train_ratio * n_benign)
        attack_train_line = math.floor(train_ratio * n_attack)
    else:
        benign_train_line = min(pre_train_line + train_number, n_benign)
        attack_train_line = min(train_number, n_attack)

    MIN_FLOW_SIZE = 10
    if attack_train_line <= MIN_FLOW_SIZE:
        attack_train_line = MIN_FLOW_SIZE

    train_data = list(zip(
        benign_seq_flows_features[pre_train_line:benign_train_line] + attack_seq_flows_features[:attack_train_line],
        benign_non_seq_flows_features[pre_train_line:benign_train_line] + attack_non_seq_flows_features[:attack_train_line], 
        benign_interact_features[pre_train_line:benign_train_line] + attack_interact_features[:attack_train_line]
    ))
    train_size = benign_flows_size[pre_train_line:benign_train_line] + attack_flows_size[:attack_train_line]
    train_benign_attack_div = benign_train_line - pre_train_line
    train_label = [0] * train_benign_attack_div + [1] * attack_train_line

    test_data = list(zip(
        benign_seq_flows_features[benign_train_line:] + attack_seq_flows_features[attack_train_line:], 
        benign_non_seq_flows_features[benign_train_line:] + attack_non_seq_flows_features[attack_train_line:],
        benign_interact_features[benign_train_line:] + attack_interact_features[attack_train_line:]
    ))
    test_size = benign_flows_size[benign_train_line:] + attack_flows_size[attack_train_line:]
    test_data_benign_attack_line = len(benign_flows_ids) - benign_train_line
    test_label = [0] * test_data_benign_attack_line + [1] * (len(attack_flows_size) - attack_train_line)

    # group pre-training data into contrastive batches.
    batch_num = len(pre_train_data) // contrast_batch_size

    src_ctr, dst_ctr = {}, {}
    for i, (_src, _dst) in enumerate(pre_train_id):
        if _src not in src_ctr: src_ctr[_src] = []
        src_ctr[_src].append(i)
        if _dst not in dst_ctr: dst_ctr[_dst] = []
        dst_ctr[_dst].append(i)
    

    pre_train_batch = []
    dst_key_ls = list(filter(lambda x: len(dst_ctr[x]) > (contrast_batch_size // 2), dst_ctr.keys()))
    # print('dst key num', len(dst_key_ls))
    for i in range(batch_num // 2):
        _target_key = dst_key_ls[i % len(dst_key_ls)]
        _target_same_ls = random.sample(dst_ctr[_target_key], contrast_batch_size // 2)
        _target_diff_ls = [] # samples randomly selected from dataset that not belong to the address of target
        while len(_target_diff_ls) < contrast_batch_size // 2:
            num = random.randint(0, len(pre_train_id) - 1)
            if num not in dst_ctr[_target_key]:
                _target_diff_ls.append(num)
        pre_train_batch.append([pre_train_data[x] for x in _target_same_ls + _target_diff_ls])
    # each contrastive batch contain contrstive_batch_size // 2 same address + contrstive_batch_size // 2 different address

    src_key_ls = list(filter(lambda x: len(src_ctr[x]) > (contrast_batch_size // 2), src_ctr.keys()))
    # print('src key num', len(src_key_ls))
    for i in range(batch_num // 2):
        _target_key = src_key_ls[i % len(src_key_ls)]
        _target_same_ls = random.sample(src_ctr[_target_key], contrast_batch_size // 2)
        _target_diff_ls = []
        while len(_target_diff_ls) < contrast_batch_size // 2:
            num = random.randint(0, len(pre_train_id) - 1)
            if num not in src_ctr[_target_key]:
                _target_diff_ls.append(num)
        pre_train_batch.append([pre_train_data[x] for x in _target_same_ls + _target_diff_ls])

    pre_train_batch_data_seq_ten = torch.FloatTensor(
        [[ pre_train_batch[b][s][0] for s in range(len(pre_train_batch[b]))] for b in range(len(pre_train_batch))]
    )
    pre_train_batch_data_non_seq_ten = torch.FloatTensor(
        [[ pre_train_batch[b][s][1] for s in range(len(pre_train_batch[b]))] for b in range(len(pre_train_batch))]
    )
    pre_train_batch_data_interact_ten = torch.FloatTensor(
        [[ pre_train_batch[b][s][2] for s in range(len(pre_train_batch[b]))] for b in range(len(pre_train_batch))]
    )

    train_data_seq_ten = torch.FloatTensor(
        [train_data[s][0] for s in range(len(train_data))]
    )
    train_data_non_seq_ten = torch.FloatTensor(
        [train_data[s][1] for s in range(len(train_data))]
    )
    train_data_interact_ten = torch.FloatTensor(
        [train_data[s][2] for s in range(len(train_data))]
    )
    train_size_ten = torch.IntTensor(train_size)
    train_label_ten = torch.IntTensor(train_label)
    
    if test_samp_ratio != 0:
        indices = random.sample(range(len(test_data)), math.floor(test_samp_ratio * len(test_data)))
        test_data = [test_data[i] for  i in indices]
        test_size = [test_size[i] for i in indices]
        test_label = [test_label[i] for i in indices]

    test_data_seq_ten = torch.FloatTensor(
        [test_data[s][0] for s in range(len(test_data))]
    )
    test_data_non_seq_ten = torch.FloatTensor(
        [test_data[s][1] for s in range(len(test_data))]
    )
    test_data_interact_ten = torch.FloatTensor(
        [test_data[s][2] for s in range(len(test_data))]
    )
    test_size_ten = torch.IntTensor(test_size)
    test_label_ten = torch.IntTensor(test_label)

    # logging.info(f'[{data_tag}] Pre-Train Batches: {len(pre_train_batch)}, Batchsize: {len(pre_train_batch[0])}')
    logging.info(f'[{data_tag}] Pre-Train Batches: {len(pre_train_batch)}')
    logging.info(f'[{data_tag}] Train Flows: {len(train_data)} [Attack: {sum(train_label)}], Test Flows: {len(test_data)} [Attack: {sum(test_label)} ]')

    return pre_train_batch_data_seq_ten, pre_train_batch_data_non_seq_ten, pre_train_batch_data_interact_ten, \
        train_data_seq_ten, train_data_non_seq_ten, train_data_interact_ten, \
        train_size_ten, train_label_ten, \
        test_data_seq_ten, test_data_non_seq_ten, test_data_interact_ten, \
        test_size_ten, test_label_ten

