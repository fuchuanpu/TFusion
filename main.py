#!/usr/bin/env python3

from typing import Dict

import torch

import json
import os
import multiprocessing
import random
import argparse

from common import *
from data_contrast import *
from train_contrast import *

seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train_task(command,
               detection_model, 
               data_tag,
               data_path, 
               log_path, 
               fig_path, 
               clean_traffic_path,
               clean_traffic_tag,
               traffic_model_path, 
               traffic_model_tag,
               detection_model_path, 
               detection_model_tag,  
               gpu_id, 
               waterline, 
               data_set_params,
               pre_train_params):
    
    dataset = load_data_contrast_remake(data_tag, 
                                        data_path, 
                                        clean_traffic_tag, 
                                        clean_traffic_path, 
                                        shuffle_seed=seed, 
                                        **data_set_params)

    if command == 'pre-train':
        pre_train_contrast(data_tag,
                           detection_model, 
                           log_path, 
                           fig_path, 
                           traffic_model_path, 
                           *dataset, 
                           gpu_id, 
                           waterline, 
                           **pre_train_params)
    elif command == 'train':
        train_detection(data_tag,
                        detection_model, 
                        log_path, 
                        fig_path, 
                        traffic_model_path, 
                        traffic_model_tag,
                        detection_model_path,
                        *dataset[3:], 
                        gpu_id, 
                        waterline)
    elif command in ('network-transfer-test', 'task-transfer-test'):
        if command == 'network-transfer-test' and detection_model_tag != traffic_model_tag:
            logging.error('Network transfer test should use the same model.')
            exit(-1)
        test_detection(data_tag,
                       detection_model, 
                       log_path, 
                       fig_path, 
                       traffic_model_path, 
                       traffic_model_tag,
                       detection_model_path,
                       detection_model_tag,
                       *dataset[(3+5):], 
                       gpu_id, 
                       waterline)
    else:
        assert False



def validate_json_config(jin: Dict) -> bool:
    common_key_list = [
        'command',
        'data_path', 
        'log_path', 
        'fig_path', 
        'gpu_enable', 
        'data_construct_param', 
        'train_param',
        'detection_model'
    ]
    model_key_list = [
        'kmeans',
        'iforest',
        'ocsvm',
        'rforest',
        'knn',
        'logistic',
        'bayes',
        'placeholder'
    ]
    
    def f_check_ley(k, slience=False):
        if isinstance(k, str):
            k = [k]
        elif isinstance(k, list):
            pass
        else:
            return False
        for l in k:
            if l not in jin:
                if not slience:
                    logging.error(f'Key {l} is missed in configuration.')
                return False
        return True
        
    # Check of command, ensuring necceary key is provided.
    if not f_check_ley(common_key_list):
        logging.error('Basic configuration is missed.')
        return False

    if jin['command'] == 'pre-train':
        if not f_check_ley('traffic_model_path'):
            return False
    elif jin['command'] == 'train':
        if not f_check_ley(['traffic_model_tag', 'traffic_model_path']):
            logging.error(f'Pre-trained model should be designtaed for training.')
            return False
        if not f_check_ley('detection_model_path'):
            logging.error(f'Key detection_traffic_model_path is missed in configuration for training.')
            return False
    elif jin['command'] == 'network-transfer-test' or jin['command'] == 'task-transfer-test':
        if not f_check_ley(['traffic_model_tag', 'traffic_model_path']):
            logging.error(f'Missed pre-trained model for testing.')
            return False
        if not f_check_ley(['detection_model_tag', 'detection_model_path']):
            logging.error(f'Missed Detection model for testing.')
            return False
    else:
        logging.error(f'Command {jin["command"]} is not supported.')
        return False
    
    # Check of Models
    if not jin['detection_model'] in model_key_list:
        logging.error(f'Detection model {jin["detection_model"]} is not supported.')
        return False
    
    # Check of Path
    try:
        if not os.path.exists(jin['log_path']):
            os.makedirs(jin['log_path'])
        if not os.path.exists(jin['fig_path']):
            os.makedirs(jin['fig_path'])
        if not os.path.exists(jin['traffic_model_path']):
            os.makedirs(jin['traffic_model_path'])
        if f_check_ley('detection_model_path', slience=True) and not os.path.exists(jin['detection_model_path']):
            os.makedirs(jin['detection_model_path'])

    except Exception as e:
        logging.error(f'Exceprtion in create folder' + e)
        return False
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TFusion: learning generic knowledge from traffic.')
    parser.add_argument('-c', '--config', type=str, default='./config.json', help='Configuration file.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        jin = json.load(f)
    if not validate_json_config(jin):
        exit(-1)
    
    pve = []
    for i, tag in enumerate(jin['data_tag'].keys()):
        _gpu_id = jin['gpu_enable'][i % len(jin['gpu_enable'])] if len(jin['gpu_enable']) != 0 else -1
        pve.append(
            multiprocessing.Process(
                target=train_task, 
                args=(
                    jin['command'],
                    jin['detection_model'],
                    tag, 
                    jin['data_path'],
                    jin['log_path'],
                    jin['fig_path'],
                    jin.get('clean_traffic_path', None),
                    jin.get('clean_traffic_tag', None),
                    jin.get('traffic_model_path', None),
                    jin.get('traffic_model_tag', None),
                    jin.get('detection_model_path', None),
                    jin.get('detection_model_tag', None),
                    _gpu_id,
                    jin['data_tag'][tag],
                    jin['data_construct_param'],
                    jin['train_param']
                )
            )
        )

    [p.start() for p in pve]
    [p.join() for p in pve]
