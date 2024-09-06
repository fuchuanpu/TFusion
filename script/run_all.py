#!/usr/bin/env python3

import os
import time
import re
import sys
import json
import subprocess
import numpy as np

# Add the parent directory to sys.pathfrom pathlib import Path
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from common import *

BASE_DIR = './config/train-test/'
if not os.path.exists(BASE_DIR):
    logging.error('Config directory not found, please execute at root dir.')
    exit(-1)

exe_targets = (
    'application', 
    'malware', 
    'misc', 
    'amplification', 
    'scan', 
    'dos', 
    'web' 
)

exe_cfg_files = []
for _t in exe_targets:
    _target = f'{BASE_DIR}{_t}.json'
    if not os.path.exists(_target):
        logging.error(f'Config {_target} not found.')
        exit(-1)
    else:
        exe_cfg_files.append(_target)

RES_SAVE_DIR = './cache/'
if not os.path.exists(RES_SAVE_DIR):
    os.makedirs(RES_SAVE_DIR)

if __name__ == '__main__':
    # execute the dynamic few-shot learning
    execution_result = {}
    for _idx, _cfg in enumerate(exe_cfg_files):
        logging.info(f'Executing [{exe_targets[_idx]}] datasets ...')
        execution_result[exe_targets[_idx]] = {}
        result = subprocess.run(['./main.py', '-c', _cfg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 'result' contains the information about the process after it has finished
        if result.returncode == 0:
            logging.info(f'{_cfg} finished successfully.')
            # print('Output:', result.stdout)
            pattern = r'\[(.+)\]  AUC ([0-9]+\.[0-9]+)'
            matches = re.findall(pattern, result.stdout)
            for _tag, _acc in matches:
                execution_result[exe_targets[_idx]][_tag] = float(_acc)
            logging.info(f'Average AUC: {np.array(list(execution_result[exe_targets[_idx]].values())).mean()}')
        else:
            logging.error('Error:', result.stderr)
            exit(-1)
    
    # Save all results:
    with open(f'{RES_SAVE_DIR}/few_shot_number.json', 'w') as f: 
        f.write(json.dumps(execution_result, indent=4))
        logging.info(f'All results saved to {RES_SAVE_DIR}/few_shot_number.json')
