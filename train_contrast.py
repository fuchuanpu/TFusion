#!/usr/bin/env python3

from typing import List
import torch
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score

from common import *
from model import TFusion
from model.detection import DetectorFactory

n_hidden = 100
n_project = 128
packet_label = False
auc_match = False

'''
    Train multi-modal traffic model using contrastive learning.
    And train a detection model upon the traffic model.
    Finally, test the detection model on the datasets.
'''
@time_log
def pre_train_contrast(data_tag:str, 
                       detection_model:str,
                       log_path:str, 
                       fig_path:str,
                       traffic_model_path:str,
                       pre_train_batch_seq:torch.FloatTensor, 
                       pre_train_batch_non_seq:torch.FloatTensor, 
                       pre_train_batch_interact:torch.FloatTensor, 
                       train_data_seq:torch.FloatTensor, 
                       train_data_non_seq:torch.FloatTensor, 
                       train_data_interact:torch.FloatTensor, 
                       train_size:torch.IntTensor,
                       train_label:torch.IntTensor,
                       test_data_seq:torch.FloatTensor,
                       test_data_non_seq:torch.FloatTensor,
                       test_data_interact:torch.FloatTensor,
                       test_size:torch.IntTensor,
                       test_label:torch.IntTensor,
                       gpu_id:int, 
                       waterline:float, 
                       lr=0.001, 
                       wd=1e-4, 
                       num_epoch=15):
    
    logging.info(f'[{data_tag}] is started.')
    fout = open(f'{log_path}/{data_tag}.log', 'w', buffering=1)
    model_name = detection_model

    train_on_gpu = torch.cuda.is_available()
    print(f'[{data_tag}] Use GPU:{gpu_id}.' if train_on_gpu else f'[{data_tag}] Use CPU.', 
            file=fout, flush=True)
    device = torch.device(f'cuda:{gpu_id}' if train_on_gpu else 'cpu')

    # the basic encoder for traffic
    model = TFusion().to(device)
    # contrastive header, which is similar to Chen (the last layer is not activated)
    model_project = nn.Sequential(
        nn.Linear(n_hidden, n_project),
        nn.ReLU(),
        nn.Linear(n_project, n_project),
    ).to(device)

    mode_stage1 = nn.Sequential(
        model, model_project
    )
    opt_contrast = torch.optim.Adam(mode_stage1.parameters(), lr=lr, weight_decay=wd)

    pre_train_batch_size = pre_train_batch_seq.size(1)
    assert pre_train_batch_size == pre_train_batch_non_seq.size(1)
    assert pre_train_batch_size == pre_train_batch_interact.size(1)

    # A renovated contrastive loss function
    def contrastive_loss(x_predict):
        same_line = pre_train_batch_size // 2
        
        sim = lambda x, y: x.dot(y) / (x.norm() * y.norm())
        sim_pair = lambda i, j: torch.exp(sim(x_predict[i], x_predict[j]))
        sim_loss = lambda i: - torch.log(sim_pair(i, i + 1) / (sim_pair(i, i + 1) + sim_pair(i, i + same_line)))

        loss_val = torch.zeros(1).to(device)
        for i in range(same_line - 1):
            loss_val += sim_loss(i) 
        loss_val /= same_line
        return loss_val

    loss_contrast = contrastive_loss
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]), file=fout, flush=True)

    for e in range(num_epoch):
    ##### pre-train
        train_loss = 0.0

        print(f'[{data_tag}] ', "Start Train")
        mode_stage1.train()
        num_train = 0
        num_pretrain_batch = pre_train_batch_seq.size(0)

        for i in range(0, num_pretrain_batch):
            x_seq = pre_train_batch_seq[i].to(device)
            x_non_seq = pre_train_batch_non_seq[i].to(device)
            x_interact = pre_train_batch_interact[i].to(device)
            num_train += 1
            opt_contrast.zero_grad()
            x_map = mode_stage1((x_seq, x_non_seq, x_interact))
            lossT = loss_contrast(x_map)
            train_loss += lossT.item()
            lossT.backward()
            opt_contrast.step()

        mode_stage1.eval()
        print(f'[{data_tag}] ', f'Pre-Training Loss: {train_loss / num_train:.4f}')
        print(f'[{data_tag}] ', f'Pre-Training Loss: {train_loss / num_train:.4f}', file=fout,flush=True)

        torch.save(mode_stage1.state_dict(), f"{traffic_model_path}/{data_tag}_model.pt")
        logging.info(f'Save pre-trained model to {traffic_model_path}/{data_tag}_model.pt')


'''
    Train detection model only.
    Traffic model is loaded from file system.
    Finally, the model is tested on where it is trained.
'''
@time_log
def train_detection(data_tag:str, 
                    detection_model:str,
                    log_path:str, 
                    fig_path:str,
                    traffic_model_path:str,
                    traffic_model_tag:str,
                    detection_model_path:str,
                    train_data_seq:torch.FloatTensor, 
                    train_data_non_seq:torch.FloatTensor, 
                    train_data_interact:torch.FloatTensor, 
                    train_size:torch.IntTensor,
                    train_label:torch.IntTensor,
                    test_data_seq:torch.FloatTensor,
                    test_data_non_seq:torch.FloatTensor,
                    test_data_interact:torch.FloatTensor,
                    test_size:torch.IntTensor,
                    test_label:torch.IntTensor,
                    gpu_id:int, 
                    waterline:float):
    
    logging.info(f'[{data_tag}] is started (w/o pre-train).')
    fout = open(f'{log_path}/{data_tag}.log', 'w', buffering=1)
    model_name = detection_model

    train_on_gpu = torch.cuda.is_available()
    print(f'[{data_tag}] Use GPU:{gpu_id}.' if train_on_gpu else f'[{data_tag}] Use CPU.', 
            file=fout, flush=True)
    device = torch.device(f'cuda:{gpu_id}' if train_on_gpu else 'cpu')

    model = TFusion().to(device)
    model_project = nn.Sequential(
        nn.Linear(n_hidden, n_project),
        nn.ReLU(),
        nn.Linear(n_project, n_project),
    ).to(device)

    mode_stage1 = nn.Sequential(
        model, model_project
    )

    assert os.path.exists(f'{traffic_model_path}/{traffic_model_tag}_model.pt')
    mode_stage1.load_state_dict(torch.load(f'{traffic_model_path}/{traffic_model_tag}_model.pt'))
    mode_stage1.eval()

    ##### detection model train
    infer_batch_size = 300
    num_train_sample = train_data_seq.size(0)
    assert num_train_sample == train_data_non_seq.size(0)
    assert num_train_sample == train_size.size(0)

    print(f'[{data_tag}] ', "Start training detetcion model")
    train_input = np.zeros((num_train_sample, n_hidden))
    for i in range(0, num_train_sample, infer_batch_size):
        x_seq = train_data_seq[i:min(i + infer_batch_size, num_train_sample)].to(device)
        x_non_seq = train_data_non_seq[i:min(i + infer_batch_size, num_train_sample)].to(device)
        x_interact = train_data_interact[i:min(i + infer_batch_size, num_train_sample)].to(device)
        din = model((x_seq, x_non_seq, x_interact))
        train_input[i:min(i + infer_batch_size, num_train_sample)] = din.cpu().detach().numpy()

    detection_model = DetectorFactory.create(model_name)
    detection_model.train(train_input, train_label)
    detection_model.save(f'{detection_model_path}/{data_tag}')

    loss_res = []
    label_res = []

    #### detection model test
    num_test_sample = test_data_seq.size(0)
    assert num_test_sample == test_data_non_seq.size(0)

    test_input = np.zeros((num_test_sample, n_hidden))
    for i in range(0, num_test_sample, infer_batch_size):
        x_seq = test_data_seq[i:min(i + infer_batch_size, num_test_sample)].to(device)
        x_non_seq = test_data_non_seq[i:min(i + infer_batch_size, num_test_sample)].to(device)
        x_interact = test_data_interact[i:min(i + infer_batch_size, num_test_sample)].to(device)
        din = model((x_seq, x_non_seq, x_interact))
        test_input[i:min(i + infer_batch_size, num_test_sample)] = din.cpu().detach().numpy()
    
    test_loss_vec = detection_model.test(test_input)
    for (label, sz, ls) in zip(test_label, test_size, test_loss_vec):            
        if packet_label: 
            loss_res.extend([ls] * sz.item())
            label_res.extend([label.item()] * sz.item())
        else: # flow label
            loss_res.append(ls)
            label_res.append(label.item())

    judge = [1 if sc > waterline else 0 for sc in loss_res]
    fpr, tpr, _ = roc_curve(label_res, judge if auc_match else loss_res)
    roc_auc = auc(fpr, tpr)
    print(f'[{data_tag}] ', "AUC", roc_auc)
    f1 = f1_score(label_res, judge, average='macro')
    per = precision_score(label_res, judge, average='macro')
    rec = recall_score(label_res, judge, average='macro')

    # print(f'Epoch: {e:2d}, train loss: {train_loss:7.4f}, test loss: {test_loss:7.4f},'
    #       f'AUC: {roc_auc:7.4f}, F1: {f1:7.4f}, Percision: {per:7.4f}, Recall: {rec:7.4f}, ',
    #       file=fout,flush=True)

    print(f'AUC: {roc_auc:7.4f}, F1: {f1:7.4f}, Percision: {per:7.4f}, Recall: {rec:7.4f}, ',
            file=fout,flush=True)

    fig = plt.figure(figsize=(10, 10 * 0.618), constrained_layout=True)
    ax = fig.subplots(1, 1)
    benign_score = [x[1] for x in filter(lambda x: not x[0], list(zip(label_res, loss_res)))]
    attack_score = [x[1] for x in filter(lambda x: x[0], list(zip(label_res, loss_res)))]
    ax.hist(benign_score, 1000, density=True, histtype='step', cumulative=True, label='Benign', color='royalblue')
    ax.hist(attack_score, 1000, density=True, histtype='step', cumulative=True, label='Attack', color='firebrick')
    ax.set_xlabel('Contrast Loss')
    ax.set_ylabel('CDF')
    ax.set_title(f'{data_tag}')
    ax.vlines(waterline, 0, 1.05, lw=1, color='grey')
    ax.legend(loc='right')

    save_addr = f'{fig_path}/{data_tag}_result.png'
    fig.savefig(save_addr, dpi=600, format='png')
    plt.cla()



@time_log
def test_detection(data_tag:str, 
                   detection_model:str,
                   log_path:str, 
                   fig_path:str,
                   traffic_model_path:str,
                   traffic_model_tag:str,
                   detection_model_path:str,
                   detection_model_tag:str,
                   test_data_seq:torch.FloatTensor,
                   test_data_non_seq:torch.FloatTensor,
                   test_data_interact:torch.FloatTensor,
                   test_size:torch.IntTensor,
                   test_label:torch.IntTensor,
                   gpu_id:int,
                   waterline:float):
    
    logging.info(f'[{data_tag}] is started (w/o pre-train).')
    fout = open(f'{log_path}/{data_tag}.log', 'w', buffering=1)
    model_name = detection_model

    train_on_gpu = torch.cuda.is_available()
    print(f'[{data_tag}] Use GPU:{gpu_id}.' if train_on_gpu else f'[{data_tag}] Use CPU.', 
            file=fout, flush=True)
    device = torch.device(f'cuda:{gpu_id}' if train_on_gpu else 'cpu')

    model = TFusion().to(device)
    model_project = nn.Sequential(
        nn.Linear(n_hidden, n_project),
        nn.ReLU(),
        nn.Linear(n_project, n_project),
    ).to(device)

    mode_stage1 = nn.Sequential(
        model, model_project
    )

    assert os.path.exists(f'{traffic_model_path}/{traffic_model_tag}_model.pt')
    mode_stage1.load_state_dict(torch.load(f'{traffic_model_path}/{traffic_model_tag}_model.pt'))
    mode_stage1.eval()

    ##### detection model train
    infer_batch_size = 300
    print(f'[{data_tag}] ', "Load detetcion model.")

    detection_model = DetectorFactory.create(model_name)
    detection_model.load(f'{detection_model_path}/{detection_model_tag}')

    loss_res = []
    label_res = []

    #### detection model test
    num_test_sample = test_data_seq.size(0)
    assert num_test_sample == test_data_non_seq.size(0)

    test_input = np.zeros((num_test_sample, n_hidden))
    for i in range(0, num_test_sample, infer_batch_size):
        x_seq = test_data_seq[i:min(i + infer_batch_size, num_test_sample)].to(device)
        x_non_seq = test_data_non_seq[i:min(i + infer_batch_size, num_test_sample)].to(device)
        x_interact = test_data_interact[i:min(i + infer_batch_size, num_test_sample)].to(device)
        din = model((x_seq, x_non_seq, x_interact))
        test_input[i:min(i + infer_batch_size, num_test_sample)] = din.cpu().detach().numpy()
    
    test_loss_vec = detection_model.test(test_input)
    for (label, sz, ls) in zip(test_label, test_size, test_loss_vec):            
        if packet_label: 
            loss_res.extend([ls] * sz.item())
            label_res.extend([label.item()] * sz.item())
        else: # flow label
            loss_res.append(ls)
            label_res.append(label.item())

    judge = [1 if sc > waterline else 0 for sc in loss_res]
    fpr, tpr, _ = roc_curve(label_res, judge if auc_match else loss_res)
    roc_auc = auc(fpr, tpr)
    print(f'[{data_tag}] ', "AUC", roc_auc)
    f1 = f1_score(label_res, judge, average='macro')
    per = precision_score(label_res, judge, average='macro')
    rec = recall_score(label_res, judge, average='macro')
    acc = sum([1 if x == y else 0 for x, y in zip(label_res, judge)]) / len(label_res)
    auprc = per * rec + (((1 - per) * rec) / 2) + (((1 - rec) * per) / 2)
    num_fp = sum([1 if x == 0 and y == 1 else 0 for x, y in zip(label_res, judge)])

    print(f'AUC: {roc_auc:7.4f}, F1: {f1:7.4f}, Percision: {per:7.4f}, Recall: {rec:7.4f}, Acc: {acc:7.4f}, AUPRC: {auprc:7.4f}, fp: {num_fp}',
            file=fout,flush=True)
    print(f'AUC: {roc_auc:7.4f}, F1: {f1:7.4f}, Percision: {per:7.4f}, Recall: {rec:7.4f}, Acc: {acc:7.4f}, AUPRC: {auprc:7.4f}, fp: {num_fp}.')
    
    fig = plt.figure(figsize=(10, 10 * 0.618), constrained_layout=True)
    ax = fig.subplots(1, 1)
    benign_score = [x[1] for x in filter(lambda x: not x[0], list(zip(label_res, loss_res)))]
    attack_score = [x[1] for x in filter(lambda x: x[0], list(zip(label_res, loss_res)))]
    ax.hist(benign_score, 1000, density=True, histtype='step', cumulative=True, label='Benign', color='royalblue')
    ax.hist(attack_score, 1000, density=True, histtype='step', cumulative=True, label='Attack', color='firebrick')
    ax.set_xlabel('Contrast Loss')
    ax.set_ylabel('CDF')
    ax.set_title(f'{data_tag}')
    ax.vlines(waterline, 0, 1.05, lw=1, color='grey')
    ax.legend(loc='right')

    save_addr = f'{fig_path}/{data_tag}_result.png'
    fig.savefig(save_addr, dpi=600, format='png')
    plt.cla()

