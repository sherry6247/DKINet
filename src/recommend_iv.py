import pickle
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils import data
from loss import cross_entropy_loss
import os
import torch.nn.functional as F
import random
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_num_replace
from beam import Beam

import sys
sys.path.append("..")
# from models import Leap, CopyDrug_batch, CopyDrug_tranformer, CopyDrug_generate_prob, CopyDrug_diag_proc_encode
# from COGNet_model import COGNet
from util import llprint, sequence_metric

torch.manual_seed(1203)

def eval_recommend_batch(model, batch_data, device, TOKENS, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = batch_data
    # continue
    # 根据vocab对padding数值进行替换
    diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
    procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
    dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
    stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
    dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
    stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
    # medications = medications.to(device)
    medications = pad_num_replace(medications, -1, MED_PAD_TOKEN).to(device)
    m_mask_matrix = m_mask_matrix.to(device)
    d_mask_matrix = d_mask_matrix.to(device)
    p_mask_matrix = p_mask_matrix.to(device)
    dec_disease_mask = dec_disease_mask.to(device)
    stay_disease_mask = stay_disease_mask.to(device)
    dec_proc_mask = dec_proc_mask.to(device)
    stay_proc_mask = stay_proc_mask.to(device)

    output_logits = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
    output_logits = output_logits[0]
    return output_logits

def test_recommend_batch(model, batch_data, device, TOKENS, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = batch_data
    # continue
    # 根据vocab对padding数值进行替换
    diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
    procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
    dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
    stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
    dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
    stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
    # medications = medications.to(device)
    medications = pad_num_replace(medications, -1, MED_PAD_TOKEN).to(device)
    m_mask_matrix = m_mask_matrix.to(device)
    d_mask_matrix = d_mask_matrix.to(device)
    p_mask_matrix = p_mask_matrix.to(device)
    dec_disease_mask = dec_disease_mask.to(device)
    stay_disease_mask = stay_disease_mask.to(device)
    dec_proc_mask = dec_proc_mask.to(device)
    stay_proc_mask = stay_proc_mask.to(device)

    output_logits = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
    output_logits = output_logits[0]
    return output_logits

# evaluate
def eval(model, eval_dataloader, voc_size, epoch, device, TOKENS, args):
    model.eval()
    # torch.manual_seed(args.seed)
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    # fw = open("prediction_results.txt", "w")

    for idx, data in enumerate(eval_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = data
        visit_cnt += seq_length.sum().item()
        output_logits = eval_recommend_batch(model, data, device, TOKENS, args) #$ [bsz, visit_len, code_vocab_size]
        # output_logits = torch.sigmoid(output_logits)
        
        labels = target_list
        predictions = torch.sigmoid(output_logits).cpu().detach().numpy()
        y_gt = []       # groud truth 表示正确的label   0-1序列
        y_pred = []     # 预测的结果    0-1序列
        y_pred_prob = []    # 预测的每一个药物的平均概率，非0-1序列
        y_pred_label = []   # 预测的结果，非0-1序列
        # 针对每一个admission的预测结果
        for label, prediction in zip(labels[0], predictions[0]):
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1    # 01序列，表示正确的label
            y_gt.append(y_gt_tmp)

            out_list = np.where(prediction>args.threshold)[0]
            y_pred_label.append(out_list)
            y_pred_prob.append(prediction)

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(out_list)

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), y_pred_label)
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\reval step: {} / {}'.format(idx, len(eval_dataloader)))


    llprint('\nJaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


# test 
def test_addPath(model, resume_path, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, epoch, device, TOKENS, args):
    model.eval()
    # torch.manual_seed(args.seed)
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    record_ja, record_prauc, record_p, record_r, record_f1 = [defaultdict(list) for _ in range(5)]
    med_cnt_list = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    all_pred_list = {}
    all_label_list = {}

    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]

    cur_embs = {}
    ori_embs = {}

    for idx, data in enumerate(test_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = data
        visit_cnt += seq_length.sum().item()

        output_logits = test_recommend_batch(model, data, device, TOKENS, args)
        labels = target_list
        predictions = torch.sigmoid(output_logits).cpu().detach().numpy()

        y_gt = []       
        y_pred = []    
        y_pred_label = [] 
        y_pred_prob = [] 

        label_hisory = []
        label_hisory_list = []
        pred_list = []
        jaccard_list = []
        def cal_jaccard(set1, set2):
            if not set1 or not set2:
                return 0
            set1 = set(set1)
            set2 = set(set2)
            a, b = len(set1 & set2), len(set1 | set2)
            return a/b
        def cal_overlap_num(set1, set2):
            count = 0
            for d in set1:
                if d in set2:
                    count += 1
            return count

        # 针对每一个admission的预测结果
        for label, prob_list in zip(labels[0], predictions[0]):
            label_hisory += label#.tolist()  ### case study

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1    # 01序列，表示正确的label
            y_gt.append(y_gt_tmp)

            out_list = np.where(prob_list>args.threshold)[0]
            pred_list.append(out_list)
            y_pred_prob.append(prob_list)
            y_pred_label.append(out_list)

            ## case study
            if label_hisory:
                jaccard_list.append(cal_jaccard(list(out_list), label_hisory))
            # pred_list.append(out_list)
            label_hisory_list.append(label) #.tolist()

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(out_list)
            med_cnt_list.append(len(out_list))


        smm_record.append(y_pred_label)
        for i in range(min(len(labels[0]), 5)):
            single_ja, single_auc, single_p, single_r, single_f1 = sequence_metric(np.array([y_gt[i]]), np.array([y_pred[i]]), np.array([y_pred_prob[i]]),np.array([y_pred_label[i]]))
            ja_by_visit[i].append(single_ja)
            auc_by_visit[i].append(single_auc)
            pre_by_visit[i].append(single_p)
            recall_by_visit[i].append(single_r)
            f1_by_visit[i].append(single_f1)
            smm_record_by_visit[i].append(y_pred_label[i:i+1])

        # 存储所有预测结果
        # all_pred_list.append(pred_list)
        # all_label_list.append(labels)
        all_pred_list['p_{}'.format(idx)] = pred_list
        all_label_list['p_{}'.format(idx)] = labels[0]
        # adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                # sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), y_pred_label)
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(idx, len(test_dataloader)))

        """将每个patient的各个指标都记录下来"""
        record_ja['v_{}'.format(data[0].shape[1])].append(adm_ja)
        record_prauc['v_{}'.format(data[0].shape[1])].append(adm_prauc)
        record_p['v_{}'.format(data[0].shape[1])].append(adm_avg_p)
        record_r['v_{}'.format(data[0].shape[1])].append(adm_avg_r)
        record_f1['v_{}'.format(data[0].shape[1])].append(adm_avg_f1)

        # 统计不同visit的指标
        if idx%100==0:
            print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
            print('count:', [len(buf) for buf in ja_by_visit])
            print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
            print('auc:', [np.mean(buf) for buf in auc_by_visit])
            print('precision:', [np.mean(buf) for buf in pre_by_visit])
            print('recall:', [np.mean(buf) for buf in recall_by_visit])
            print('f1:', [np.mean(buf) for buf in f1_by_visit])

    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('count:', [len(buf) for buf in ja_by_visit])
    print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
    print('auc:', [np.mean(buf) for buf in auc_by_visit])
    print('precision:', [np.mean(buf) for buf in pre_by_visit])
    print('recall:', [np.mean(buf) for buf in recall_by_visit])
    print('f1:', [np.mean(buf) for buf in f1_by_visit])

    return smm_record, ja, prauc, avg_p, avg_r, avg_f1, med_cnt_list

def test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, epoch, device, TOKENS, args):
    model.eval()
    # torch.manual_seed(args.seed)
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    record_ja, record_prauc, record_p, record_r, record_f1 = [defaultdict(list) for _ in range(5)]
    med_cnt_list = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    all_pred_list = {}
    all_label_list = {}

    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]

    cur_embs = {}
    ori_embs = {}

    for idx, data in enumerate(test_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = data
        visit_cnt += seq_length.sum().item()

        output_logits = test_recommend_batch(model, data, device, TOKENS, args)
        labels = target_list
        predictions = torch.sigmoid(output_logits).cpu().detach().numpy()

        y_gt = []       
        y_pred = []    
        y_pred_label = [] 
        y_pred_prob = [] 

        label_hisory = []
        label_hisory_list = []
        pred_list = []
        jaccard_list = []
        def cal_jaccard(set1, set2):
            if not set1 or not set2:
                return 0
            set1 = set(set1)
            set2 = set(set2)
            a, b = len(set1 & set2), len(set1 | set2)
            return a/b
        def cal_overlap_num(set1, set2):
            count = 0
            for d in set1:
                if d in set2:
                    count += 1
            return count

        # 针对每一个admission的预测结果
        for label, prob_list in zip(labels[0], predictions[0]):
            label_hisory += label#.tolist()  ### case study

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1    # 01序列，表示正确的label
            y_gt.append(y_gt_tmp)

            out_list = np.where(prob_list>=args.threshold)[0]
            pred_list.append(out_list)
            y_pred_prob.append(prob_list)
            y_pred_label.append(out_list)

            ## case study
            if label_hisory:
                jaccard_list.append(cal_jaccard(list(out_list), label_hisory))
            # pred_list.append(out_list)
            label_hisory_list.append(label) #.tolist()

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(out_list)
            med_cnt_list.append(len(out_list))


        smm_record.append(y_pred_label)
        for i in range(min(len(labels[0]), 5)):
            single_ja, single_auc, single_p, single_r, single_f1 = sequence_metric(np.array([y_gt[i]]), np.array([y_pred[i]]), np.array([y_pred_prob[i]]),np.array([y_pred_label[i]]))
            ja_by_visit[i].append(single_ja)
            auc_by_visit[i].append(single_auc)
            pre_by_visit[i].append(single_p)
            recall_by_visit[i].append(single_r)
            f1_by_visit[i].append(single_f1)
            smm_record_by_visit[i].append(y_pred_label[i:i+1])

        all_pred_list['p_{}'.format(idx)] = pred_list
        all_label_list['p_{}'.format(idx)] = labels[0]
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), y_pred_label)
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(idx, len(test_dataloader)))

        """将每个patient的各个指标都记录下来"""
        record_ja['v_{}'.format(data[0].shape[1])].append(adm_ja)
        record_prauc['v_{}'.format(data[0].shape[1])].append(adm_prauc)
        record_p['v_{}'.format(data[0].shape[1])].append(adm_avg_p)
        record_r['v_{}'.format(data[0].shape[1])].append(adm_avg_r)
        record_f1['v_{}'.format(data[0].shape[1])].append(adm_avg_f1)

        # 统计不同visit的指标
        if idx%100==0:
            print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
            print('count:', [len(buf) for buf in ja_by_visit])
            print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
            print('auc:', [np.mean(buf) for buf in auc_by_visit])
            print('precision:', [np.mean(buf) for buf in pre_by_visit])
            print('recall:', [np.mean(buf) for buf in recall_by_visit])
            print('f1:', [np.mean(buf) for buf in f1_by_visit])

    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('count:', [len(buf) for buf in ja_by_visit])
    print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
    print('auc:', [np.mean(buf) for buf in auc_by_visit])
    print('precision:', [np.mean(buf) for buf in pre_by_visit])
    print('recall:', [np.mean(buf) for buf in recall_by_visit])
    print('f1:', [np.mean(buf) for buf in f1_by_visit])

    return smm_record, ja, prauc, avg_p, avg_r, avg_f1, med_cnt_list
