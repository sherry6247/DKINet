'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-11-03 21:52:30
LastEditors: Please set LastEditors
LastEditTime: 2023-05-03 18:56:39
FilePath: /model/lsc_code/DrugRe_UseUMLS/MDKINet_NIPS22Data_204/KG_models.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.modules.linear import Linear
# from data.processing import process_visit_lg2

from layers import SelfAttend
from layers import GraphConvolution

from modules import *
# from block_recurrent_transformer.transformer import BlockRecurrentAttention
from torch_geometric.nn import GCNConv
from GraphConv_module import GraphConv
from CLUB_MI import CLUB, CLUBSample, My_CLUBSample, MyCLUB

# ***********
class IntentKG(nn.Module):
    """
    KG 聚合后，code采用intent方式搜集KG信息；根据患者历史visit进行intent搜集KG信息。同时，保证搜集到的信息越多越好
    在最后visit level利用知识聚合
    """
    def __init__(self, voc_size, ehr_adj, 
                n_nodes, n_intents, n_relations, n_entities, kg_graph, kg_adj_mat, sim_regularity=1e-4, codeMI_regularity=1e-2, emb_dim=128, 
                context_hops = 3, ind='mi', node_dropout=True, node_dropout_rate=0.3, mess_dropout=True, mess_dropout_rate=0.3,
                device=torch.device('cpu:0'), num_inds=32, dim_hidden=128, num_heads=2, ln=False, isab_num=2, kgloss_alpha=0.001): 
        super(IntentKG, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.nhead = num_heads
        self.SOS_TOKEN = voc_size[2]        # start of sentence
        self.END_TOKEN = voc_size[2]+1      # end   新增的两个编码，两者均是针对于药物的embedding
        self.MED_PAD_TOKEN = voc_size[2]+2      # 用于embedding矩阵中的padding（全为0）
        self.DIAG_PAD_TOKEN = voc_size[0]+2
        self.PROC_PAD_TOKEN = voc_size[1]+2

        self.dim_hidden = dim_hidden

        ## KG related
        self.n_nodes = n_nodes
        self.kg_leaf = voc_size[0] + voc_size[1] + voc_size[2]
        self.n_intents = n_intents
        self.n_relations = n_relations
        self.n_entities = n_entities
        self.ind = ind
        self.context_hops = context_hops
        self.node_dropout = node_dropout
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout = mess_dropout
        self.mess_dropout_rate = mess_dropout_rate
        self.kg_graph = kg_graph
        self.kg_adj_mat = kg_adj_mat
        self.kg_edge_index, self.kg_edge_type = self._get_edges(kg_graph)  
        self.sim_regularity = sim_regularity      
        self.codeMI_regularity = codeMI_regularity

        self.isab_num = isab_num
        self.num_inds = num_inds

        # num_outputs = k seed vector
        num_outputs = voc_size[2]
        dim_output = 1
        # num_outputs = 1
        # dim_output = voc_size[2]

        # dig_num * emb_dim
        self.diag_embedding = nn.Sequential(
            nn.Embedding(voc_size[0]+3, emb_dim, self.DIAG_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # proc_num * emb_dim
        self.proc_embedding = nn.Sequential(
            nn.Embedding(voc_size[1]+3, emb_dim, self.PROC_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            # 添加padding_idx，表示取0向量
            nn.Embedding(voc_size[2]+3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        self.ori_agg = nn.Linear(3*emb_dim, emb_dim)
        self.curt_agg = nn.Sequential(
            nn.Linear(3*emb_dim, emb_dim),
            nn.Tanh()
        )

        # 历史code 采用intent聚合KG
        initializer = nn.init.xavier_uniform_
        hist_weight = initializer(torch.empty(n_relations - 1, emb_dim))  # not include interact
        self.hist_weight = nn.Parameter(hist_weight)  # [n_relations - 1, in_channel]
        hist_disen_weight_att = initializer(torch.empty(n_intents, n_relations - 1))
        self.hist_disen_weight_att = nn.Parameter(hist_disen_weight_att)

        # CLUB sample 互信息，尽可能让原始的embedding与KG-enhanced的表示相互独立。
        self.club_mi = MyCLUB(self.emb_dim, self.emb_dim, self.dim_hidden)

        # 序列表示学习
        # self.visit_att = MAB_diffV_visit(emb_dim, emb_dim, emb_dim, num_heads, ln=ln)
        # self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True, num_layers=2)
        self.visit_self_att = MultiHeadAttention(1, emb_dim, emb_dim, emb_dim, dropout=0.3)
        # self.visit_kg_att = MultiHeadAttention(num_heads, emb_dim, emb_dim, emb_dim, dropout=0.3)
        self.med_agg =nn.Linear(2*emb_dim, emb_dim)
        self.visit_dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(emb_dim, voc_size[2])

        # 这里使用recurrent-transformer来编码visit-level的时序信息
        self.dim_hidden = dim_hidden
        # self.recurrent_attn = BlockRecurrentAttention(dim_hidden*3, dim_hidden*3)

        self.softmax = nn.Softmax(dim=-1)
        # self.output_layer = nn.Linear(emb_dim, voc_size[2])
        
        self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)       

        ## KG init
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.hist_latent_emb = nn.Parameter(self.hist_latent_emb)
        self.kg_gcn = self._init_model()

        self.diag_kg_agg = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.proc_kg_agg = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.med_kg_agg = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.Tanh(),
            nn.Dropout(0.3)
        )

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_dim))
        self.latent_emb = initializer(torch.empty(self.n_intents, self.emb_dim))
        self.hist_latent_emb = initializer(torch.empty(self.n_intents, self.emb_dim))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.kg_adj_mat).to(self.device)
    
    def _init_model(self):
        return GraphConv(channel=self.emb_dim,
                         n_hops=self.context_hops,
                         n_users=self.kg_leaf,
                         n_relations=self.n_relations,
                         n_factors=self.n_intents,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device) 

    def forward(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20):
        device = self.device
        
        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        # 1. 首先计算code-level的embedding
        diag_emb = self.diag_embedding(diseases) # [batch_size, diag_code_len, emb]
        proc_emb = self.proc_embedding(procedures) # [batch_size, proc_code_len, emb]
        
        # 2. 由于medication需要增加一个padding的visit记录。构造一个new_medication，表示上一个visit的记录，然后与[0,t-1]时刻的medication进行拼接
        new_medication = torch.full((batch_size, 1, max_med_num), self.MED_PAD_TOKEN).to(device)
        new_medication = torch.cat([new_medication, medications[:, :-1, :]], dim=1) # new_medication.shape=[b,max_seq_len, max_med_num]
        # m_mask_matrix 同样也需要移动
        new_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device) # 这里用较大负值，避免softmax之后分走了概率
        new_m_mask = torch.cat([new_m_mask, m_mask_matrix[:, :-1, :]], dim=1)
        med_emb = self.med_embedding(new_medication)

        # 3. KG 的表示，获得与medical code相关的节点表示
        kg_leaf_emb = self.all_embed[:self.kg_leaf, :] # 是用新的embedding还是用三个embedding的合并？
        kg_entity_emb = self.all_embed[self.kg_leaf:, :]
        # kg_leaf_intent_embed: 表示根据KG采用intent聚合的leaf信息；kg_leaf_intact_emb表示根据KG直接聚合的leaf信息
        entity_gnn_embed, kg_leaf_intent_embd, intent_cor, kg_leaf_intact_emb = self.kg_gcn(
                                                            kg_leaf_emb, 
                                                            kg_entity_emb, 
                                                            self.latent_emb, 
                                                            self.kg_edge_index, 
                                                            self.kg_edge_type, 
                                                            self.interact_mat, 
                                                            mess_dropout=self.mess_dropout,
                                                            node_dropout=self.node_dropout
                                                            )
        # 3.1 根据历史的code记录从KG中选择数据，使用attention
        kg_code_intent_embd = kg_leaf_intent_embd.unsqueeze(0).repeat(batch_size*max_seq_length, 1, 1)

        # # 3.1.4 根据历史信息使用intent来聚合节点信息。
        diag_enc_input = diag_emb.view(batch_size*max_seq_length,max_diag_num, -1)
        d_enc_mask_matrix = d_mask_matrix.view(batch_size*max_seq_length,max_diag_num).unsqueeze(-1)
        diag_kg_int_weight = torch.matmul(diag_enc_input, kg_code_intent_embd.transpose(1,2)) # batch_size*s, diag_len, vocab_size
        diag_kg_int_weight = self.softmax(diag_kg_int_weight)
        diag_kg_int_weight = diag_kg_int_weight.masked_fill(d_enc_mask_matrix.to(bool), 0)
        diag_kg_interact = torch.matmul(diag_kg_int_weight, kg_code_intent_embd)

        p_enc_mask_matrix = p_mask_matrix.view(batch_size*max_seq_length,max_proc_num).unsqueeze(-1)
        proc_enc_input = proc_emb.view(batch_size*max_seq_length,max_proc_num, -1)
        proc_kg_int_weight = torch.matmul(proc_enc_input, kg_code_intent_embd.transpose(1,2))
        proc_kg_int_weight = self.softmax(proc_kg_int_weight)
        proc_kg_int_weight = proc_kg_int_weight.masked_fill(p_enc_mask_matrix.to(bool), 0)
        proc_kg_interact = torch.matmul(proc_kg_int_weight, kg_code_intent_embd) # batch_size*s, proc_len, emb

        m_enc_mask_matrix = new_m_mask.view(batch_size*max_seq_length,max_med_num).unsqueeze(-1)
        m_enc_input = med_emb.view(batch_size*max_seq_length,max_med_num, -1)
        med_kg_int_weight = torch.matmul(m_enc_input, kg_code_intent_embd.transpose(1,2)) # batch_size*s, med_len, vocab_size
        med_kg_int_weight = self.softmax(med_kg_int_weight)
        med_kg_int_weight = med_kg_int_weight.masked_fill(m_enc_mask_matrix.to(bool), 0)
        med_kg_interact = torch.matmul(med_kg_int_weight, kg_code_intent_embd)     

        # 4. 根据intent聚合的leaf rep获得新的code表示
        kg_pad_weight = torch.zeros((3, self.emb_dim)).to(device)
        diag_intent_kg_rep = nn.Embedding.from_pretrained(torch.cat([kg_leaf_intent_embd[:self.voc_size[0], :], kg_pad_weight], dim=0), freeze=True)
        proc_intent_kg_rep = nn.Embedding.from_pretrained(torch.cat([kg_leaf_intent_embd[self.voc_size[0]: self.voc_size[0]+self.voc_size[1], :], kg_pad_weight], dim=0), freeze=True)
        med_intent_kg_rep = nn.Embedding.from_pretrained(torch.cat([kg_leaf_intent_embd[self.voc_size[0]+self.voc_size[1]:, :], kg_pad_weight], dim=0), freeze=True)
        diag_intent_embedding = diag_intent_kg_rep(diseases).view(batch_size*max_seq_length,max_diag_num, -1)
        proc_intent_embedding = proc_intent_kg_rep(procedures).view(batch_size*max_seq_length,max_proc_num, -1)
        med_intent_embedding = med_intent_kg_rep(new_medication).view(batch_size*max_seq_length,max_med_num, -1)

        # 5. 将kg-intent和hist-intent的表示汇总
        curt_diag_rep = self.diag_kg_agg(torch.cat([diag_kg_interact, diag_intent_embedding],dim=-1)).view(batch_size, max_seq_length, max_diag_num, -1)
        curt_proc_rep = self.proc_kg_agg(torch.cat([proc_kg_interact, proc_intent_embedding],dim=-1)).view(batch_size, max_seq_length, max_proc_num, -1)
        curt_med_rep = self.med_kg_agg(torch.cat([med_kg_interact, med_intent_embedding],dim=-1)).view(batch_size, max_seq_length, max_med_num, -1)      
        
        # 6.将三种code分别aggregate，转化为visit-level，
        curt_diag_enc = torch.sum(curt_diag_rep, dim=2).view(batch_size, max_seq_length, -1)
        curt_proc_enc = torch.sum(curt_proc_rep, dim=2).view(batch_size, max_seq_length, -1)
        curt_med_enc = torch.sum(curt_med_rep, dim=2).view(batch_size, max_seq_length, -1)
        curt_visit_enc = self.curt_agg(torch.cat([curt_diag_enc, curt_proc_enc, curt_med_enc], dim=-1)) # [batch_size, max_seq_length, 3*hdm]
        ori_diag_enc = torch.sum(diag_emb, dim=2).view(batch_size, max_seq_length, -1)
        ori_proc_enc = torch.sum(proc_emb, dim=2).view(batch_size, max_seq_length, -1)
        ori_med_enc = torch.sum(med_emb, dim=2).view(batch_size, max_seq_length, -1)
        original_visit_enc =self.ori_agg(torch.cat([ori_diag_enc, ori_proc_enc, ori_med_enc], dim=-1))
        code_mi = []
        for idx, seq_len in enumerate(seq_length.tolist()):
            tmp_curt_enc = curt_visit_enc[idx, :seq_len, :].reshape(seq_len, -1)
            tmp_ori_enc = original_visit_enc[idx, :seq_len, :].reshape(seq_len, -1)
            tmp_mi = self.club_mi(tmp_ori_enc, tmp_curt_enc) # 为什么小于0呢？debug一下
            code_mi.append(tmp_mi)
        code_mi = torch.stack(code_mi)
        code_mi_loss = torch.mean(code_mi)

        # 7. 将原始embedding和扩充后的embedding一起作为新的表示
        new_visit_rep = torch.add(original_visit_enc, curt_visit_enc) # [B, max_len, emb_size]
        # 8. 序列的表示学习
        visit_mask = torch.full((batch_size, max_seq_length), 0).to(device)
        for i, v_l in enumerate(seq_length):
            visit_mask[i, :v_l] = 1
        # # # 8.1 以med作为K，V
        # kg_rep = kg_leaf_intent_embd.unsqueeze(0).repeat(batch_size,1,1)
        new_med_rep = self.med_agg(torch.cat([ori_med_enc, curt_med_enc], dim=-1))
        tri_mask = torch.tril(torch.ones(new_visit_rep.shape[1], new_visit_rep.shape[1])).to(self.device)
        visit_tri_mask = tri_mask
        visit_tri_mask = visit_tri_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        visit_output1, _ = self.visit_self_att(new_visit_rep, new_med_rep, new_med_rep, visit_tri_mask)
        # visit_output2, _ = self.visit_kg_att(new_visit_rep, kg_rep, kg_rep)
        # visit_output = self.visit_agg(torch.cat([visit_output1, visit_output2], dim=-1))
        visit_output = visit_output1 #+ ori_med_enc
        
        # 8.2 直接使用RNN
        # h0 = torch.zeros_like(new_visit_rep)
        # visit_output, _ = self.gru(new_visit_rep)

        visit_output = torch.matmul(visit_output, self.med_embedding[0].weight[:self.voc_size[2],:].repeat(batch_size, 1, 1).transpose(1,2))#self.output_layer(visit_output)
        visit_output = visit_output * visit_mask.unsqueeze(-1)

        neg_pred_prob = F.sigmoid(visit_output)
        neg_pred_prob = torch.einsum("nsc,nsk->nsck",[neg_pred_prob, neg_pred_prob])  # [seq, voc_size, voc_size]


        return visit_output, 0 , self.codeMI_regularity*code_mi_loss, self.sim_regularity*intent_cor
        #, curt_visit_enc, original_visit_enc

