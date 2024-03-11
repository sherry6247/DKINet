from collections import defaultdict
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import torch
import scipy.sparse as sp

## KG 相关的数据预处理
# 1.首先尝试先在模型外面直接使用gdc， failed，要占用的内存太大了
def gdc(args, data):
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data) 
    return data

class Voc(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[len(self.word2idx)] = word
            self.word2idx[word] = len(self.word2idx)
# 2. 采用KGIN模型的输入数据,将与数据对应的kg节点单独列出，按照D，P，M的顺序保存
def KGIN_data_loader(kg_triples, kg_voc, data_voc):
    """
    kg_triple:[h,r,t]
    data_voc: diag_voc, proc_voc, med_voc
    train_data: [[([d,p,m]),(),...],[]]
    """
    n_leaf_nodes = 0
    n_entitys = 0
    diag2cui, proc2cui, med2cui, kg2idx, rel2idx = kg_voc['diag2cui'], kg_voc['proc2cui'], kg_voc['med2cui'], kg_voc['kg2id'], kg_voc['rel2id']
    diag_voc, pro_voc, med_voc = data_voc['diag_voc'], data_voc['pro_voc'], data_voc['med_voc']
    kg_graph = nx.MultiDiGraph()
    relation_dict = defaultdict(list)
    # 首先将kg中的idx重新映射
    new_kg_voc = Voc()
    # 将code与KG CUI的关系加到 relation_dict中
    code_cui = []
    for diag in diag_voc.word2idx:
        diag = diag.split('_')[0]
        try:
            diag_cui = diag2cui.code2cui[diag]
            code_cui.append(('d_'+diag, diag_cui))
        except:
            code_cui.append(('d_'+diag, 'd_'+diag))
        new_kg_voc.add_word('d_'+diag)
        n_leaf_nodes += 1
    for proc in pro_voc.word2idx:
        proc = proc.split('_')[0]
        try:
            proc_cui = proc2cui.code2cui[proc]
            code_cui.append(('p_'+proc, proc_cui))
        except:
            code_cui.append(('p_'+proc, 'p_'+proc))
        new_kg_voc.add_word('p_'+proc)
        n_leaf_nodes += 1
    for med in med_voc.word2idx:
        med_cui = med2cui.code2cui[med]
        code_cui.append(('m_'+med, med_cui))
        new_kg_voc.add_word('m_'+med)
        n_leaf_nodes += 1
    for entity in kg2idx.kg2idx:
        new_kg_voc.add_word(entity)

    n_nodes = len(new_kg_voc.word2idx) # 9475
    n_entities = n_nodes - n_leaf_nodes # 9475 - 3519

    print('\n Begin to load the code&CUI interaction triples...')
    rel2idx.add_kg('co_occur') # 将code和CUI的关系定义为co_occur，和KG的关系区分开来
    for cc in tqdm(code_cui):
        code, cui = cc
        c1_idx = new_kg_voc.word2idx[code]
        c2_idx = new_kg_voc.word2idx[cui]
        r_idx = rel2idx.kg2idx['co_occur']
        relation_dict[r_idx].append([c1_idx, c2_idx])
    
    print('\n Begin to load knowledge graph triples...')
    for triple in tqdm(kg_triples, ascii=True):
        h, r, t = triple 
        h_idx = new_kg_voc.word2idx[h] - n_leaf_nodes
        t_idx = new_kg_voc.word2idx[t] - n_leaf_nodes #  remove the n_leaf_KG  in kg_graph
        r_idx = rel2idx.kg2idx[r]
        kg_graph.add_edge(h_idx, t_idx, key=r_idx)
        relation_dict[r_idx].append([h_idx, t_idx])
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict, rel2idx, n_nodes, n_leaf_nodes, n_entities)
    return n_nodes, n_entities, n_leaf_nodes, kg_graph, adj_mat_list, norm_mat_list, mean_mat_list
    
# build sparse relational graph
def build_sparse_relational_graph(relation_dict, rel2idx, n_nodes, n_leaf_nodes, n_entitys):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        # if r_id == 0:
        #     cf = np_mat.copy()
        #     cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
        #     vals = [1.] * len(cf)
        #     adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        # else:
        vals = [1.] * len(np_mat)
        adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))

        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_leaf_nodes, n_leaf_nodes:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_leaf_nodes, n_leaf_nodes:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list
    


    





# 构建图
def build_graph(kg_triple, kgvoc, relvoc, save_path, args):
    """
    kg_triple:[h,r,t]
    """
    # from torch_geometric.data import Data
    #    data = Data(x=x, edge_index=edge_index, ...)
    x = np.array(list(kgvoc.idx2kg.keys()))
    edge_index = torch.from_numpy(np.zeros(shape=(2, len(kg_triple)))).long().to(args.device)
    kg_graph = nx.MultiDiGraph()
    for i, triple in tqdm(enumerate(kg_triple), desc='build graph'):
        h, r, t = triple
        h_idx = kgvoc.kg2idx[h]
        t_idx = kgvoc.kg2idx[t]
        r_idx = relvoc.kg2idx[r]
        edge_index[0,i] = h_idx
        edge_index[1,i] = t_idx
        kg_graph.add_edge(h_idx, t_idx, key=r_idx)

        # if i == 1000:
        #     draw_graph(kgvoc, kg_graph, save_path, type='test_1000')
        #     break
    data = Data(x = x, edge_index=edge_index)
    # draw kg_graph
    # draw_graph(kgvoc, kg_graph, save_path, type='orginal')
    return data, kg_graph

# example graph 
def example_graph(kg_triple, kg_voc, voc, save_path, args):
    """
    "diag":"85221,5180,E8844,2989,2449,42731,4019,4263,412,V103,V4571",
    "proc":"0131",
    "med":"H03A,C02D,N03A,J01D,C03C,N02A,A12B,C07A,A01A,B05C,N02B,B01A,A12C,A02B,A06A"    
    """
    diag2cui, proc2cui, med2cui, kg_voc, rel_voc = kg_voc['diag2cui'], kg_voc['proc2cui'], kg_voc['med2cui'], kg_voc['kg2id'], kg_voc['rel2id']
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    diag_list = "85221,5180,E8844,2989,2449,42731,4019,4263,412,V103,V4571"
    proc_list = "0131"
    med_list = "H03A,C02D,N03A,J01D,C03C,N02A,A12B,C07A,A01A,B05C,N02B,B01A,A12C,A02B,A06A" 
    def convert_list(str_list):
        s = str_list.split(',')
        return s
    diag_list = convert_list(diag_list)
    proc_list = convert_list(proc_list)
    med_list = convert_list(med_list)
    code_cui_list = []

    kg_graph = nx.MultiDiGraph()
    for d in diag_list:
        d_id = diag_voc.word2idx[d]
        d_cui = diag2cui.code2cui[d_id]
        code_cui_list.append(d_cui)
        kg_graph.add_edge(d, kg_voc.kg2idx[d_cui])
    for p in proc_list:
        p_id = pro_voc.word2idx[p]
        p_cui = proc2cui.code2cui[p_id]
        code_cui_list.append(p_cui)
        kg_graph.add_edge(p, kg_voc.kg2idx[p_cui])
    for m in med_list:
        m_id = med_voc.word2idx[m]
        m_cui = med2cui.code2cui[m_id]
        code_cui_list.append(m_cui)
        kg_graph.add_edge(m, kg_voc.kg2idx[m_cui])
    
    for triple in kg_triple:
        h,r,t = triple
        h_idx = kg_voc.kg2idx[h]
        t_idx = kg_voc.kg2idx[t]
        r_idx = rel_voc.kg2idx[r]
        if h in code_cui_list:
            kg_graph.add_edge(h_idx, t_idx, key=r_idx)
    draw_graph(kg_voc, diag_voc, pro_voc, med_voc, kg_graph, save_path, type='sample')

# draw graph 
def draw_graph(kgvoc, diag_voc, pro_voc, med_voc, G, save_path, type='orginal'):
    seed = 1234
    pos = nx.spring_layout(G, seed=seed)
    # pos = graphviz_layout(G, prog="dot")
    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    fig, ax = plt.subplots(figsize=(25,25))
    cmap = plt.cm.plasma

    labels = {}
    for n, lab in kgvoc.idx2kg.items():
        if n in pos:
            labels[n]=lab
    for n, lab in diag_voc.word2idx.items():
        if n in pos:
            labels[n]=n
    for n, lab in pro_voc.word2idx.items():
        if n in pos:
            labels[n]=n
    for n, lab in med_voc.word2idx.items():
        if n in pos:
            labels[n]=n
    

    n_labels = nx.draw_networkx_labels(G, pos, labels=labels) # {n:lab for n,lab in kgvoc.idx2kg.items() if n in pos}
    nodes = nx.draw_networkx_nodes(G, pos, node_size=50, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=50,
        arrowstyle="->",
        arrowsize=20,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )
    nx.draw_networkx_edge_labels(G, pos)
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.show()
    plt.savefig(save_path+'graph_{}.jpg'.format(type), dpi=500, bbox_inches = 'tight')
    print("example graph shown!!!")




def eicu_KGIN_data_loader(kg_triples, kg_voc, data_voc):
    """
    kg_triple:[h,r,t]
    data_voc: diag_voc, proc_voc, med_voc
    train_data: [[([d,p,m]),(),...],[]]
    """
    n_leaf_nodes = 0
    n_entitys = 0
    diag2cui, med2cui, kg2idx, rel2idx = kg_voc['diag2cui'], kg_voc['med2cui'], kg_voc['kg2id'], kg_voc['rel2id']
    diag_voc, med_voc = data_voc['diag_voc'], data_voc['med_voc']
    kg_graph = nx.MultiDiGraph()
    relation_dict = defaultdict(list)
    # 首先将kg中的idx重新映射
    new_kg_voc = Voc()
    # 将code与KG CUI的关系加到 relation_dict中
    code_cui = []
    for diag in diag_voc.word2idx:
        diag = diag.split('_')[0]
        try:
            diag_cui = diag2cui.code2cui[diag]
            code_cui.append(('d_'+diag, diag_cui))
        except:
            code_cui.append(('d_'+diag, 'd_'+diag))
        new_kg_voc.add_word('d_'+diag)
        n_leaf_nodes += 1
    # for proc in pro_voc.word2idx:
    #     proc = proc.split('_')[0]
    #     try:
    #         proc_cui = proc2cui.code2cui[proc]
    #         code_cui.append(('p_'+proc, proc_cui))
    #     except:
    #         code_cui.append(('p_'+proc, 'p_'+proc))
    #     new_kg_voc.add_word('p_'+proc)
    #     n_leaf_nodes += 1
    for med in med_voc.word2idx:
        med = str(med)
        code_cui.append(('m_'+med, 'm_'+med))
        new_kg_voc.add_word('m_'+med)
        n_leaf_nodes += 1
    for entity in kg2idx.kg2idx:
        new_kg_voc.add_word(entity)

    n_nodes = len(new_kg_voc.word2idx) # 9475
    n_entities = n_nodes - n_leaf_nodes # 9475 - 3519

    print('\n Begin to load the code&CUI interaction triples...')
    rel2idx.add_kg('co_occur') # 将code和CUI的关系定义为co_occur，和KG的关系区分开来
    for cc in tqdm(code_cui):
        code, cui = cc
        c1_idx = new_kg_voc.word2idx[code]
        c2_idx = new_kg_voc.word2idx[cui]
        r_idx = rel2idx.kg2idx['co_occur']
        relation_dict[r_idx].append([c1_idx, c2_idx])
    
    print('\n Begin to load knowledge graph triples...')
    for triple in tqdm(kg_triples, ascii=True):
        h, r, t = triple 
        h_idx = new_kg_voc.word2idx[h] - n_leaf_nodes
        t_idx = new_kg_voc.word2idx[t] - n_leaf_nodes #  remove the n_leaf_KG  in kg_graph
        r_idx = rel2idx.kg2idx[r]
        kg_graph.add_edge(h_idx, t_idx, key=r_idx)
        relation_dict[r_idx].append([h_idx, t_idx])
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict, rel2idx, n_nodes, n_leaf_nodes, n_entities)
    return n_nodes, n_entities, n_leaf_nodes, kg_graph, adj_mat_list, norm_mat_list, mean_mat_list