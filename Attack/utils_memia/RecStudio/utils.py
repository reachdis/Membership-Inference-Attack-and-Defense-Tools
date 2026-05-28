from typing import Dict, Union
from recstudio.utils import get_model, print_logger, color_dict_normal, set_color, parser_yaml
from utils import *
import torch
from recstudio.data.dataset import SeqDataset
from recstudio.ann import sampler
import numpy as np
import recstudio.model.loss_func as lfc
import os
import shutil
from torch.nn.utils.rnn import pad_sequence
from typing import Sized, Dict, Optional, Iterator, Union
import torch.optim as optim
import time
import copy

def init(model_name, dataset, epochs=30, max_seq_len=20, val_check=True, early_stop_patience=10, _best_ckpt_path=None, lr=None, cutoff=10, gpu=None):
    model_class, model_conf = get_model(model_name)
    model = model_class(model_conf)
    if gpu is not None:
        model.config['gpu'] = [gpu]
    if lr is not None:
        model.config['learning_rate'] = lr
    if _best_ckpt_path is not None:
        model._best_ckpt_path = _best_ckpt_path
        model.config['save_path'] = '/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved'
    model.config['cutoff'] = cutoff
    dataset_class = model._get_dataset_class()
    datasets = dataset_class(name=dataset, config={'max_seq_len': max_seq_len}).build()

    model.config['early_stop_patience'] = early_stop_patience

    print_logger.info(f"{datasets[0]}")
    print_logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))

    if val_check:
        model.fit(*datasets[:2], run_mode='light', epochs=epochs)
        save_path = os.path.join(model.config['save_path'])
        best_ckpt_path = os.path.join(save_path, model._best_ckpt_path)
        model.load_checkpoint(best_ckpt_path)
        model.evaluate(datasets[-1])
    else:
        model.fit(datasets[0], run_mode='light', epochs=epochs)

    return model, datasets


def add_topk(datasets, model, topk_shuffle=False):
    datasets.eval_mode = True  
    fiid2token2idx = model.fiid2token2idx
    oral_fiid2token2idx = datasets.field2token2idx[datasets.fiid]
    datasets.gen_idx2idx(fiid2token2idx)

    model_idx2token = {}
    for key in fiid2token2idx.keys():
        model_idx2token[fiid2token2idx[key]] = key
        
    id2id = torch.LongTensor(list(model_idx2token.keys()))
    for i in range(len(id2id)):
        id2id[i] = oral_fiid2token2idx[model_idx2token[i]]

    k = model.config['cutoff']
    n_samples = len(datasets.data_index)
    topk_items = torch.LongTensor(np.zeros((n_samples, k)))
    data_loader = datasets.loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'], shuffle=False, drop_last = False)
    for _, data in enumerate(data_loader):
        data = model._to_device(data, model.device)
        _, topk_item = model.topk(model.query_encoder(data), k=k, user_h=data['user_hist'])
        indexs = data['index'].cpu().detach().clone()
        topk_item = topk_item.cpu().detach().clone()
        topk_items[indexs] = id2id[torch.LongTensor(topk_item)]
    if topk_shuffle:
        topk_items = topk_items.numpy()
        for i in range(topk_items.shape[0]):
            np.random.shuffle(topk_items[i,:])
        topk_items = torch.LongTensor(topk_items)
    datasets.topk_items = topk_items
    datasets.gen_idx2idx(oral_fiid2token2idx)

def eva(model, datasets):
    fiid2token2idx = model.fiid2token2idx

    datasets[0].gen_idx2idx(fiid2token2idx)
    output0 = model.evaluate(datasets[0])
    del datasets[0].id2id

    datasets[1].gen_idx2idx(fiid2token2idx)
    output1 = model.evaluate(datasets[1])
    del datasets[1].id2id

    datasets[2].gen_idx2idx(fiid2token2idx)
    output2 = model.evaluate(datasets[2])
    del datasets[2].id2id
    return (output0, output1, output2)


def model_stealing(model_name, train_datasets, val_datasets, epochs=30, active_sampler='active_sampler', _best_ckpt_path=None, val_check=True, lr=None, cutoff=10, load_save=True, gpu=None, mode1=None, mode2=None, temp=1):
    
    model_class, model_conf = get_model(model_name)
    model = model_class(model_conf)
    if model_name == 'BERT4Rec':
        model.config['steal'] = True
        model.config['pooling_type'] = 'last'
    if gpu is not None:
        model.config['gpu'] = [gpu]
    if _best_ckpt_path is not None:
        model._best_ckpt_path = _best_ckpt_path
        model.config['save_path'] = '/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved'
    if lr is not None:
        model.config['learning_rate'] = lr
    model.config['cutoff'] = cutoff
    if load_save:
        model.config['early_stop_patience'] = 100   
    model.config['val_metrics'].append('agg')
    model.config['test_metrics'].append('agg')

    if mode1 is None or mode2 is None:
        model.loss_fn = lfc.ModelStealingLoss(train_datasets.config['lamda1'], train_datasets.config['lamda2'])
    else:
        model.loss_fn = lfc.ExtractionLoss(train_datasets.config['lamda1'], train_datasets.config['lamda2'], mode1, mode2, temp)
    model.sampler = sampler.UniformSampler(train_datasets.num_items-1)
    model.active_sampler = active_sampler

    print_logger.info(f"{train_datasets}")
    print_logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))

    if val_check:
        model.fit(train_datasets, val_datasets, run_mode='light', epochs=epochs)
        save_path = os.path.join(model.config['save_path'])
        best_ckpt_path = os.path.join(save_path, model._best_ckpt_path)
        if load_save:
            model.load_checkpoint(best_ckpt_path)
    else:
        model.fit(train_datasets, run_mode='light', epochs=epochs)
    return model
    

def _score(model, data_loader, fiid, fuid):
    softmax = torch.nn.Softmax(dim=1)
    dscore = {}
    for i, data in enumerate(data_loader):
        data = model._to_device(data, model.device)

        pos_score, all_item_score = model(data, True)
        user_embeddings = model.query_encoder(data)
        top_score, _ = model.topk(user_embeddings, k=model.config['cutoff'], user_h=data['user_hist'])

        softmax_score = softmax(all_item_score)
        mean_score = torch.mean(all_item_score,axis=1)
        std_score = torch.std(all_item_score,axis=1)
        top1_score = torch.max(top_score,dim=1).values
        topk_score = torch.min(top_score,dim=1).values
        mean_topk_score = torch.mean(top_score,axis=1)
        user_ids = data[fuid]
        end = data['end'].cpu().detach().clone().numpy()
        user_embeddings = user_embeddings.cpu().detach().clone().numpy()
        top_score = top_score.cpu().detach().clone().numpy()

        for i in range(len(user_ids)):
            user_id = user_ids[i].item()
            if user_id not in dscore.keys():
                dscore[user_id] = {'pos_score':[],'mean_score':[],'top1_score':[],'topk_score':[],\
                    'mean_topk_score':[], 'softmax_score':[], 'features':[], 'end':[]}
                    #, 'in_item_id':[], 'item_id':[]}
            dscore[user_id]['pos_score'].append(pos_score[i].item())
            dscore[user_id]['mean_score'].append(mean_score[i].item())
            dscore[user_id]['top1_score'].append(top1_score[i].item())
            dscore[user_id]['topk_score'].append(topk_score[i].item())
            dscore[user_id]['mean_topk_score'].append(mean_topk_score[i].item())
            dscore[user_id]['softmax_score'].append(softmax_score[i][data[fiid][i].item()-1].item())
            dscore[user_id]['end'].append(end[i])
            dscore[user_id]['features'].append(np.hstack((user_embeddings[i], pos_score[i].item(), mean_score[i].item(), \
                top_score[i], std_score[i].item())))
    return dscore

def score(model, datasets):
    fiid = datasets[0].fiid
    fuid = datasets[0].fuid
    oral_fiid2token2idx = datasets[0].field2token2idx[datasets[0].fiid]
    fiid2token2idx = model.fiid2token2idx

    datasets[0].gen_idx2idx(fiid2token2idx)
    datasets[1].gen_idx2idx(fiid2token2idx)
    datasets[2].gen_idx2idx(fiid2token2idx)

    model.eval()

    sampler = model.sampler
    model.sampler = None
    loss_fn = model.loss_fn
    model.loss_fn = lfc.BinaryCrossEntropyLoss()

    train_data_loader = datasets[0].eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])
    val_data_loader = datasets[1].eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])
    test_data_loader = datasets[2].eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])
    
    train_d_score = _score(model, train_data_loader, fiid, fuid)
    val_d_score = _score(model, val_data_loader, fiid, fuid)
    test_d_score = _score(model, test_data_loader, fiid, fuid)

    datasets[0].gen_idx2idx(oral_fiid2token2idx)
    datasets[1].gen_idx2idx(oral_fiid2token2idx)
    datasets[2].gen_idx2idx(oral_fiid2token2idx)

    model.sampler = sampler
    model.loss_fn = loss_fn
    return train_d_score, val_d_score, test_d_score


def sta(dscore, mode='mean'):
    members = dscore.keys()
    values = {'pos_score':[],'mean_score':[],'top1_score':[],'topk_score':[],'mean_topk_score':[]}
    for m in members:
        for key in dscore[m].keys():
            if key in values.keys():
                values[key].append(np.mean(np.array(dscore[m][key])))

    for v in values.keys():
        if mode == 'min':
            values[v] = np.min(np.array(values[v]))
        elif mode == 'max':
            values[v] = np.max(np.array(values[v]))
        else:
            values[v] = np.mean(np.array(values[v]))
    return values

# mode: top1, topk, random

def wr_config(old, new):
    path = os.path.join('recstudio/data/config', new + '.yaml')
    config_file = os.path.join('recstudio/data/config', old + '.yaml')
    with open(config_file, 'r') as rfile:
        contents = rfile.readlines()
        rfile.close()
    with open(path, 'w') as file:
        for content in contents:
            c = content.replace(old, new)
            file.write(c)
        file.close()


def gen_data_file(new, his_items, oral_datasets):
    inter_feat_name = oral_datasets.config['inter_feat_name']
    user_feat_name = oral_datasets.config['user_feat_name'][0]
    item_feat_name = oral_datasets.config['item_feat_name'][0]
    if not os.path.exists(os.path.join('../datasets', new)):
        os.makedirs(os.path.join('../datasets', new))
    
    num_users, topk = his_items.shape 
    num_items = len(oral_datasets.field2token2idx[oral_datasets.fiid]) - 1
    wr_ui(new, num_users, user_feat_name)

    suffix = inter_feat_name.split('.')[-1]
    old = inter_feat_name.split('.')[0]
    shutil.copy(os.path.join('../datasets', old, item_feat_name), os.path.join('../datasets', new, new + '.item'))

    with open(os.path.join('../datasets', old, inter_feat_name), 'r') as file1:
        contents = file1.readlines()
        cstrs0 = contents[0]
        file1.close()

    suffix = inter_feat_name.split('.')[-1]
    with open(os.path.join('../datasets', new, new + '.' + suffix), 'w') as file2:
        file2.write(cstrs0)
        timestamp = 1
        for i in range(num_users):
            for j in range(topk):
                file2.write(str(i+1) + '\t' + str(his_items[i][j]) + '\t' + str(5) + '\t' + str(timestamp) + '\n')
                timestamp = timestamp + 1
        

def wr_ui(new, num, feat_name):
    suffix = feat_name.split('.')[-1]
    old = feat_name.split('.')[0]
    with open(os.path.join('../datasets', old, feat_name), 'r') as file1:
        contents = file1.readlines()
        cstrs0 = contents[0]
        cstrs1 = contents[1].split()[1:]
        cstrs11 = ''
        for cst in cstrs1:
            cstrs11 = cstrs11 + '\t' + cst
        cstrs11 = cstrs11 + '\n'
        file1.close()

    with open(os.path.join('../datasets', new, new + '.' +suffix), 'w') as file2:
        file2.write(cstrs0)
        for i in range(1,num+1):
            file2.write(str(i))
            file2.write(cstrs11)
        file2.close()

def gen_data(gen_dataset, model, gen_len, num_users, oral_dataset, mode='top1'):
    # generata config file
    topk = model.config['cutoff']
    wr_config(oral_dataset, gen_dataset)
    
    token2idx = model.fiid2token2idx
    idx2token = np.zeros(len(token2idx))
    for key in token2idx.keys():
        if key == '[PAD]':
            idx2token[token2idx[key]] = 0
        else:
            idx2token[token2idx[key]] = int(key)

    num_items = len(model.fiid2token2idx)
    his_items = torch.randint(1,num_items,(num_users,1))
    assert mode in ['top1', 'topk', 'random'], 'Mode Error!'

    oral_datasets = SeqDataset(name=oral_dataset, config=None).build()
    max_len = oral_datasets[0].config['max_seq_len']
    
    for i in range(1, gen_len):
        data = {}
        data['user_id'] = torch.LongTensor(np.arange(num_users))
        seqlen = min(i, max_len)
        data['seqlen'] = torch.LongTensor(np.ones(num_users)*seqlen)
        data['in_item_id'] = his_items[:,-seqlen:]
        data['user_hist'] = torch.zeros_like(data['in_item_id'])

        data = model._to_device(data, model.device)
        _, topk_items = model.topk(model.query_encoder(data), k=topk, user_h=data['user_hist'])
        topk_items = topk_items.detach().clone()
        if mode == 'random':
            index = torch.randint(0, topk, (num_users,1))
        elif mode == 'topk':
            index = torch.randint(topk-1, topk, (num_users,1))
        else:
            index = torch.randint(0, 1, (num_users,1))

        data['item_id'] = torch.LongTensor(np.arange(num_users))
        for j in range(num_users):
            data['item_id'][j]  = topk_items[j][index[j][0]]

        data['rating'] = torch.LongTensor(np.ones(num_users)*5)
        his_items = torch.hstack((his_items, data['item_id'].cpu().clone().unsqueeze(1)))

    his_items = his_items.numpy()
    for i in range(gen_len):
        his_items[:,i] = idx2token[his_items[:,i]]

    gen_data_file(gen_dataset, his_items, oral_datasets[0])
    return his_items

from recstudio.data.dataset import *
def get_adv_scores(model, datasets):
    fiid = datasets.fiid
    if model is not None:
        k = model.config['cutoff']
    n_samples = len(datasets.data_index)
    user_ids = np.zeros(n_samples)
    scores = np.zeros(n_samples)
    data_loader = datasets.loader(batch_size=512, num_workers=0, shuffle=False, drop_last = False)
    for _, data in enumerate(data_loader):
        index = data['index'].detach().clone().numpy()
        user_ids[index] = data['user_id'].detach().clone().numpy()
        data = model._to_device(data, model.device)
        _, top_item = model.topk(model.query_encoder(data), k=k, user_h=data['user_hist'])
        top_item = top_item[:,:k].cpu().detach().clone().numpy()
        target_item = data['topk_' + fiid][:,:k].cpu().detach().clone().numpy()
        bsize, _ = top_item.shape
        agg_score = []
        for i in range(bsize):
            agg_score.append(len(set(top_item[i]).intersection(set(target_item[i]))) / k)
        scores[index] = np.array(agg_score)

    user_ids.astype(int)
    return user_ids, scores

def item_sim_func(model):
    embeddings = model.item_encoder.weight.cpu().detach().clone().numpy()
    num_item, n_embed = embeddings.shape
    item_sim = np.zeros((num_item, num_item))
    for i in range(num_item):
        for j in range(num_item):
            if j == i:
                item_sim[i][j] = 0
            else:
                item_sim[i][j] = np.dot(embeddings[i], embeddings[j])
    return item_sim

class NewSeqDataset(SeqDataset):

    def __getitem__(self, index):
        r"""Get data at specific index.

        Args:
            index(int): The data index.
        Returns:
            dict: A dict contains different feature.
        """
        #----------------
        data = {}
        data['seqlen'] = torch.ones(len(index)) * 20 
        data['seqlen'] = data['seqlen'].long()
        data['index'] = index
        data['in_' + self.fiid] = self.in_item_id[index]
        data[self.fiid] = self.item_id[index]

        if hasattr(self, 'topk_items'):
            data['topk_' + self.fiid] = self.topk_items[index]

        if hasattr(self, 'id2id'):
            data['in_' + self.fiid] = self.trans(data['in_' + self.fiid])
            data[self.fiid] = self.trans(data[self.fiid])

            if hasattr(self, 'topk_items'):
                data['topk_' + self.fiid] = self.trans(data['topk_' + self.fiid])
        data['user_hist'] = torch.zeros_like(data['in_' + self.fiid])
        data[self.frating] = torch.zeros_like(data['in_' + self.fiid])
        return data

from recstudio.model import loss_func

def search_nei(embeddings, emb, sort='max'):
    n_item, _ = embeddings.shape
    sim = np.zeros(n_item)
    for i in range(n_item):
        sim[i] = np.dot(emb, embeddings[i])
    return np.argsort(-sim) if sort=='max' else np.argsort(sim)
        
def gen_aug_datasets(model, adv_scores, item_sim, datasets, sample_num=500, pos='random', adv_num=10, mode='real', sort='max'):
    sort_index = np.argsort(adv_scores)
    select_index = sort_index[:sample_num]
    num_item = len(item_sim)
    neis = np.zeros((num_item, adv_num))
    

    if mode != 'real':
        sort = 'random'

    for i in range(len(item_sim)):
        if sort == 'max':
            neis[i] = np.argsort(-item_sim[i])[:adv_num]
        elif sort == 'min':
            neis[i] = np.argsort(item_sim[i])[:adv_num]
        else:
            neis[i] = np.random.choice(len(item_sim[i]), adv_num, replace=False)

    select_in_item_id = datasets[select_index]['in_' + datasets.fiid]
    _, n_items = select_in_item_id.shape
    select_item_id = datasets[select_index][datasets.fiid]
    select_in_item_ids = torch.zeros((sample_num*adv_num + len(datasets),n_items))
    select_item_ids = torch.zeros(sample_num*adv_num + len(datasets)) 

    for i in range(sample_num):
        idx = select_index[i]
        if i % 1000 == 0:
            print(i)
        data = datasets[torch.tensor([idx])]
        user_hist = data['in_'+model.fiid]
        data = model._to_device(data, model.device)
        if pos == 'random':
            r_ix = np.random.choice(len(user_hist[0]), 1)[0]
        else:
            r_ix = len(user_hist[0]) - 1
        ix = user_hist[0][r_ix]
            
        nei = neis[ix]

        for j in range(adv_num):
            select_item_ids[i*adv_num + j] = select_item_id[i]
            if mode == 'real':
                select_in_item_ids[i*adv_num + j] = select_in_item_id[i]
                select_in_item_ids[i*adv_num + j][r_ix] = torch.tensor(nei[j])
            else:
                select_in_item_ids[i*adv_num + j] = torch.tensor(np.random.choice(num_item, len(select_in_item_id[i]), replace=True))

    select_in_item_ids[-len(datasets):,:] =  datasets[np.arange(len(datasets))]['in_' + datasets.fiid]
    select_item_ids[-len(datasets):] = datasets[np.arange(len(datasets))][datasets.fiid]
    
    return select_in_item_ids.long(), select_item_ids.long()

