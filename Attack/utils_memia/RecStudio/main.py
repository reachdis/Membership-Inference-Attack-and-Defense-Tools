#!/usr/bin/env python
# coding: utf-8

import argparse
from pickle import FALSE
from random import choice
from typing import Dict, Union
from recstudio.utils import get_model, print_logger, color_dict_normal, set_color, parser_yaml
from utils import *
from MIAModel import *
import torch
from recstudio.data.dataset import SeqDataset
from recstudio.ann import sampler
import logging
import recstudio.model.loss_func as lfc
import sys

if __name__ == '__main__':
    mia_hidden_size = [64, 32, 8]
    argparser = argparse.ArgumentParser("MIA")
    argparser.add_argument('--target_model_name', type=str, default='NARM',help='The type of sequence model',choices=['NARM', 'SASRec', 'GRU4Rec', 'STAMP', 'TransRec', 'NPE', 'FPMC'])
    argparser.add_argument('--shadow_model_name', type=str, default='NARM',help='The type of sequence model',choices=['NARM', 'SASRec', 'GRU4Rec', 'STAMP', 'TransRec', 'NPE', 'FPMC'])
    argparser.add_argument('--know_model_name', type=int, default=1, help='make the model type of shadow model the same with the target model', choices=[1,0])
    argparser.add_argument('--dataset', type=str, default='ml-100k', help='Datasets', choices=['ml-100k','ml-1m', 'Amazon_Digital_Music', 'Amazon_Beauty', 'ta-feng', 'ml-10m', 'netflix']) 
    argparser.add_argument('--cutoff', type=int, default=10, help='The length of the recommendation list') 

    argparser.add_argument('--seq_epochs', type=int, default=100, help='The training epochs for sequence model') 
    argparser.add_argument('--max_seq_len', type=int, default=20, help='Max sequence lens for sequence model') 
    argparser.add_argument('--seq_val_check', type=int, default=0, help='Whether to use val datasets in the training process of sequence model', choices=[1,0])
    argparser.add_argument('--seq_early_stop_patience', type=int, default=20, help='Early stop patience for sequence model') 
    argparser.add_argument('--save_seq_model', type=int, default=1, help='Whether to save the seq_model', choices=[1,0])

    argparser.add_argument('--classifier_avg_mode', type=str, default='mean',help='The aggregation mode in the classifier',choices=['mean']) #Other aggregations need to be reviewed
    argparser.add_argument('--mia_data_mode', type=str, default='mean',help='The aggregation mode in the MIA datasets',choices=['mean'])
    argparser.add_argument('--mia_batch_size', type=int, default=1024, help='The batch size of mia model')
    argparser.add_argument('--mia_epochs', type=int, default=300, help='The training epochs of mia model')
    argparser.add_argument('--mia_fea_start', type=int, default=64, help='The starting position of the feature used in MIA model')
    #argparser.add_argument('--mia_fea_end', type=int, default=0, help='The ending position of the feature used in MIA model')
    argparser.add_argument('--mia_lr', type=float, default=0.001, help='The learning rate of MIA model')
    argparser.add_argument('--mia_val_check', type=int, default=1, help='Whether to use val datasets in the training process of MIA model', choices=[1,0])

    argparser.add_argument('--steal_epochs', type=int, default=300, help='The training epochs of steal model') #300
    argparser.add_argument('--steal_val_check', type=int, default=1, help='Whether to use val datasets in the training process of Stealing model', choices=[1,0])
    argparser.add_argument('--steal_lr', type=float, default=0.001, help='The learning rate of stealing model')
    argparser.add_argument('--save_steal_model', type=int, default=1, help='Whether to save the steal_model', choices=[1,0])
    argparser.add_argument('--steal_mode1', type=str, default='hinge', help='', choices=['hinge', 'bpr', 'no_rank'])
    argparser.add_argument('--steal_mode2', type=str, default='hinge', help='', choices=['hinge', 'bpr', 'info_nce'])
    
    argparser.add_argument('--gen_data_mode', type=str, default='random',help='The mode to generate artificial datasets', choices=['random', 'top1', 'topk'])
    argparser.add_argument('--save_gen_model', type=int, default=1, help='Whether to save the gen_steal_model', choices=[1,0])
    
    argparser.add_argument('--train_seq', type=int, default=1, help='Whether to train seq model', choices=[1,0])
    argparser.add_argument('--train_steal1', type=int, default=1, help='Whether to train steal model1', choices=[1,0])
    argparser.add_argument('--train_steal2', type=int, default=1, help='Whether to train steal model2', choices=[1,0])
    argparser.add_argument('--train_steal3', type=int, default=1, help='Whether to train steal model3', choices=[1,0])
    argparser.add_argument('--train_gen', type=int, default=1, help='Whether to train artificial model', choices=[1,0])

    argparser.add_argument('--sample_rate', type=float, default=1, help='The sample rate to implementing data enhancement')
    # argparser.add_argument('--data_aug_mode', type=str, default='adv2', help='The method to implementing data enhancement', choices=['noise', 'adv', 'adv2', 'adv3', 'random'])
    # argparser.add_argument('--data_aug_num', type=int, default=2, help='The num of data enhancement adv3')
    argparser.add_argument('--data_aug_mode', type=str, default='real', help='The method to implementing data enhancement', choices=['real', 'fake'])
    argparser.add_argument('--data_aug_loss_fn', type=str, default='Steal', help='None')
    argparser.add_argument('--data_aug_sort', type=str, default='max', help='None', choices=['max', 'min', 'random'])
    
    argparser.add_argument('--gpu', type=int, default=-1, help='The used gpu id')
    argparser.add_argument('--file', type=str, default='a', help='None')
    
    torch.set_num_threads(10)



    args = argparser.parse_args()

    dataset = args.dataset
    target_model_name = args.target_model_name
    shadow_model_name = args.shadow_model_name
    if args.know_model_name == 1:
        shadow_model_name = target_model_name
    cutoff = args.cutoff


    seq_epochs = args.seq_epochs

    max_seq_len = args.max_seq_len
    seq_val_check = True if args.seq_val_check==1 else False
    seq_early_stop_patience = args.seq_early_stop_patience

    classifier_avg_mode = args.classifier_avg_mode
    mia_data_mode = args.mia_data_mode
    mia_batch_size = args.mia_batch_size
    mia_epochs = args.mia_epochs
    mia_fea_start = args.mia_fea_start
    num_features = cutoff + 3
    mia_fea_end = num_features + mia_fea_start
    mia_lr = args.mia_lr
    mia_val_check = True if args.mia_val_check==1 else False

    steal_epochs = args.steal_epochs
    steal_val_check = True if args.steal_val_check==1 else False
    steal_lr = args.steal_lr
    mode1 = args.steal_mode1 
    mode2 = args.steal_mode2

    gen_data_mode = args.gen_data_mode

    train_seq = True if args.train_seq==1 else False
    train_steal1 = True if args.train_steal1==1 else False
    train_steal2 = True if args.train_steal2==1 else False
    train_steal3 = True if args.train_steal3==1 else False
    train_gen = True if args.train_gen==1 else False

    if train_seq == True:
        train_steal1 = True
        train_steal2 = True
        train_steal3 = True
        train_gen = True
    
    sample_rate = args.sample_rate
    data_aug_mode = args.data_aug_mode
    data_aug_loss_fn = args.data_aug_loss_fn if args.data_aug_loss_fn != 'Steal' else None
    data_aug_sort = args.data_aug_sort
    gpu = args.gpu if args.gpu >= 0 else None

    shadow_dataset = 'shadow_' + dataset
    target_dataset = 'target_' + dataset
    member_shadow_dataset = 'member_' + shadow_dataset
    nonmember_shadow_dataset = 'nonmember_' + shadow_dataset
    member_target_dataset = 'member_' + target_dataset
    nonmember_target_dataset = 'nonmember_' + target_dataset

    target_seq_model_path = target_model_name + '_' + target_dataset + '_' + str(seq_epochs) + 'epochs_val_check_' + str(seq_val_check) + '_'  + data_aug_mode + data_aug_sort
    shadow_seq_model_path = shadow_model_name + '_' + shadow_dataset + '_' + str(seq_epochs) + 'epochs_val_check_' + str(seq_val_check) + '_'  + data_aug_mode + data_aug_sort
    
    steal_model_path1 = target_model_name + '_' + shadow_model_name + '_' + dataset + '_seq_' + str(seq_epochs) + 'epochs_val_check_' + str(seq_val_check) \
    + '_steal_' + str(steal_epochs) + 'epochs_val_check_' + str(steal_val_check) + '_steal1_' + data_aug_mode + data_aug_sort + mode1 + mode2

    steal_model_path2 = target_model_name + '_' + shadow_model_name + '_' + dataset + '_seq_' + str(seq_epochs) + 'epochs_val_check_' + str(seq_val_check) \
    + '_steal_' + str(steal_epochs) + 'epochs_val_check_' + str(steal_val_check) + '_steal2_' + data_aug_mode + data_aug_sort + mode1 + mode2

    steal_model_path3 = target_model_name + '_' + shadow_model_name + '_' + dataset + '_seq_' + str(seq_epochs) + 'epochs_val_check_' + str(seq_val_check) \
    + '_steal_' + str(steal_epochs) + 'epochs_val_check_' + str(steal_val_check) + '_steal3_' + data_aug_mode + data_aug_sort + mode1 + mode2

    gen_model_path = target_model_name + '_' + shadow_model_name + '_' + dataset + '_seq_' + str(seq_epochs) + 'epochs_val_check_' + str(seq_val_check) \
    + '_steal_' + str(steal_epochs) + 'epochs_val_check_' + str(steal_val_check) + '_gen_' + data_aug_mode + data_aug_sort + mode1 + mode2

    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_file = os.path.join('./mia_log', str(now_time) + '.log')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', \
    filename=log_file, filemode='w')
    logging.info(vars(args))
    
    results = []
    methods = []

    if not train_seq:
        target_model, member_target_datasets = init(target_model_name, member_target_dataset, epochs=1, max_seq_len=max_seq_len, val_check=seq_val_check, early_stop_patience=seq_early_stop_patience, cutoff = cutoff, gpu=gpu)
        shadow_model, member_shadow_datasets = init(shadow_model_name, member_shadow_dataset, epochs=1, max_seq_len=max_seq_len, val_check=seq_val_check, early_stop_patience=seq_early_stop_patience, cutoff = cutoff, gpu=gpu)
        target_model.load_checkpoint(os.path.join('/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved', target_seq_model_path))
        shadow_model.load_checkpoint(os.path.join('/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved', shadow_seq_model_path))
    else:
        target_model, member_target_datasets = init(target_model_name, member_target_dataset, epochs=seq_epochs, max_seq_len=max_seq_len, val_check=seq_val_check, early_stop_patience=seq_early_stop_patience, _best_ckpt_path=target_seq_model_path, cutoff = cutoff, gpu=gpu)
        shadow_model, member_shadow_datasets = init(shadow_model_name, member_shadow_dataset, epochs=seq_epochs, max_seq_len=max_seq_len, val_check=seq_val_check, early_stop_patience=seq_early_stop_patience, _best_ckpt_path=shadow_seq_model_path, cutoff = cutoff, gpu=gpu)
    
    _, nonmember_shadow_datasets = init(shadow_model_name, nonmember_shadow_dataset, epochs=0, gpu=gpu)
    _, nonmember_target_datasets = init(shadow_model_name, nonmember_target_dataset, epochs=0, gpu=gpu)
    _, shadow_datasets = init(shadow_model_name, shadow_dataset, epochs=0, gpu=gpu)
    _, target_datasets = init(shadow_model_name, target_dataset, epochs=0, gpu=gpu)

    a, _, _ = eva(target_model, member_target_datasets)
    b, _, _ = eva(target_model, nonmember_target_datasets)

    logging.info('The performance of target model in the member datasets')
    logging.info(a)
    logging.info('The performance of target model in the nonmember datasets')
    logging.info(b)

    results.append(round((a['recall@10'].item()-b['recall@10'].item())/2 + 0.5, 5))
    methods.append('baseline')

    eva(shadow_model, member_shadow_datasets)

    dscore_target_member = score(target_model, member_target_datasets)
    dscore_target_nonmember = score(target_model, nonmember_target_datasets)
    dscore_shadow_member = score(shadow_model, member_shadow_datasets)
    dscore_shadow_nonmember = score(shadow_model, nonmember_shadow_datasets)

    add_topk(shadow_datasets[0], target_model)
    add_topk(shadow_datasets[1], target_model)
    add_topk(shadow_datasets[2], target_model)

    add_topk(target_datasets[0], target_model)
    add_topk(target_datasets[1], target_model)
    add_topk(target_datasets[2], target_model)

    add_topk(member_target_datasets[0], target_model)
    add_topk(member_target_datasets[1], target_model)
    add_topk(member_target_datasets[2], target_model)

    add_topk(nonmember_target_datasets[0], target_model)
    add_topk(nonmember_target_datasets[1], target_model)
    add_topk(nonmember_target_datasets[2], target_model)


    train_val_datasets = MIADataset(dscore_shadow_member[0], dscore_shadow_nonmember[0], mia_data_mode=mia_data_mode)
    train_datasets, val_datasets = train_val_datasets.build()
    train_loader = train_datasets.loader(batch_size=mia_batch_size, num_workers=0, shuffle=True, drop_last=False)
    val_loader = val_datasets.loader(batch_size=mia_batch_size, num_workers=0, shuffle=False, drop_last=False)
    mia_model = MIAModel(num_fea=mia_fea_end - mia_fea_start, hiddens=mia_hidden_size, avg_mode = classifier_avg_mode)
    mia_model.fit(train_loader, val_loader, epochs=mia_epochs, lr=mia_lr, val_check=mia_val_check, start=mia_fea_start, end=mia_fea_end, optim='Adam')

    # White + Classifier based
    test_datasets = MIADataset(dscore_target_member[0],dscore_target_nonmember[0],mia_data_mode=mia_data_mode)
    test_loader = test_datasets.loader(batch_size=mia_batch_size, num_workers=0, shuffle=False, drop_last=False)
    classifier_white_acc = mia_model.evaluate(test_loader)

    logging.info('White + Classifier based:{:.2%}'.format(classifier_white_acc))
    results.append(round(classifier_white_acc, 5))
    methods.append('white')

    # White + Threshold based
    white_results, thd_white_acc = thd_mia(dscore_shadow_member[0], dscore_shadow_nonmember[0], dscore_target_member[0], dscore_target_nonmember[0], metrics=['pos_score'], select_num=5)
    logging.info('White + Threshold based:{:.2%}'.format(thd_white_acc))


    #Without model stealing
    logging.info('Without model stealing')
    wost_member_score = score(shadow_model, member_target_datasets)
    wost_nonmember_score = score(shadow_model, nonmember_target_datasets)
    wost_test_datasets = MIADataset(wost_member_score[0], wost_nonmember_score[0], mia_data_mode=mia_data_mode)
    wost_test_loader = wost_test_datasets.loader(batch_size=mia_batch_size, num_workers=0, shuffle=False, drop_last=False)
    classifier_black_wost_acc = mia_model.evaluate(wost_test_loader)

    logging.info('Without model stealing + Black + Classifier based:{:.2%}'.format(classifier_black_wost_acc))

    results.append(round(classifier_black_wost_acc, 5))
    methods.append('wo-me')

    wost_black_results, thd_black_wost_acc = thd_mia(dscore_shadow_member[0], dscore_shadow_nonmember[0], wost_member_score[0], wost_nonmember_score[0], metrics=['pos_score'], select_num=5)

    logging.info('Without model stealing + Black + Threshold based:{:.2%}'.format(thd_black_wost_acc))

    # Black
    if not train_steal1:
        steal_model1 = model_stealing(shadow_model_name, shadow_datasets[0], shadow_datasets[1], epochs=1, lr=steal_lr, cutoff=cutoff, gpu=gpu, mode1=mode1, mode2=mode2)
        steal_model1.load_checkpoint(os.path.join('/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved', steal_model_path1))
    else:
        steal_model1 = model_stealing(shadow_model_name, shadow_datasets[0], shadow_datasets[1], epochs=steal_epochs, lr=steal_lr, _best_ckpt_path=steal_model_path1, cutoff=cutoff, gpu=gpu, mode1=mode1, mode2=mode2)
    eva(steal_model1, member_target_datasets)

    steal_member_score = score(steal_model1, member_target_datasets)
    steal_nonmember_score = score(steal_model1, nonmember_target_datasets)
    steal_test_datasets = MIADataset(steal_member_score[0], steal_nonmember_score[0], mia_data_mode=mia_data_mode)
    steal_test_loader = steal_test_datasets.loader(batch_size=mia_batch_size, num_workers=0, shuffle=False, drop_last=False)
    classifier_black_acc = mia_model.evaluate(steal_test_loader)
    logging.info('Black + Classifier based:{:.2%}'.format(classifier_black_acc))

    black_results, thd_black_acc = thd_mia(dscore_shadow_member[0], dscore_shadow_nonmember[0], steal_member_score[0], steal_nonmember_score[0], metrics=['pos_score'], select_num=5)
    logging.info('Black + Threshold based:{:.2%}'.format(thd_black_acc))
    results.append(round(classifier_black_acc, 5))
    methods.append('me')
    del steal_model1

    # Fine-tune
    if not train_steal2:
        steal_model2 = model_stealing(shadow_model_name, shadow_datasets[0], shadow_datasets[1], epochs=1, lr=steal_lr, cutoff=cutoff, gpu=gpu, mode1=mode1, mode2=mode2)
        steal_model2.load_checkpoint(os.path.join('/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved', steal_model_path2))
    else:
        steal_model2 = model_stealing(shadow_model_name, shadow_datasets[0], shadow_datasets[1], epochs=1, lr=steal_lr, cutoff=cutoff, _best_ckpt_path=steal_model_path2, gpu=gpu, mode1=mode1, mode2=mode2)
        steal_model2.load_checkpoint(os.path.join('/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved', steal_model_path1))

        new_datasets = NewSeqDataset(name=shadow_dataset, config={'max_seq_len': max_seq_len}).build()[0]
        uids, adv_scores = get_adv_scores(steal_model2, shadow_datasets[0])
        item_sim = item_sim_func(steal_model2)
        select_in_item_id, select_item_id = gen_aug_datasets(steal_model2, adv_scores, item_sim, shadow_datasets[0], sample_num=len(shadow_datasets[0]), adv_num=5, mode=data_aug_mode, sort=data_aug_sort)
        new_datasets.in_item_id = select_in_item_id
        new_datasets.item_id = select_item_id
        data_index = np.zeros((len(select_item_id),3))
        for i in range(len(select_item_id)):
            data_index[i] = np.array([i, 0, max_seq_len])
        new_datasets.data_index = torch.tensor(data_index)
        add_topk(new_datasets, target_model)

    # Black + aug
    if not train_steal3:
        steal_model3 = model_stealing(shadow_model_name, shadow_datasets[0], shadow_datasets[1], epochs=1, lr=steal_lr, cutoff=cutoff, gpu=gpu, mode1=mode1, mode2=mode2)
        steal_model3.load_checkpoint(os.path.join('/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved', steal_model_path3))
    else:
        steal_model3 = model_stealing(shadow_model_name, new_datasets, shadow_datasets[1], epochs=steal_epochs, lr=steal_lr, _best_ckpt_path=steal_model_path3, cutoff=cutoff, gpu=gpu, mode1=mode1, mode2=mode2)

    eva(steal_model3, member_target_datasets)

    steal_member_score = score(steal_model3, member_target_datasets)
    steal_nonmember_score = score(steal_model3, nonmember_target_datasets)
    steal_test_datasets = MIADataset(steal_member_score[0], steal_nonmember_score[0], mia_data_mode=mia_data_mode)
    steal_test_loader = steal_test_datasets.loader(batch_size=mia_batch_size, num_workers=0, shuffle=False, drop_last=False)
    classifier_black_acc = mia_model.evaluate(steal_test_loader)
    logging.info('Black + Aug + Classifier based:{:.2%}'.format(classifier_black_acc))

    results.append(round(classifier_black_acc, 5))
    methods.append('me+aug')

    black_results, thd_black_acc = thd_mia(dscore_shadow_member[0], dscore_shadow_nonmember[0], steal_member_score[0], steal_nonmember_score[0], metrics=['pos_score'], select_num=5)
    logging.info('Black + Aug + Threshold based:{:.2%}'.format(thd_black_acc))

    gen_dataset = target_model_name + '_' + dataset + '_seq_' + str(seq_epochs) + 'epochs_val_check_' + str(seq_val_check) + \
        '_' + str(shadow_datasets[0].num_users) + 'users_' + str(int(len(shadow_datasets[0]) / shadow_datasets[0].num_users)) + 'len'

    # Artificial datasets
    # if not train_gen:
    #     pass
    # else:
    #     gen_data(gen_dataset, target_model, gen_len=int(len(shadow_datasets[0]) / shadow_datasets[0].num_users), num_users=shadow_datasets[0].num_users, oral_dataset=dataset, mode=gen_data_mode)
    
    # _, gen_datasets = init(shadow_model_name, gen_dataset, epochs=0, gpu=gpu)

    # add_topk(gen_datasets[0], target_model)
    # add_topk(gen_datasets[1], target_model)
    # add_topk(gen_datasets[2], target_model)

    # if not train_gen:
    #     gen_model = model_stealing(shadow_model_name, gen_datasets[0], gen_datasets[1], epochs=1, lr=steal_lr, cutoff=cutoff, gpu=gpu, mode1=mode1, mode2=mode2)
    #     gen_model.load_checkpoint(os.path.join('/data1/home/zhihao/code/miars/seqrec/RecStudio-main/mia_saved', gen_model_path))
    # else:
    #     gen_model = model_stealing(shadow_model_name, gen_datasets[0], gen_datasets[1], epochs=steal_epochs, lr=steal_lr, _best_ckpt_path=gen_model_path, cutoff=cutoff, gpu=gpu, mode1=mode1, mode2=mode2)
    
    # eva(gen_model, member_target_datasets)
    # gen_member_score = score(gen_model, member_target_datasets)
    # gen_nonmember_score = score(gen_model, nonmember_target_datasets)
    # gen_test_datasets = MIADataset(gen_member_score[0], gen_nonmember_score[0], mia_data_mode=mia_data_mode)
    # gen_test_loader = gen_test_datasets.loader(batch_size=mia_batch_size, num_workers=0, shuffle=False, drop_last=False)
    # classifier_black_gen_acc = mia_model.evaluate(gen_test_loader)
    # logging.info('Gen_datasets + Black + Classifier based:{:.2%}'.format(classifier_black_gen_acc))

    # results.append(round(classifier_black_gen_acc, 5))
    # methods.append('me+gen_data')

    # gen_black_results, thd_black_gen_acc = thd_mia(dscore_shadow_member[0], dscore_shadow_nonmember[0], gen_member_score[0], gen_nonmember_score[0], metrics=['pos_score'], select_num=5)
    # logging.info('Gen_datasets + Black + Threshold based:{:.2%}'.format(thd_black_gen_acc))


    path = './main_results.txt'

    with open(path, 'a') as f:
        f.write(str(vars(args)) + '\n')
        f.write('dataset:' + dataset + '\n')
        f.write('target_model:' + target_model_name + '\n')
        f.write('shadow_model:' + shadow_model_name + '\n')
        f.write('seq_epochs:' + str(seq_epochs) + '\n')
        f.write('data_aug_mode:' + str(data_aug_mode) + '\n')
        f.write('data_aug_sort:' + str(data_aug_sort) + '\n')
        f.write('steal_mode1:' + str(mode1) + '\n')
        f.write('steal_mode2:' + str(mode2) + '\n')
        f.write(str(methods) + '\n')
        f.write(str(results) + '\n\n')






        

 