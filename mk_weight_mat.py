# -*- coding=utf-8 -*-
# @Time: 2022/11/14 16:09
# @Author: N
# @Software: PyCharm
import argparse
import os
import scipy.io as sio

# src_dir = '/data2/huangweichen/toybox/FewShot-main/dataset/CUB_200_2011/images/'
# target_dir = '/data2/huangweichen/toybox/asl/data/CUB_data/images/'
#
# for i, cla_name in enumerate(os.listdir(src_dir)):
#     new_cla_name = cla_name.split('.')[-1]
#     os.system(f'cp -r {src_dir + cla_name} {target_dir}')
#     # os.system(f'mv {target_dir + cla_name} {target_dir + new_cla_name}')
#     print(i, cla_name)
import torch
import numpy as np


def relax(x, mu = 0, max = 2):
    if mu == 0:
        return x * max
    else:
        return max / (np.exp(mu) - 1) * (torch.exp(mu * x) - 1)


def get_avg(similarity_matrix, weight_type):
    k = int(float(weight_type[3:]) * similarity_matrix.shape[1])
    return similarity_matrix.topk(k)[0].mean(1)


def get_savg(similarity_matrix, weight_type):
    k = int(float(weight_type[4:]) * len(similarity_matrix))
    return (similarity_matrix.topk(k)[0].mean(1) + similarity_matrix.transpose(1, 0).topk(k)[0].mean(1)) / 2


def get_weight_dict(args, dataset, dataname):
    weight_dict = {}
    means = []
    for cname in dataset.categories:
        path = f'{args.load_dir}/{cname}'
        saved_names = torch.load(f'{path}_name.pth')
        similarity_matrix = torch.load(f'{path}_matrix.pth')
        names = saved_names

        # 这里是avg1 意思是top 1*1000个 的平均
        # similarity_matrix shape [1000, 1000]
        if args.weight_type.startswith('avg'):
            weights = get_avg(similarity_matrix, args.weight_type)
        elif args.weight_type.startswith('savg'):
            weights = get_savg(similarity_matrix, args.weight_type)
        elif args.weight_type == 'none':
            weights = similarity_matrix.new_ones(len(similarity_matrix))
        else:
            raise NotImplementedError

        # one_mean
        if 'one_mean' in args.weight_norm:
            weights /= weights.mean()
        elif 'max' in args.weight_norm:
            weights /= weights.max()
            max, mu = [float(t) for t in args.weight_norm.split('_')[1:]]
            weights = relax(weights, mu = mu, max = max)
        else:
            raise NotImplementedError

        # names = [(dataset.root_path + '/Dog_web' + n.split('Dog_web')[1]).replace('\\', os.sep).replace('/', os.sep) for
        #          n in saved_names]
        # vis_weights(weights, names, cname)

        for name, weight in zip(names, weights):
            # web_dir = os.path.basename(dataset.root_path) + '_web'
            # p = (dataset.root_path + f'/{web_dir}' + name.split(web_dir)[1]).replace('\\', os.sep).replace('/', os.sep)
            # weight_dict[p] = weight

            weight_dict[name.replace('\\', os.sep).replace(f'..{os.sep}', '')] = weight

        means.append(similarity_matrix.mean())
    return weight_dict


parser = argparse.ArgumentParser(description = 'word embeddign type')
parser.add_argument('-dataset', type = str, default = 'CUB', help = 'dataset: [AWA2, CUB, APY, SUN]')
parser.add_argument('-weight_dir', type = str)
parser.add_argument('-n_classes', type = int)
parser.add_argument('-inter_cla', action = 'store_true', default = False)
args = parser.parse_args()
dataset_dir = f'dataset/{args.dataset}/'
if args.dataset == 'AWA2':
    sub_r_dir = 'Animals_with_Attributes2'
elif args.dataset == 'APY':
    sub_r_dir = 'origin'
elif args.dataset == 'SUN':
    sub_r_dir = 'origin'
elif args.dataset == 'FLO':
    sub_r_dir = 'origin'
else:
    raise NotImplementedError()
subset_dir = os.path.join(dataset_dir, sub_r_dir)

# todo
# 先提取特征 再加载weight
base_mat_path = os.path.join(dataset_dir, 'mat', f'{args.dataset}_source_features.mat')
web_mat_path = os.path.join(dataset_dir, 'mat', f'{args.dataset}_web_features.mat')
base_mat = sio.loadmat(base_mat_path)
web_mat = sio.loadmat(web_mat_path)
base_labels = base_mat['labels']
web_labels = web_mat['labels']-web_mat['labels'].min()

category2idx = {i[1]: int(i[0]) - 1 for i in
                np.loadtxt(os.path.join(subset_dir, 'classes.txt'), dtype = str)}
# print(category2idx)

# todo
# weight_dir = '/data2/huangweichen/toybox/SimTrans-Weak-Shot-Classification/saves/clean_pretrained_weight_baidu/'
# weight_dir = 'saves/clean_pretrained_weight_AWA2_baidu/'
weight_dir = args.weight_dir if args.weight_dir[-1] == '/' else args.weight_dir + '/'
weight_file_names = sorted([i for i in os.listdir(weight_dir) if i.find("matrix.pth") > -1])
# class2weightPath_unseen = {int(i[:3]) - 1: i for i in weight_file_names}
class2weightPath_unseen = {category2idx[i.split('_matrix')[0]]: i for i in weight_file_names}
# print(class2weightPath_unseen)

# ordered set
weights = []
weight_type = 'savg1' if not args.inter_cla else 'avg1'
cnt =0
for i in range(args.n_classes):
    if i in class2weightPath_unseen:
        weight = torch.load(weight_dir + class2weightPath_unseen[i])
        if weight_type.startswith('avg'):
            weight = get_avg(weight, weight_type)
        elif weight_type.startswith('savg'):
            weight = get_savg(weight, weight_type)
        # todo
        weight /= weight.mean()
        cnt+=weight.shape[0]
        # _l = sorted([str(i) for i in range(len(weight))])
        # for j in range(len(weight)):
        #     print(_l[j], weight[j])
        # exit()
    else:
        # seen类的web没用
        weight = torch.zeros((web_labels == i).sum())
        # continue
        # print(weight.shape)
    weights.append(weight)
print(cnt)

web_mat['weights'] = np.array(torch.cat(weights, dim = 0)).reshape((1, -1))
base_mat['weights'] = np.ones_like(base_labels)
print(web_mat.keys())
print(web_mat['features'].shape)
print(web_mat['weights'].shape)
print(web_mat['labels'].shape)
print(base_mat['weights'].shape)
assert web_mat['labels'].squeeze().shape[0] == web_mat['weights'].squeeze().shape[0]
sio.savemat(base_mat_path, base_mat)
sio.savemat(web_mat_path, web_mat)

# python mk_weight_mat.py -dataset APY -weight_dir saves/clean_pretrained_weight_APY_google -n_classes 32