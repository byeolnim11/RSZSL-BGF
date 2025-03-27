#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-10-31 14:29:42
LastEditTime: 2021-12-18 10:24:42
LastEditors: Kitiro
Description: 
FilePath: /exp/web_zsl/extract_feature.py
'''

import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

data_dir = 'dataset'

# 这里用的是img net的std和mean
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
# 如果 size 只是一个 int ，那就会在保持长宽比的情况下，把短边缩放到指定 size ，长边按照长宽比缩放
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


def get_torchvision_model(args, num_classes, resume = None):
    if args.imagenet_pretrained:
        print('use image net pretrained')
        model = eval(f'models.{args.backbone}')(pretrained = True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        print('learn from scratch')
        model = eval(f'models.{args.backbone}')(num_classes = num_classes, pretrained = False)

    if resume is not None:
        saved = torch.load(resume)['state_dict']

        model = eval(f'models.{args.backbone}')(pretrained = False)
        model.fc = nn.Linear(model.fc.in_features, 972)
        model = nn.DataParallel(model)
        model.load_state_dict(saved)
        model = model.module
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


class Net(nn.Module):
    def __init__(self, args, class_num):
        super(Net, self).__init__()
        self.backbone = get_torchvision_model(args, class_num)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        x = self.backbone.fc(x)
        return x, feat


def conver_img(img_path, extract = False):
    img = Image.open(img_path)
    img = transform(img)
    if img.shape[0] == 1:
        img = torch.Tensor(np.tile(img, (3, 1, 1)))
    elif img.shape[0] == 4:
        img = img[:3, :, :]
    img = normalize(img)
    if extract:
        # 将img第一维unsqueeze，传入模型
        img = Variable(torch.unsqueeze(img, dim = 0).float())
    return img


class DataSet(Dataset):
    def __init__(self, img_list, label_list):
        self.img_list = img_list
        self.label_list = label_list
        assert len(self.img_list) == len(self.label_list)

    # for training
    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = conver_img(img_path)
        return img, label

    def __len__(self):
        return len(self.label_list)


def get_model(args):
    # if args.model == 'res50':
    #     model = models.resnet50(pretrained = True)
    # elif args.model == 'res101':
    #     model = models.resnet101(pretrained = True)
    # elif args.model == 'res152':
    #     model = models.resnet152(pretrained = True)
    # elif args.model == 'finetuned':
    #     # print('using finetuned model')
    model = torch.load(args.model_path).module
    try:
        del model.backbone.fc
        model.backbone.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1, padding = 0)
        model.backbone.fc = lambda x: x
    except:
        del model.fc
        model.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1, padding = 0)
        model.fc = lambda x: x
    print('extract feature with pretrained %s' % args.model_path)
    model.eval()
    print(model)
    return model


# load data 
def get_search_list(args):
    if args.ds == 'AWA2':
        class_dir = os.path.join(data_dir, args.ds, 'Animals_with_Attributes2')
        image_dir = os.path.join(class_dir, 'JPEGImages')
    elif args.ds == 'CUB':
        class_dir = os.path.join(data_dir, args.ds, 'CUB_200_2011')
        image_dir = os.path.join(class_dir, 'images')
    elif args.ds == 'APY':
        class_dir = os.path.join(data_dir, args.ds, 'origin')
        image_dir = class_dir
    elif args.ds == 'SUN':
        class_dir = os.path.join(data_dir, args.ds, 'origin')
        image_dir = class_dir
    elif args.ds == 'FLO':
        class_dir = os.path.join(data_dir, args.ds, 'origin')
        image_dir = class_dir
    else:
        raise ValueError('the specified dataset is not supported')

    with open(os.path.join(class_dir, 'classes.txt')) as f:
        category2label_ = [i.split() for i in f.readlines()]
    class_list = [j for _, j in category2label_]
    labels = [int(i) for i, _ in category2label_]
    print('DataSet:', args.ds, 'Contains', str(len(class_list)) + ' classes.')
    return class_list, labels


# logits:tensor. label:numpy
def get_pred(logits):
    probs = F.softmax(logits, dim = 1).detach().cpu().numpy()
    pred = np.argmax(probs, axis = 1)
    return pred


def extract_web(args):
    web_data_dir = os.path.join(data_dir, args.ds, f'{args.ds}_web')
    web_feat_dir = os.path.join(data_dir, args.ds, f'mat')
    if not os.path.exists(web_feat_dir):
        os.mkdir(web_feat_dir)
    batch_size = 128
    feats, labels = [], []
    class_list, label_list = get_search_list(args)
    model = get_model(args).cuda()
    batch_feats = []
    # 每次处理一类
    for cla, label in zip(class_list, label_list):
        print(label, '/', len(label_list), cla)
        img_dir = os.path.join(web_data_dir, cla)
        if not os.path.exists(img_dir):
            continue
        else:
            img_files = os.listdir(img_dir)
            # img_files.sort(key = lambda x: int(x[:-4]))
            # 按index排序!!!
            img_files.sort()
            img_files = [os.path.join(img_dir, file) for file in img_files]
            cnt = 0
            for i in range(len(img_files))[::batch_size]:
                batch_img_files = img_files[i:i + batch_size]
                batch_imgs = []
                for img_path in batch_img_files:
                    try:
                        img = conver_img(img_path = img_path, extract = True).cuda()
                    except:
                        print('failed!!!', img_path)
                    batch_imgs.append(img)
                    cnt += 1
                batch_feat = model(
                    torch.cat(batch_imgs) if len(batch_imgs) > 1 else batch_imgs[0]).detach().cpu().numpy().squeeze()
                batch_feats.append(np.reshape(batch_feat, [-1, 2048]))
            label_class = [label] * cnt
            labels.extend(label_class)
            print('cla img num:', cnt)

    feats = np.concatenate(batch_feats)
    print(args.ds)
    print('feature shape:', feats.shape)
    mat_path = f'{args.ds}_web_features.mat'
    sio.savemat(os.path.join(web_feat_dir, mat_path), {
        'features': feats,
        'labels': np.array(labels)
    })
    print('save features on %s' % os.path.join(web_feat_dir, mat_path))


# 'ft_res101_feature_AWA2.mat'

def extract_source(args):
    # mat = sio.loadmat(os.path.join(class_dir, 'res101.mat'))
    # img_files = mat['image_files'].squeeze()
    # labels = mat['labels'].squeeze() - 1
    origin_labels = np.array([])
    if args.ds == 'AWA2':
        class_dir = os.path.join(data_dir, args.ds, 'Animals_with_Attributes2')
        image_dir = os.path.join(class_dir, 'JPEGImages')
    elif args.ds == 'CUB':
        class_dir = os.path.join(data_dir, args.ds, 'CUB_200_2011')
        image_dir = os.path.join(class_dir, 'images')
    elif args.ds == 'APY':
        class_dir = os.path.join(data_dir, args.ds, 'origin')
        image_dir = class_dir
        att_splits = sio.loadmat(os.path.join(class_dir, 'res101.mat'))
        img_names = [i[0].split('APY/')[-1] for i in att_splits['image_files'].squeeze()]
        origin_labels = att_splits['labels'].squeeze()
        assert len(img_names) == len(origin_labels)
    elif args.ds == 'SUN':
        class_dir = os.path.join(data_dir, args.ds, 'origin')
        image_dir = class_dir
        att_splits = sio.loadmat(os.path.join(class_dir, 'res101.mat'))
        img_names = [i[0].split('SUN/')[-1] for i in att_splits['image_files'].squeeze()]
        origin_labels = att_splits['labels'].squeeze()
        assert len(img_names) == len(origin_labels)
    elif args.ds == 'FLO':
        class_dir = os.path.join(data_dir, args.ds, 'origin')
        image_dir = class_dir
        att_splits = sio.loadmat(os.path.join(class_dir, 'res101.mat'))
        img_names = [i[0].split('Flowers/')[-1] for i in att_splits['image_files'].squeeze()]
        origin_labels = att_splits['labels'].squeeze()
        assert len(img_names) == len(origin_labels)
    else:
        raise ValueError('the specified dataset is not supported')

    images = np.loadtxt(os.path.join(class_dir, 'images.txt'), dtype = str, delimiter = ' ')
    img_files_relative = images[:, 1]
    img_files = [os.path.join(image_dir, i) for i in img_files_relative]
    # category2label_ = np.loadtxt(os.path.join(class_dir, 'classes.txt'), dtype = str, delimiter = ' ')
    with open(os.path.join(class_dir, 'classes.txt')) as f:
        category2label_ = [i.split() for i in f.readlines()]
    category2label = {j: int(i) for i, j in category2label_}
    model = get_model(args).cuda()
    feats = []
    labels = []

    for img_path in tqdm(img_files):
        img = conver_img(img_path = img_path, extract = True).cuda()
        feat = model(img)[0].detach().cpu().numpy().squeeze()
        feats.append(feat)
        labels.append(category2label[img_path.split('/')[-2]] if not origin_labels.any() else None)
    feats = np.stack(feats)
    labels = np.stack(labels) if not origin_labels.any() else None
    mat_r_path = f'{args.ds}_source_features.mat'
    mat_dir = os.path.join(data_dir, args.ds, 'mat')
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    sio.savemat(os.path.join(mat_dir, mat_r_path), {
        'features': feats,
        'labels': np.array(labels) if not origin_labels.any() else origin_labels
    })
    print('save features on %s' % os.path.join(mat_dir, mat_r_path))


def parse_arg():
    parser = argparse.ArgumentParser(description = 'word embeddign type')
    parser.add_argument('--ds', type = str, default = 'CUB',
                        help = 'dataset: [AWA2, CUB, APY, SUN]')
    parser.add_argument('--class_num', type = int, default = 200)
    parser.add_argument("--is_source", type = int, default = 0, help = "script options")
    # saves/naive_NoisyNovel_SUN_lr0.0001_b64_wd0.0001_04180137/naive_NoisyNovel_SUNF_lr0.0001_b64_wd0.0001_04180137_best.pth
    # saves/naive_NoisyNovel_FLO_lr0.0001_b64_wd0.0001_04180102/naive_NoisyNovel_FLO_lr0.0001_b64_wd0.0001_04180102_best.pth

    # python extract_feature.py --ds SUN --class_num 717 --is_source 0 --model_path saves/naive_NoisyNovel_SUN_lr0.0001_b64_wd0.0001_04281612/naive_NoisyNovel_SUN_lr0.0001_b64_wd0.0001_04281612_best.pth
    # python extract_feature.py --ds FLO --class_num 102 --is_source 0 --model_path saves/naive_NoisyNovel_FLO_lr0.0001_b64_wd0.0001_04180102/naive_NoisyNovel_FLO_lr0.0001_b64_wd0.0001_04180102_best.pth
    parser.add_argument('--model_path', type = str, default = 'pretrained/CUB/step3/final_classifier.pth')
    parser.add_argument('--backbone', default = 'resnet50', type = str)
    # parser.add_argument("--ft", action="store_true", default=False, help="whethre extract with fine-tuned model")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    # options: [extra_web, extra_source]
    if args.is_source == 0:
        extract_web(args)
    elif args.is_source == 1:
        extract_source(args)
    else:
        raise NotImplementedError("not implemented other options")
