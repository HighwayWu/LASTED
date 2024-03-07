import os
import cv2
import time
import shutil
import random
import datetime
import argparse
import numpy as np
import logging as logger

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
import importlib
from torch_kmeans import KMeans as torchKMeans
from torch_kmeans.utils.distances import CosineSimilarity


logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, data_size=512, val_ratio=None):
        self.data_root = data_root
        self.data_size = data_size
        self.train_list = []
        self.anchor_list = []
        self.isAnchor = False
        self.isVal = False
        self.albu_pre_train = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()

        while line:
            image_path, image_label = line.rsplit(' ', 1)
            label = int(image_label)
            if random.random() < 0.1 and label == 0 and len(self.anchor_list) < 100:
                self.anchor_list.append((image_path, label))
            else:
                self.train_list.append((image_path, label))
            line = train_file_buf.readline().strip()

        if val_ratio is not None:
            np.random.shuffle(self.train_list)
            self.test_list = self.train_list[-int(len(self.train_list) * val_ratio):]
            self.train_list = self.train_list[:-int(len(self.train_list) * val_ratio)]
        else:
            self.test_list = self.train_list

    def transform(self, x):
        if self.isVal:
            x = self.albu_pre_val(image=x)['image']
        else:
            x = self.albu_pre_train(image=x)['image']
        x = self.imagenet_norm(x)  # .unsqueeze(0)
        return x

    def __len__(self):
        if self.isAnchor:
            return len(self.anchor_list)
        elif self.isVal:
            return len(self.test_list)
        else:
            return len(self.train_list)

    def __getitem__(self, index):
        if self.isAnchor:
            return self.getitem(index, self.anchor_list)
        elif self.isVal:
            return self.getitem(index, self.test_list)
        else:
            return self.getitem(index, self.train_list)

    def getitem(self, index, data_list):
        image_path, onehot_label = data_list[index]

        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)

        if image is None:
            logger.info('Error Image: %s' % image_path)
            image = np.zeros([512, 512, 3], dtype=np.uint8)
        image = image[..., ::-1]

        crop = self.transform(image)
        onehot_label = torch.LongTensor([onehot_label])
        return crop, onehot_label

    def set_val_True(self):
        self.isVal = True

    def set_val_False(self):
        self.isVal = False

    def set_anchor_True(self):
        self.isAnchor = True

    def set_anchor_False(self):
        self.isAnchor = False


def train_one_epoch(data_loader, model, optimizer, cur_epoch, loss_meter, args):
    loss_meter.reset()
    batch_idx = 0
    for (images, labels) in data_loader:
        images = images.cuda()
        labels = labels.cuda().flatten().squeeze()

        prob, features_logits = model(images)

        # image-axis loss
        loss_img = args.criterion_ce(features_logits, labels)

        # text-axis loss
        labels = labels.t()
        text_feats = features_logits.t()
        tmp_loss = []
        for tmp_class_idx in range(args.num_class):
            cur_tmp_loss = [text_feats[tmp_class_idx][labels == tmp_class_idx].mean().unsqueeze(0)]
            for cur_tmp_inner_idx in range(args.num_class):
                if cur_tmp_inner_idx == tmp_class_idx:
                    continue
                cur_tmp_loss.append(text_feats[tmp_class_idx][labels == cur_tmp_inner_idx].mean().unsqueeze(0))
            tmp_loss.append(torch.cat(cur_tmp_loss))
        loss_text = args.criterion_ce(torch.stack(tmp_loss), torch.zeros(args.num_class).long().to(labels.device))

        # total loss
        loss = (loss_img + loss_text) / 2 if not torch.isnan(loss_text).any() else loss_img

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f' % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            loss_meter.reset()
        if batch_idx > 1000:
            break
        batch_idx += 1

    return loss_avg


def validation_cluster(model, args, test_file='', data_loader=None):
    if data_loader is None:
      data_loader = DataLoader(
          ImageDataset(args.data_root, test_file, data_size=args.data_size), args.batch_size, shuffle=False,
          num_workers=min(48, args.batch_size), )
    data_loader.dataset.set_val_True()
    model.eval()
    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    label2features_dict = {}
    for (images, labels) in data_loader:
        images = images.cuda()
        b, C, H, W = images.shape
        labels = labels.flatten().squeeze().cpu().numpy()

        try:
            with torch.no_grad():
                prob, feats = model(images, isTrain=False)
                for feat, label in zip(feats, labels):
                    if label in label2features_dict.keys():
                        features = label2features_dict[label]
                        features.append(feat)
                    else:
                        features = [feat]
                    label2features_dict.update({label: features})
        except:
            continue

    cluster_result = args.torchKMeans(x=torch.stack(label2features_dict[0] + label2features_dict[1]).unsqueeze(0), k=2)
    cluster_predict = cluster_result.labels[0].cpu().detach().numpy()
    cluster_gt = np.concatenate([np.zeros(len(label2features_dict[0])), np.ones(len(label2features_dict[1]))])
    accuracy = max(accuracy_score(cluster_gt, cluster_predict), accuracy_score(cluster_gt, 1 - cluster_predict))

    keys = label2features_dict.keys()
    for _ in range(10000):
        pos = random.sample([0, 1], 1)[0]
        if pos == 1:
            feat1, feat2 = random.sample(label2features_dict[random.sample(keys, 1)[0]], 2)
        else:
            label1, label2 = random.sample(keys, 2)
            feat1 = random.sample(label2features_dict[label1], 1)[0]
            feat2 = random.sample(label2features_dict[label2], 1)[0]
        cos_sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))
        gt_labels_list.append(pos)
        prob_labels_list.append(cos_sim[0].cpu().numpy())

    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    ap = -1

    model.train()
    return auc, accuracy, ap


def validation_similarity(model, args, test_file='', data_loader=None):
    if data_loader is None:
      data_loader = DataLoader(
          ImageDataset(args.data_root, test_file, data_size=args.data_size), args.batch_size, shuffle=False,
          num_workers=min(48, args.batch_size), )
    data_loader.dataset.set_val_True()
    data_loader.dataset.set_anchor_True()
    model.eval()

    anchor_feats = []
    for (images, labels) in data_loader:
        images = images.cuda()
        b, C, H, W = images.shape
        try:
            with torch.no_grad():
                prob, feats = model(images, isTrain=False)
                anchor_feats.append(feats)
        except:
            continue
    anchor_feats = torch.mean(torch.cat(anchor_feats, dim=0), dim=0, keepdim=True)

    data_loader.dataset.set_anchor_False()
    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    label2features_dict = {}
    for (images, labels) in data_loader:
        images = images.cuda()
        b, C, H, W = images.shape
        labels = labels.flatten().squeeze().cpu().numpy()
        try:
            with torch.no_grad():
                prob, feats = model(images, isTrain=False)
                for feat, label in zip(feats, labels):
                    if label in label2features_dict.keys():
                        features = label2features_dict[label]
                        features.append(feat)
                    else:
                        features = [feat]
                    label2features_dict.update({label: features})
        except:
            continue

    keys = label2features_dict.keys()
    for key in keys:
        feats = label2features_dict[key]
        cos_sim = F.cosine_similarity(anchor_feats, torch.stack(feats))
        gt_labels_list.append(np.repeat(1 - key, len(cos_sim)))
        prob_labels_list.append(cos_sim.cpu().numpy())
    gt_labels_list = np.concatenate(gt_labels_list)
    prob_labels_list = np.concatenate(prob_labels_list)

    fpr, tpr, thres = roc_curve(gt_labels_list, prob_labels_list)
    pred_labels_list = prob_labels_list
    pred_labels_list[pred_labels_list > thres[len(thres) // 2]] = 1
    pred_labels_list[pred_labels_list <= thres[len(thres) // 2]] = 0

    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    accuracy = accuracy_score(gt_labels_list, pred_labels_list)
    ap = average_precision_score(gt_labels_list, prob_labels_list)
    model.train()
    return auc, accuracy, ap


def main(args):
    train_data_loader = DataLoader(
        ImageDataset(args.data_root, args.train_file, data_size=args.data_size, val_ratio=args.val_ratio),
        args.batch_size, shuffle=True, num_workers=min(48, args.batch_size), drop_last=True)
    if args.test_file == '':
        test_file_list = [
            # (your_test_file.txt, nickname of the dataset)
            ('annotation/Test_DreamBooth_num1052.txt', 'DreamBooth'),
            ('annotation/Test_MidjourneyV4_num1354.txt', 'MidjourneyV4'),
            ('annotation/Test_MidjourneyV5_num2000.txt', 'MidjourneyV5'),
            ('annotation/Test_NightCafe_num1300.txt', 'NightCafe'),
            ('annotation/Test_StableAI_num1290.txt', 'StableAI'),
            ('annotation/Test_YiJian_num796.txt', 'YiJian'),
        ]
    else:
        test_file_list = [
            (args.test_file, 'Test Dataset'),
        ]

    args.criterion_ce = torch.nn.CrossEntropyLoss().cuda()
    args.torchKMeans = torchKMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)

    model = getattr(importlib.import_module('model'), args.model)(num_class=args.num_class)
    model = torch.nn.DataParallel(model).cuda()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Params: %.2f' % (params / (1024 ** 2)))

    if args.resume != '':
        pretrained = torch.load(args.resume)
        model.load_state_dict(pretrained)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)
    # optimizer = optim.AdamW(parameters, lr=args.lr)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    loss_meter = AverageMeter()
    test_best = -1
    for epoch in range(args.epoches):
        model.train()
        train_data_loader.dataset.set_val_False()
        if args.isTrain == 1:
            train_one_epoch(train_data_loader, model, optimizer, epoch, loss_meter, args)

            train_data_loader.dataset.set_val_True()
            val_auc, val_acc, val_ap = validation_cluster(model, args, data_loader=train_data_loader)
            # val_auc, val_acc, val_ap = validation_similarity(model, args, data_loader=train_data_loader)
            logger.info('Score: Validation AUC: %5.4f, Acc: %5.4f, AP: %5.4f' % (val_auc, val_acc, val_ap))

        # Testing:
        test_score_list = []
        for test_file, nickname in test_file_list:
            test_auc, test_acc, test_ap = validation_cluster(model, args, test_file)
            # test_auc, test_acc, test_ap = validation_similarity(model, args, test_file)
            test_score_list.append(test_auc)
            logger.info('Score of %s: AUC: %5.4f, Acc: %5.4f, AP: %5.4f' % (nickname, test_auc, test_acc, test_ap))
        test_score = np.mean(test_score_list)
        if test_score > test_best:
            test_best = test_score
            saved_name = 'Ep%03d_%5.4f.pt' % (epoch, test_score)
            isBest = '(Best)'
        else:
            saved_name = 'latest.pt'
            isBest = ''
        logger.info('Score: Mean: %5.4f  %s' % (test_score, isBest))

        if args.isTrain == 0:
            exit()

        torch.save(model.state_dict(), os.path.join(args.out_dir, saved_name))
        lr_schedule.step(val_score)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    conf = argparse.ArgumentParser()
    conf.add_argument("--data_root", type=str, default='data/',
                      help="The root folder of training set.")
    conf.add_argument("--train_file", type=str,
                      default='annotation/Train_num398700.txt')
    conf.add_argument("--test_file", type=str,
                      default='annotation/Test_MidjourneyV5_num2000.txt')
    conf.add_argument('--val_ratio', type=float, default=0.005)
    conf.add_argument('--isTrain', type=int, default=1)
    conf.add_argument("--model", type=str, default='LASTED')
    conf.add_argument("--num_class", type=int, default=2, help='The class number of training dataset')
    conf.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    conf.add_argument("--weights", type=str, default='out_dir', help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=9999, help='The training epoches.')
    conf.add_argument('--batch_size', type=int, default=48, help='The training batch size over all gpus.')
    conf.add_argument('--data_size', type=int, default=448, help='The image size for training.')
    conf.add_argument('--gpu', type=str, default='0,1,2,3', help='The gpu')
    conf.add_argument("--resume", type=str, default='')
    args = conf.parse_args()
    os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.isTrain == 1:
        date_now = datetime.datetime.now()
        date_now = '/Log_v%02d%02d%02d%02d' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
        args.time = date_now
        args.out_dir = args.out_dir + args.time
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)

    logger.info(args)
    main(args)
