import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_label_smooth import CrossEntropyLabelSmooth
from .sigmoid_focal_loss import SigmoidFocalLoss

class GlContrastLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, weakshot=True, random_seed=-1, print_info=False):
        super(GlContrastLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.weakshot = weakshot
        self.random_seed = random_seed
        self.print_info = print_info

    def forward(self, feature, label, label_type=None):
        """
        Args:
            feature: elemenet-wise features (batch_size, feat_dim)
            label: ground truth labels with shape (num_classes)
        """
        feature = feature.unsqueeze(0)
        label = label.unsqueeze(0)
        label_type = label_type.unsqueeze(0)

        #############################################################
        self.random_seed = self.random_seed + 1
        if self.print_info and self.random_seed % 100 == 0:
            self.print_pair_acc(feature.detach(), label.detach(), label_type.detach())
        #############################################################

#############################################################
        # from part_contrast_loss
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(2) == label.unsqueeze(1)).bool().view(-1)
        if self.weakshot:
            hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).int()
            true_hn_mask1 = (label_type.unsqueeze(2).repeat(1, 1, m) > 0).int() # anchor-negative of clean label as anchor
            true_hn_mask2 = ((label_type.unsqueeze(2) - label_type.unsqueeze(1)) < 0).int() # anchor-negative of noise label as anchor and clean label as negative
            true_hn_mask = ((hn_mask * true_hn_mask1 + hn_mask * true_hn_mask2) > 0).bool().view(-1)
        else:
            true_hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).bool().view(-1)

        dist = self.batch_cos_dist(feature)
        dist = dist.view(-1)
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, -1)
        full_hn_dist = torch.masked_select(dist, true_hn_mask).view(n, -1)
        # pos_cnt = (hp_mask.view(n, -1) != 0).sum(-1)
        # neg_cnt = (true_hn_mask.view(n, -1) != 0).sum(-1)
        # print("pos_cnt={}, neg_cnt={}".format(pos_cnt, neg_cnt))
        # print("pos_dist={}, neg_dist={}".format(full_hp_dist.mean(dim=-1), full_hn_dist.mean(dim=-1)))
        #############################################################
        full_hp_loss_metric = F.relu(full_hp_dist - self.pos_margin)
        hp_nonzero_num = (full_hp_loss_metric != 0).sum(1).float()
        full_hp_loss_metric_mean = full_hp_loss_metric.sum(1) / hp_nonzero_num
        full_hp_loss_metric_mean[hp_nonzero_num == 0] = 0
        #############################################################
        full_hn_loss_metric = F.relu(self.neg_margin - full_hn_dist)
        hn_nonzero_num = (full_hn_loss_metric != 0).sum(1).float()
        full_hn_loss_metric_mean = full_hn_loss_metric.sum(1) / hn_nonzero_num
        full_hn_loss_metric_mean[hn_nonzero_num == 0] = 0
        #############################################################

        # #############################################################
        # if self.random_seed % 100 == 0:
        #     print('hp_nonzero_num={}, hn_nonzero_num={}'.format(hp_nonzero_num, hn_nonzero_num))
        # #############################################################

        return full_hp_loss_metric_mean.mean(), full_hn_loss_metric_mean.mean(), hp_nonzero_num.mean(), hn_nonzero_num.mean()

    def batch_euc_dist(self, x):
        x2 = torch.sum(x ** 2, 2) # n x m
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2)) # n x m x m
        dist = torch.sqrt(F.relu(dist)) # n x m x m
        return dist
    
    def batch_cos_dist(self, x):
        x2 = F.normalize(x, p=2, dim=2) # n x m x d
        dist = 1 - torch.matmul(x2, x2.transpose(1, 2)) # n x m x m
        return dist
#############################################################

#############################################################
    def weakshot_mask(self, label, label_type):
        hp_mask = (label.unsqueeze(2) == label.unsqueeze(1)).int().view(-1)
        hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).int()
        true_hn_mask1 = (label_type.unsqueeze(2) > 0).int() # anchor-negative of clean label as anchor
        true_hn_mask2 = ((label_type.unsqueeze(2) - label_type.unsqueeze(1)) < 0).int() # anchor-negative of noise label as anchor and clean label as negative
        true_hn_mask = ((hn_mask * true_hn_mask1 + hn_mask * true_hn_mask2) > 0).int().view(-1)
        true_mask = (hp_mask + true_hn_mask).bool()
        return true_mask

    def print_pos_neg_acc(self, pair_sim, pair_label, prefix='Known'):
        print("####################################################")
        pos_gt_num = (pair_label==1).sum()
        if pos_gt_num > 0:
            for thres in np.arange(0.9, 0.0, -0.1):
                pos_pred_num = (pair_sim>thres).sum()
                pos_hit_num = ((pair_label==1) * (pair_sim>thres) > 0).sum()
                accuracy = pos_hit_num / pos_pred_num
                recall = pos_hit_num / pos_gt_num
                print('{} Positive: thres={:.2f}, pos_gt_num={}, pos_pred_num={}, pos_hit_num={}, accuracy={}, recall={}'.format( \
                        prefix, thres, pos_gt_num, pos_pred_num, pos_hit_num, accuracy, recall))
        else:
            print('{} Positive: pos_gt_num=0'.format(prefix))
        print("####################################################")
        neg_gt_num = (pair_label==0).sum()
        if neg_gt_num > 0:
            for thres in np.arange(0.9, 0.0, -0.1):
                neg_pred_num = (pair_sim<thres).sum()
                neg_hit_num = ((pair_label==0) * (pair_sim<thres) > 0).sum()
                accuracy = neg_hit_num / neg_pred_num
                recall = neg_hit_num / neg_gt_num
                print('{} Negative: thres={:.2f}, neg_gt_num={}, neg_pred_num={}, neg_hit_num={}, accuracy={}, recall={}'.format( \
                        prefix, thres, neg_gt_num, neg_pred_num, neg_hit_num, accuracy, recall))
        else:
            print('{} Negative: neg_gt_num=0'.format(prefix))
        print("####################################################")

    def print_pair_acc(self, feature, label, label_type, total_id=110, clean_id=36):
        all_pair_sim = 1 - self.batch_cos_dist(feature).view(-1)
        all_pair_label = (label.unsqueeze(2) == label.unsqueeze(1)).int().view(-1)

        # known
        true_mask = self.weakshot_mask(label, label_type)
        pair_sim, pair_label = all_pair_sim[true_mask], all_pair_label[true_mask]
        self.print_pos_neg_acc(pair_sim, pair_label, prefix='Known')

        # clean label
        label = label.squeeze(0)
        clean_label = []
        for l in label:
            if l < clean_id:
                clean_label.append(l)
            if l >= clean_id:
                if (l-clean_id) % 2 == 0:
                    clean_label.append(l)
                else:
                    clean_label.append(l-1)
        clean_label = torch.stack(clean_label)
        clean_label = clean_label.unsqueeze(0)
        clean_all_pair_label = (clean_label.unsqueeze(2) == clean_label.unsqueeze(1)).int().view(-1)

        # unknown mask
        pair_sim, pair_label = all_pair_sim[~true_mask], clean_all_pair_label[~true_mask]
        self.print_pos_neg_acc(pair_sim, pair_label, prefix='Unknown')
#############################################################        