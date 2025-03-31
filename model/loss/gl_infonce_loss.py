import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_label_smooth import CrossEntropyLabelSmooth
from .sigmoid_focal_loss import SigmoidFocalLoss

# https://zhuanlan.zhihu.com/p/346686467
# https://arxiv.org/pdf/2004.11362.pdf
class GlInfoNCELoss(nn.Module):
    def __init__(self, temperature, pos_hard_mining=False, neg_hard_mining=False, weakshot=True, random_seed=-1, print_info=False):
        super(GlInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.pos_hard_mining = pos_hard_mining
        self.neg_hard_mining = neg_hard_mining
        self.weakshot = weakshot
        self.random_seed = random_seed
        self.print_info = print_info

    def forward(self, feature, label, label_type=None, label_origin=None):
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
            self.print_pair_acc(feature.detach(), label.detach(), label_type.detach(), label_origin.detach())
        #############################################################

#############################################################
        # modified from part_triplet_loss
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(2) == label.unsqueeze(1)).bool()
        if self.weakshot:
            hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).int()
            true_hn_mask1 = (label_type.unsqueeze(2).repeat(1, 1, m) > 0).int() # anchor-negative of clean label as anchor
            true_hn_mask2 = ((label_type.unsqueeze(2) - label_type.unsqueeze(1)) < 0).int() # anchor-negative of noise label as anchor and clean label as negative
            true_hn_mask = ((hn_mask * true_hn_mask1 + hn_mask * true_hn_mask2) > 0).bool()
        else:
            true_hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).bool()

        ssim = 1 - self.batch_cos_dist(feature)
        full_loss_metric_list = []
        for i in range(m):
            #############################################################
            # pos_cnt = (hp_mask[:, i, :] != 0).sum(-1)
            # neg_cnt = (true_hn_mask[:, i, :] != 0).sum(-1)
            # print("index={}, pos_cnt={}, neg_cnt={}".format(i, pos_cnt, neg_cnt))
            #############################################################
            full_hp_ssim = torch.masked_select(ssim[:, i, :], hp_mask[:, i, :]).view(n, -1)
            full_hn_ssim = torch.masked_select(ssim[:, i, :], true_hn_mask[:, i, :]).view(n, -1)
            if self.pos_hard_mining:
                full_hp_ssim = torch.min(full_hp_ssim, dim=1, keepdim=True)[0]
            if self.neg_hard_mining:
                full_hn_ssim = torch.max(full_hn_ssim, dim=1, keepdim=True)[0]
            full_hp_ssim_exp = (full_hp_ssim * self.temperature).exp()
            full_hn_ssim_exp = (full_hn_ssim * self.temperature).exp()
            full_hn_ssim_expsum = torch.sum(full_hn_ssim_exp, dim=1, keepdim=True)
            full_loss_metric = -torch.log(full_hp_ssim_exp/(full_hp_ssim_exp+full_hn_ssim_expsum))
            full_loss_metric_list.append(full_loss_metric.mean(1, keepdim=True))
        full_loss_metric_list = torch.cat(full_loss_metric_list, dim=1) # cat along batch dimension
        full_loss_metric_mean = full_loss_metric_list.mean(dim=1) # mean along batch dimension

        return full_loss_metric_mean.mean(), ssim.squeeze(0)

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

    # def print_pair_acc(self, feature, label, label_type, total_id=110, clean_id=36):
    def print_pair_acc(self, feature, label, label_type, label_origin):
        all_pair_sim = 1 - self.batch_cos_dist(feature).view(-1)
        all_pair_label = (label.unsqueeze(2) == label.unsqueeze(1)).int().view(-1)

        # known
        true_mask = self.weakshot_mask(label, label_type)
        pair_sim, pair_label = all_pair_sim[true_mask], all_pair_label[true_mask]
        self.print_pos_neg_acc(pair_sim, pair_label, prefix='Known')

        '''
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
        '''
        label_origin = label_origin.unsqueeze(0)
        clean_all_pair_label = (label_origin.unsqueeze(2) == label_origin.unsqueeze(1)).int().view(-1)

        # unknown mask
        pair_sim, pair_label = all_pair_sim[~true_mask], clean_all_pair_label[~true_mask]
        self.print_pos_neg_acc(pair_sim, pair_label, prefix='Unknown')
#############################################################        