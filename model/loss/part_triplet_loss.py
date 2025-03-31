import torch
import torch.nn as nn
import torch.nn.functional as F

class PartTripletLoss(nn.Module):
    def __init__(self, margin, hard_mining=False, nonzero=True, weakshot=False):
        super(PartTripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.nonzero = nonzero
        self.weakshot = weakshot

    def forward(self, feature, label, label_type=None):
        # feature: [n, m, d], label: [n, m]
        # print('label={}, num={}'.format(label, len(label)))
        # print('label_type={}, num={}'.format(label_type, len(label_type)))      
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(2) == label.unsqueeze(1)).bool()
        if self.weakshot:
            hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).int()
            true_hn_mask1 = (label_type.unsqueeze(2).repeat(1, 1, m) > 0).int() # anchor-negative of clean label as anchor
            true_hn_mask2 = ((label_type.unsqueeze(2) - label_type.unsqueeze(1)) < 0).int() # anchor-negative of noise label as anchor and clean label as negative
            true_hn_mask = ((hn_mask * true_hn_mask1 + hn_mask * true_hn_mask2) > 0).bool()
        else:
            true_hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).bool()

        dist = self.batch_dist(feature) # n x m x m
        full_loss_metric_list = []
        nonzero_num_list = []
        for i in range(m):
            #############################################################
            # pos_cnt = (hp_mask[:, i, :] != 0).sum(-1)
            # neg_cnt = (true_hn_mask[:, i, :] != 0).sum(-1)
            # print("index={}, pos_cnt={}, neg_cnt={}".format(i, pos_cnt, neg_cnt))
            #############################################################
            full_hp_dist = torch.masked_select(dist[:, i, :], hp_mask[:, i, :]).view(n, -1).unsqueeze(-1)
            full_hn_dist = torch.masked_select(dist[:, i, :], true_hn_mask[:, i, :]).view(n, -1).unsqueeze(-2)
            full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
            full_loss_metric_list.append(full_loss_metric.sum(1, keepdim=True))
            nonzero_num_list.append((full_loss_metric != 0).sum(1, keepdim=True).float())
        full_loss_metric_list = torch.cat(full_loss_metric_list, dim=1)
        nonzero_num_list = torch.cat(nonzero_num_list, dim=1)
        full_loss_metric = full_loss_metric_list.sum(1)
        nonzero_num = nonzero_num_list.sum(1)
        full_loss_metric_mean = full_loss_metric / nonzero_num
        full_loss_metric_mean[nonzero_num == 0] = 0

        # print("full_loss_metric={}, nonzero_num={}".format(full_loss_metric_mean, nonzero_num))
        return full_loss_metric_mean.mean(), nonzero_num.mean()
        '''
        dist = dist.view(-1)
        if self.hard_mining:
            # hard
            hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
            hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
            if self.margin > 0:
                hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)
            else:
                hard_loss_metric = F.softplus(hard_hp_dist - hard_hn_dist).view(n, -1)
                
            nonzero_num = (hard_loss_metric != 0).sum(1).float()

            if self.nonzero:
                hard_loss_metric_mean = hard_loss_metric.sum(1) / nonzero_num
                hard_loss_metric_mean[nonzero_num == 0] = 0
            else:
                hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)

            return hard_loss_metric_mean.mean(), nonzero_num.mean()
        else:
            # full
            full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
            full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
            if self.margin > 0:
                full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
            else:
                full_loss_metric = F.softplus(full_hp_dist - full_hn_dist).view(n, -1)  

            nonzero_num = (full_loss_metric != 0).sum(1).float()

            if self.nonzero:
                full_loss_metric_mean = full_loss_metric.sum(1) / nonzero_num
                full_loss_metric_mean[nonzero_num == 0] = 0
            else:
                full_loss_metric_mean = full_loss_metric.mean(1)
            
            # print("full_loss_metric={}, nonzero_num={}".format(full_loss_metric_mean, nonzero_num))
            return full_loss_metric_mean.mean(), nonzero_num.mean()
        '''

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist