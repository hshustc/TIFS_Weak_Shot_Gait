import torch
import torch.nn as nn
import torch.nn.functional as F

class PartAdacontLoss(nn.Module):
    def __init__(self, pos_thres=-1, neg_thres=-1):
        super(PartAdacontLoss, self).__init__()
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres

    def forward(self, feature, label, label_type=None, gl_sim=None):
        # feature: [n, m, d], label: [n, m], label_type: [n, m], gl_sim: [m, m]
        # print('feature shape={}, label shape={}, label_type shape={}, gl_sim={}'.format(feature.shape, label.shape, label_type.shape, gl_sim.shape))   
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(2) == label.unsqueeze(1)).bool() # n x m x m
        hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).int() # n x m x m
        true_hn_mask1 = (label_type.unsqueeze(2).repeat(1, 1, m) > 0).int() # anchor-negative of clean label as anchor
        true_hn_mask2 = ((label_type.unsqueeze(2) - label_type.unsqueeze(1)) < 0).int() # anchor-negative of noise label as anchor and clean label as negative
        true_hn_mask = ((hn_mask * true_hn_mask1 + hn_mask * true_hn_mask2) > 0).bool() # n x m x m
        true_mask = hp_mask | true_hn_mask
        noise_mask = ~true_mask

        gl_sim = gl_sim.unsqueeze(0).repeat(n, 1, 1) # n x m x m
        pred_hp_mask = noise_mask & (gl_sim>self.pos_thres) # n x m x m
        pred_hn_mask = noise_mask & (gl_sim<self.neg_thres) # n x m x m

        dist = self.batch_dist(feature) # n x m x m
        full_hp_loss_metric_list = []
        hp_nonzero_num_list = []
        full_hn_loss_metric_list = []
        hn_nonzero_num_list = []
        for i in range(m):
            pred_total_cnt = (noise_mask[:, i, :] != 0).sum(-1).float()
            pred_pos_cnt = (pred_hp_mask[:, i, :] != 0).sum(-1).float()
            pred_neg_cnt = (pred_hn_mask[:, i, :] != 0).sum(-1).float()
            # print("index={}, pred_total_cnt={}, pred_pos_cnt={}, pred_neg_cnt={}".format(i, pred_total_cnt.mean(), pred_pos_cnt.mean(), pred_neg_cnt.mean()))

            if self.pos_thres > -1 and pred_pos_cnt.mean() > 0:
                full_hp_dist = torch.masked_select(dist[:, i, :], hp_mask[:, i, :]).view(n, -1)
                pred_hp_dist = torch.masked_select(dist[:, i, :], pred_hp_mask[:, i, :]).view(n, -1)
                pred_hp_sim = torch.masked_select(gl_sim[:, i, :], pred_hp_mask[:, i, :]).view(n, -1)
                pred_hp_margin = torch.max(full_hp_dist, dim=1, keepdim=True)[0] * self.pos_sim2coeff(pred_hp_sim)
                full_hp_loss_metric = F.relu(pred_hp_dist - pred_hp_margin.detach())
                full_hp_loss_metric_list.append(full_hp_loss_metric.sum(1, keepdim=True))
                hp_nonzero_num_list.append((full_hp_loss_metric != 0).sum(1, keepdim=True).float())

            if self.neg_thres > -1 and pred_neg_cnt.mean() > 0:
                full_hn_dist = torch.masked_select(dist[:, i, :], true_hn_mask[:, i, :]).view(n, -1)
                pred_hn_dist = torch.masked_select(dist[:, i, :], pred_hn_mask[:, i, :]).view(n, -1)
                pred_hn_sim = torch.masked_select(gl_sim[:, i, :], pred_hn_mask[:, i, :]).view(n, -1)
                pred_hn_margin = torch.min(full_hn_dist, dim=1, keepdim=True)[0] * self.neg_sim2coeff(pred_hn_sim)
                full_hn_loss_metric = F.relu(pred_hn_margin.detach() - pred_hn_dist)
                full_hn_loss_metric_list.append(full_hn_loss_metric.sum(1, keepdim=True))
                hn_nonzero_num_list.append((full_hn_loss_metric != 0).sum(1, keepdim=True).float())

        # hp
        if len(full_hp_loss_metric_list) > 0:
            full_hp_loss_metric = torch.cat(full_hp_loss_metric_list, dim=1).sum(1)
            hp_nonzero_num = torch.cat(hp_nonzero_num_list, dim=1).sum(1)
            full_hp_loss_metric_mean = full_hp_loss_metric / hp_nonzero_num
            full_hp_loss_metric_mean[hp_nonzero_num == 0] = 0
        else:
            full_hp_loss_metric_mean = torch.zeros(n)
            hp_nonzero_num = torch.zeros(n)
        # hn
        if len(full_hn_loss_metric_list) > 0:
            full_hn_loss_metric = torch.cat(full_hn_loss_metric_list, dim=1).sum(1)
            hn_nonzero_num = torch.cat(hn_nonzero_num_list, dim=1).sum(1)
            full_hn_loss_metric_mean = full_hn_loss_metric / hn_nonzero_num
            full_hn_loss_metric_mean[hn_nonzero_num == 0] = 0
        else:
            full_hn_loss_metric_mean = torch.zeros(n)
            hn_nonzero_num = torch.zeros(n)

        return full_hp_loss_metric_mean.mean(), full_hn_loss_metric_mean.mean(), hp_nonzero_num.mean(), hn_nonzero_num.mean()

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist

    def pos_sim2coeff(self, x):
        return (3-x)/2 # x=1, y=1, negative correlation, x~[-1,1]->y~[1,2], x~[0.5,1]->y~[1,1.25]

    def neg_sim2coeff(self, x):
        return (1-x)/2 # x=-1, y=1, negative correlation, x~[-1,1]->y~[0,1], x~[-1,0.1]->y~[0.45,1] 