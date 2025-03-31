import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from .re_ranking import re_ranking
from .metric import compute_View_Aware_ACC_mAP, compute_CMC_mAP, compute_PR

def cuda_euc_dist(x, y):
    x = x.permute(1, 0, 2).contiguous() # num_parts * num_probe * part_dim
    y = y.permute(1, 0, 2).contiguous() # num_parts * num_gallery * part_dim
    dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
        2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2)) # num_parts * num_probe * num_gallery
    dist = torch.sqrt(F.relu(dist)) # num_parts * num_probe * num_gallery
    dist = torch.mean(dist, 0) # num_probe * num_gallery
    return dist

def cuda_cos_dist(x, y):
    x = F.normalize(x, p=2, dim=2).permute(1, 0, 2) # num_parts * num_probe * part_dim
    y = F.normalize(y, p=2, dim=2).permute(1, 0, 2) # num_parts * num_gallery * part_dim
    dist = 1 - torch.mean(torch.matmul(x, y.transpose(1, 2)), 0) # num_probe * num_gallery
    return dist

def evaluation(data, config):
    print("############################")
    if config['euc_or_cos_dist'] == 'euc':
        print("Compute Euclidean Distance")
    elif config['euc_or_cos_dist'] == 'cos':
        print("Compute Cosine Distance")
    else:
        print('Illegal Distance Type')
        os._exit(0)
    print("############################")

    assert(len(config['dataset'])==1)
    dataset = config['dataset'][0].replace('-', '_')
    probe_seq_dict = {'CASIA_B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]
                      }
    gallery_seq_dict = {'CASIA_B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]
                        }

    feature, view, seq_type, label = data
    label = np.asarray(label)
    view_list = sorted(list(set(view)))
    view_num = len(view_list)
    sample_num = len(feature)

    print("############################")
    print("Feature Shape: ", feature.shape)
    print("############################")

    CMC = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, config['max_rank']])
    mAP = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    P_thres = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    R_thres = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_y = label[gseq_mask]
                    gseq_mask = torch.from_numpy(np.asarray(gseq_mask, dtype=np.uint8))
                    gallery_x = feature[gseq_mask, :, :]

                    if config['remove_no_gallery']:
                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)
                    else:
                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_y = label[pseq_mask]
                    pseq_mask = torch.from_numpy(np.asarray(pseq_mask, dtype=np.uint8))
                    probe_x = feature[pseq_mask, :, :]

                    print('probe_type={}, gallery_type={}, probe_view={}, gallery_view={}, num_probe={}, num_gallery={}'.format( \
                            probe_seq, gallery_seq, probe_view, gallery_view, pseq_mask.sum(), gseq_mask.sum()))

                    if config['reranking']:
                        assert(config['euc_or_cos_dist'] == 'cos')
                        dist_p_p = 1 - cuda_cos_dist(probe_x, probe_x).cpu().numpy()
                        dist_p_g = 1 - cuda_cos_dist(probe_x, gallery_x).cpu().numpy()
                        dist_g_g = 1 - cuda_cos_dist(gallery_x, gallery_x).cpu().numpy()
                        dist = re_ranking(dist_p_g, dist_p_p, dist_g_g, lambda_value=config['relambda'])
                    else:
                        if config['euc_or_cos_dist'] == 'euc':
                            dist = cuda_euc_dist(probe_x, gallery_x)
                        elif config['euc_or_cos_dist'] == 'cos':
                            dist = cuda_cos_dist(probe_x, gallery_x)
                        dist = dist.cpu().numpy()
                    eval_results = compute_CMC_mAP(dist, probe_y, gallery_y, config['max_rank'])
                    CMC[p, v1, v2, :] = np.round(eval_results[0] * 100, 2)
                    mAP[p, v1, v2] = np.round(eval_results[1] * 100, 2)
                    if config['euc_or_cos_dist'] == 'cos' and config['cos_sim_thres'] > -1:
                        eval_results = compute_PR(dist, probe_y, gallery_y, config['cos_sim_thres'])
                        P_thres[p, v1, v2] = np.round(eval_results[0] * 100, 2)
                        R_thres[p, v1, v2] = np.round(eval_results[1] * 100, 2)

    return CMC, mAP, [P_thres, R_thres]
