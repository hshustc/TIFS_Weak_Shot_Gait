from datetime import datetime
import numpy as np
import argparse
import os
import pickle

from model.initialization import initialization
from model.eval import evaluation
from model.utils import print_CMC, print_metric, print_metric_exclude_negative

from config import *
config.update({'phase':'test'})
m = initialization(config)
print('#######################################')
print("Network Structures:", m.encoder)
print('#######################################')

print('Loading the model of %s' % config['ckp_prefix'])
m.init_model('{}encoder.ptm'.format(config['ckp_prefix']))
print('Transforming...')
print('#######################################')
print('Feat_idx={}'.format(config['feat_idx']))
print('#######################################')
time = datetime.now()
eval_feature_pkl = '{}_{}set_{}eval_feature.pkl'.format('_'.join(config['dataset']), config['test_set'], os.path.split(config['ckp_prefix'])[-1])
if config['resume'] and os.path.exists(eval_feature_pkl):
    print("{} EXISTS".format(eval_feature_pkl))
    data = pickle.load(open(eval_feature_pkl, 'rb'))
else:
    data = m.transform(config['test_set'], batch_size=config['clean_batch_size'][0], feat_idx=config['feat_idx'])
    pickle.dump(data, open(eval_feature_pkl, 'wb'), protocol=4)
    print("{} SAVED".format(eval_feature_pkl))

#eval
print('#######################################')
print('Evaluating with Reranking={} ...'.format(config['reranking']))
CMC, mAP, PR_thres = evaluation(data, config)
print('Evaluation Complete. Cost:', datetime.now() - time)
print('#######################################')
if config['remove_no_gallery']:
    print("The Seqs that have NO GALLERY are REMOVED.")
else:
    print("The Seqs that have NO GALLERY are INCLUDED.")
print('#######################################')
print_CMC(CMC, config)
print_metric(mAP, config, metric_name='mAP')
if config['euc_or_cos_dist'] == 'cos' and config['cos_sim_thres'] > -1:
    print_metric_exclude_negative(PR_thres[0], config, metric_name='Precision@COS_SIM_THRES={}'.format(config['cos_sim_thres']))
    print_metric(PR_thres[1], config, metric_name='Recall@COS_SIM_THRES={}'.format(config['cos_sim_thres']))
print('#######################################')
