#!/usr/bin/env python
# coding=utf-8
import os
import os.path as osp
import cv2
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CASIA_B', choices=['CASIA_B', 'OUMVLP_CL', 'Outdoor_Gait', 'CCPG', 'CASIA_E'], type=str, help='name of dataset')
parser.add_argument('--dataset_path', default='/mnt/Dataset/casia_b/', type=str, help='path to dataset')
parser.add_argument('--resolution', default=128, type=int, help='resolution')
parser.add_argument('--noise_rate', default=0.5, type=float, help='split train for noise')
parser.add_argument('--seed', default=2022, type=int, help='random seed')

args = parser.parse_args()
if args.dataset_path[-1] == '/':
    args.dataset_path = args.dataset_path[:-1]
print("Args:", args)

if args.resolution == 128:
    src_dir = osp.join(args.dataset_path, 'silhouettes_cut128_pkl')
    des_dir = osp.join('{}_nr{:.2f}_seed{}'.format(args.dataset_path, args.noise_rate, args.seed), \
                        'silhouettes_cut128_pkl_nr{:.2f}_seed{}'.format(args.noise_rate, args.seed))
elif args.resolution == 64:
    src_dir = osp.join(args.dataset_path, 'silhouettes_cut_pkl')
    des_dir = osp.join('{}_nr{:.2f}_seed{}'.format(args.dataset_path, args.noise_rate, args.seed), \
                        'silhouettes_cut_pkl_nr{:.2f}_seed{}'.format(args.noise_rate, args.seed))
print('src_dir={}'.format(src_dir))
print('des_dir={}'.format(des_dir))

if args.dataset.replace('-', '_') == 'CASIA_B':
    total_id_list = []
    for i in range(1, 125):
        id_name = '{:0>3d}'.format(i)
        if args.dataset.replace('-', '_') == 'CASIA_B' and id_name == '005':
            continue
        total_id_list.append(id_name)
    train_id_list = total_id_list[:73]
    test_id_list = total_id_list[73:]     
#############################################################
clean_train_id_list = []
noise_train_id_list = []
noise_num = int(len(train_id_list)*args.noise_rate)
np.random.seed(args.seed)
np.random.shuffle(train_id_list)
noise_train_id_list = sorted(train_id_list[:noise_num])
clean_train_id_list = sorted(train_id_list[noise_num:])
assert(len(clean_train_id_list) + len(noise_train_id_list) == len(train_id_list))
#############################################################

print("####################################################")            
print('train_id_list={}, num={}'.format(train_id_list, len(train_id_list)))
print('test_id_list={}, num={}'.format(test_id_list, len(test_id_list)))
print('clean_train_id_list={}, num={}'.format(clean_train_id_list, len(clean_train_id_list)))
print('noise_train_id_list={}, num={}'.format(noise_train_id_list, len(noise_train_id_list)))
print("####################################################")

def process_id(id0):
    id_path = os.path.join(src_dir, id0)
    if id0 in test_id_list or id0 in clean_train_id_list:
        new_id = id0
        new_id_path = os.path.join(des_dir, new_id)
        os.makedirs(new_id_path, exist_ok=True)
        cmd = 'cp -r {}/* {}'.format(id_path, new_id_path)
        # print(cmd)
        os.system(cmd)
        print("Clean ID={}".format(id0))
    elif id0 in noise_train_id_list:
        for i, type0 in enumerate(sorted(os.listdir(id_path))):
            type_path = os.path.join(id_path, type0)
            if 'CCPG' in src_dir:
                new_id = "{}_noise{}".format(id0, i+1)
                new_id_path = os.path.join(des_dir, new_id)
                os.makedirs(new_id_path, exist_ok=True)
                cmd = 'cp -r {} {}'.format(type_path, new_id_path)
                # print(cmd)
                os.system(cmd)
            else:
                if 'cl' in type0:
                    new_id = "{}_noise".format(id0)
                    new_id_path = os.path.join(des_dir, new_id)
                    os.makedirs(new_id_path, exist_ok=True)
                    cmd = 'cp -r {} {}'.format(type_path, new_id_path)
                    # print(cmd)
                    os.system(cmd)
                else:
                    new_id = id0
                    new_id_path = os.path.join(des_dir, new_id)
                    os.makedirs(new_id_path, exist_ok=True)
                    cmd = 'cp -r {} {}'.format(type_path, new_id_path)
                    # print(cmd)
                    os.system(cmd)
        print("Noise ID={}".format(id0))
    else:
        print("####################################################")
        print("Ignore ID={}".format(id0))
        print("####################################################")              
    return

id_list = sorted(os.listdir(src_dir))
# for id0 in id_list:
#     process_id(id0)
from multiprocessing import Pool
pool = Pool()
pool.map(process_id, id_list)
pool.close()

# resplit after adding id noise
new_train_id_list = []
new_test_id_list = test_id_list.copy()
new_clean_train_id_list = clean_train_id_list.copy()
new_noise_train_id_list = []
# for id_name in clean_train_id_list:
#     new_train_id_list.append(id_name)
# for id_name in noise_train_id_list:
#     new_train_id_list.append(id_name)
#     new_train_id_list.append("{}_noise".format(id_name))
#     new_noise_train_id_list.append(id_name)
#     new_noise_train_id_list.append("{}_noise".format(id_name))
des_id_list = sorted(os.listdir(des_dir))
new_train_id_list = [_ for _ in des_id_list if _ not in new_test_id_list]
new_noise_train_id_list = [_ for _ in des_id_list if _ not in new_test_id_list+new_clean_train_id_list]
del train_id_list, test_id_list, clean_train_id_list, noise_train_id_list
assert(len(new_clean_train_id_list) + len(new_noise_train_id_list) == len(new_train_id_list))
print("####################################################")
print('new_train_id_list={}, num={}'.format(new_train_id_list, len(new_train_id_list)))
print('new_test_id_list={}, num={}'.format(new_test_id_list, len(new_test_id_list)))
print('new_clean_train_id_list={}, num={}'.format(new_clean_train_id_list, len(new_clean_train_id_list)))
print('new_noise_train_id_list={}, num={}'.format(new_noise_train_id_list, len(new_noise_train_id_list)))
print("####################################################")
pid_fname = osp.join('partition', '{}_nr{:.2f}_seed{}.npy'.format(args.dataset, args.noise_rate, args.seed))
if not osp.exists(pid_fname):
    pid_dict = {}
    pid_dict.update({'train_id_list':new_train_id_list})
    pid_dict.update({'test_id_list':new_test_id_list})
    pid_dict.update({'clean_train_id_list':new_clean_train_id_list})
    pid_dict.update({'noise_train_id_list':new_noise_train_id_list})
    np.save(pid_fname, pid_dict)
    print('{} Saved'.format(pid_fname))
else:
    pid_dict = np.load(pid_fname).item()
    assert(set(pid_dict['train_id_list']) == set(new_train_id_list))
    assert(set(pid_dict['test_id_list']) == set(new_test_id_list))
    assert(set(pid_dict['clean_train_id_list']) == set(new_clean_train_id_list))
    assert(set(pid_dict['noise_train_id_list']) == set(new_noise_train_id_list))
    print('{} Exists'.format(pid_fname))
