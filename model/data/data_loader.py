import os
import os.path as osp

import numpy as np
import pickle

from .data_set import DataSet

def load_data(config):
    print("####################################################")
    dataset, dataset_path, resolution, pid_num, pid_shuffle = \
        config['dataset'], config['dataset_path'], config['resolution'], config['pid_num'], config['pid_shuffle']
    print("dataset={}, dataset_path={}, resolution={}".format(dataset, dataset_path, resolution))
    print("####################################################")
    seq_dir_list = list()
    seq_id_list = list()

    cut_padding_ok = False
    for i, dataset_path_i in enumerate(dataset_path):
        dataset_i = dataset[i].replace('-', '_')
        assert(dataset_i.lower().replace('-', '_') in dataset_path_i.lower().replace('-', '_'))
        prefix = "{}-".format(dataset_i) if i > 0 else ''
        print("dataset_path={}, prefix={}".format(dataset_path_i, prefix))
        check_frames = config['check_frames']
        check_resolution = config['check_resolution']
        for _id in sorted(list(os.listdir(dataset_path_i))):
            # In CASIA_B, data of subject #5 is incomplete. Thus, we ignore it in training.
            if dataset_i == 'CASIA_B' and _id.split('_')[0] == '005':
                continue
            id_path = osp.join(dataset_path_i, _id)
            for _type in sorted(list(os.listdir(id_path))):
                type_path = osp.join(id_path, _type)
                for _view in sorted(list(os.listdir(type_path))):
                    view_path = osp.join(type_path, _view)
                    if check_frames:
                        pkl_name = '{}.pkl'.format(os.path.basename(view_path))
                        all_imgs = pickle.load(open(osp.join(view_path, pkl_name), 'rb'))
                        if all_imgs.shape[0] < 15:
                            continue
                    seq_dir_list.append(view_path)
                    seq_id_list.append(prefix+_id)
                    #############################################################
                    if not cut_padding_ok:
                        pkl_name = '{}.pkl'.format(os.path.basename(view_path))
                        all_imgs = pickle.load(open(osp.join(view_path, pkl_name), 'rb'))
                        assert(all_imgs.shape[1]==resolution)
                        assert(all_imgs.shape[2]==resolution or all_imgs.shape[2]==(resolution-2*int(float(resolution)/64*10)))
                        if all_imgs.shape[2]==resolution:
                            cut_padding = int(float(resolution)/64*10)
                        else:
                            cut_padding = 0
                        cut_padding_ok = True
                    #############################################################
                    if check_resolution:
                        pkl_name = '{}.pkl'.format(os.path.basename(view_path))
                        all_imgs = pickle.load(open(osp.join(view_path, pkl_name), 'rb'))
                        assert(all_imgs.shape[1]==resolution)
                        if cut_padding > 0:
                            assert(all_imgs.shape[2]==resolution)
                        else:
                            assert(all_imgs.shape[2]==(resolution-2*int(float(resolution)/64*10)))
                        check_resolution = False
                        print("Check Resolution: view_path={}, resolution={}, cut_padding={}, img_shape={}".format(\
                                view_path, resolution, cut_padding, all_imgs.shape))
                    #############################################################-
    print("####################################################")
    print('DataLoader: Check Data')
    print('number of total id: {}'.format(len(set(seq_id_list))))
    print('number of total seq: {}'.format(len(seq_dir_list)))
    print("####################################################")

    total_id = len(list(set(seq_id_list)))
    pid_fname = config['pid_fname']
    if not osp.exists(pid_fname):
        print('{} *NOT* exists'.format(pid_fname))
        os._exit(1)  
    pid_dict = np.load(pid_fname).item()
    train_id_list = pid_dict['train_id_list']
    test_id_list = pid_dict['test_id_list']
    clean_train_id_list = pid_dict['clean_train_id_list']
    noise_train_id_list = pid_dict['noise_train_id_list']

    print("####################################################")
    print('DataLoader: Load Partition')
    print("pid_fname:", pid_fname)
    print("resolution={}, cut_padding={}".format(resolution, cut_padding))
    print('train_id_list={}, num={}'.format(train_id_list, len(train_id_list)))
    print('test_id_list={}, num={}'.format(test_id_list, len(test_id_list)))
    print('clean_train_id_list={}, num={}'.format(clean_train_id_list, len(clean_train_id_list)))
    print('noise_train_id_list={}, num={}'.format(noise_train_id_list, len(noise_train_id_list)))
    print("####################################################")

    # train source
    train_seq_dir_list = [seq_dir_list[i] for i, l in enumerate(seq_id_list) if l in train_id_list]
    train_seq_id_list = [seq_id_list[i] for i, l in enumerate(seq_id_list) if l in train_id_list]
    train_index_info = {}
    for i, l in enumerate(train_seq_id_list):
        if l not in train_index_info.keys():
            train_index_info[l] = []
        train_index_info[l].append(i)
    if config['noise_split']:
        train_source = DataSet(train_seq_dir_list, train_seq_id_list, train_index_info, resolution, cut_padding, \
                                    clean_label_set=clean_train_id_list, noise_label_set=noise_train_id_list)
    else:
        train_source = DataSet(train_seq_dir_list, train_seq_id_list, train_index_info, resolution, cut_padding, \
                                    clean_label_set=train_id_list, noise_label_set=[])

    # test source
    test_seq_dir_list = [seq_dir_list[i] for i, l in enumerate(seq_id_list) if l in test_id_list]
    test_seq_id_list = [seq_id_list[i] for i, l in enumerate(seq_id_list) if l in test_id_list]
    test_index_info = {}
    for i, l in enumerate(test_seq_id_list):
        if l not in test_index_info.keys():
            test_index_info[l] = []
        test_index_info[l].append(i)
    test_source = DataSet(test_seq_dir_list, test_seq_id_list, test_index_info, resolution, cut_padding)

    print("####################################################")
    print('DataLoader: Split Train and Test')
    print('train_label_set={}, num={}'.format(sorted(list(set(train_seq_id_list))), len(list(set(train_seq_id_list)))))
    print('number of train seq: {}'.format(len(train_seq_dir_list)))
    print('test_label_set={}, num={}'.format(sorted(list(set(test_seq_id_list))), len(list(set(test_seq_id_list)))))
    print('number of test seq: {}'.format(len(test_seq_dir_list)))
    print("####################################################")

    return train_source, test_source
