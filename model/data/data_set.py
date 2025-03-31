import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2

class DataSet(tordata.Dataset):
    def __init__(self, seq_dir_list, seq_label_list, index_dict, resolution, cut_padding, clean_label_set=[], noise_label_set=[]):
        self.seq_dir_list = seq_dir_list
        self.seq_label_list = seq_label_list
        self.index_dict = index_dict
        self.resolution = int(resolution)
        self.cut_padding = int(cut_padding)
        self.data_size = len(self.seq_label_list)
        self.label_set = sorted(list(set(self.seq_label_list)))
        self.clean_label_set = sorted(clean_label_set)
        self.noise_label_set = sorted(noise_label_set)
        assert(len( set(self.clean_label_set).intersection(set(self.noise_label_set)) )==0)
        if len(self.clean_label_set) > 0 or len(self.noise_label_set) > 0:
            assert(len(self.clean_label_set) + len(self.noise_label_set) == len(self.label_set))
        else:
            self.clean_label_set = self.label_set.copy()
        #############################################################
        self.noise_class_center = None
        self.noise_class_sim = None
        self.all2noise_index_map = {}
        for label in self.label_set:
            if label in self.noise_label_set:
                all_index = self.label_set.index(label)
                noise_index = self.noise_label_set.index(label)
                self.all2noise_index_map.update({all_index:noise_index})
        #############################################################
        print("####################################################")
        print('DataSet Initialization')
        print('label_set={}, num={}'.format(self.label_set, len(self.label_set)))
        print('clean_label_set={}, num={}'.format(self.clean_label_set, len(self.clean_label_set)))
        print('noise_label_set={}, num={}'.format(self.noise_label_set, len(self.noise_label_set)))
        print("####################################################")

    def __loader__(self, path):
        if self.cut_padding > 0:
            return self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0
        else: 
            return self.img2xarray(
                path).astype(
                'float32') / 255.0

    def __getitem__(self, index):
        seq_path = self.seq_dir_list[index]
        seq_imgs = self.__loader__(seq_path)
        seq_label = self.seq_label_list[index]
        return seq_imgs, seq_label

    def img2xarray(self, file_path):
        pkl_name = '{}.pkl'.format(os.path.basename(file_path))
        all_imgs = pickle.load(open(osp.join(file_path, pkl_name), 'rb'))
        return all_imgs

    def __len__(self):
        return self.data_size
