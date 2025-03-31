import torch
import torch.utils.data as tordata
import torch.distributed as dist
import math
import random
import numpy as np

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, clean_batch_size, noise_batch_size):
        self.dataset = dataset
        self.clean_batch_size = clean_batch_size
        self.noise_batch_size = noise_batch_size
        self.clean_label_set = self.dataset.clean_label_set
        self.noise_label_set = self.dataset.noise_label_set

    def __iter__(self):
        while (True):
            sample_indices = list()
            # pid_list = np.random.choice(self.dataset.label_set, 
            #     self.batch_size[0], replace=False)
            clean_pid_list = np.asarray([])
            if len(self.clean_label_set) > 0 and self.clean_batch_size[0] > 0:
                clean_pid_list = np.random.choice(self.clean_label_set, self.clean_batch_size[0], replace=False)
            noise_pid_list = np.asarray([])
            if len(self.noise_label_set) > 0 and self.noise_batch_size[0] > 0:
                if self.dataset.noise_class_sim is None:
                    noise_pid_list = np.random.choice(self.noise_label_set, self.noise_batch_size[0], replace=False)
                else:
                    noise_pid_list_s1 = np.random.choice(self.noise_label_set, self.noise_batch_size[0]//2, replace=False)
                    noise_pid_list_s2 = []
                    for npid in noise_pid_list_s1:
                        npid_index = self.noise_label_set.index(npid)
                        npid_class_sim = self.dataset.noise_class_sim[npid_index, :].data.cpu().numpy()
                        assert(npid_class_sim[npid_index] == 0)
                        npid_s2 = np.random.choice(self.noise_label_set, 1, replace=False, p=npid_class_sim)
                        noise_pid_list_s2.append(npid_s2[0])
                        # print('npid_s1={}, npid_s2={}'.format(npid, npid_s2))
                    noise_pid_list = np.concatenate((noise_pid_list_s1, np.asarray(noise_pid_list_s2)))
            pid_list = np.concatenate((clean_pid_list, noise_pid_list))
            # print('pid_list={}'.format(pid_list))
            for pid in pid_list:
                if pid in self.clean_label_set:
                    batch_seqs_per_id = self.clean_batch_size[1]
                elif pid in self.noise_label_set:
                    batch_seqs_per_id = self.noise_batch_size[1]
                _index = self.dataset.index_dict[pid]
                if len(_index) >= batch_seqs_per_id:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=False).tolist()
                else:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=True).tolist()             
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

class DistributedTripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, clean_batch_size, noise_batch_size, world_size=None, rank=None, random_seed=2019):
        np.random.seed(random_seed)
        random.seed(random_seed)
        print("random_seed={} for DistributedTripletSampler".format(random_seed))
        
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.clean_batch_size = clean_batch_size
        self.noise_batch_size = noise_batch_size
        self.clean_label_set = self.dataset.clean_label_set
        self.noise_label_set = self.dataset.noise_label_set
        self.world_size = world_size
        self.rank = rank
        self.random_seed = 0
        
        #############################################################
        if len(self.clean_label_set) > 0 and self.clean_batch_size[0] > 0:
            if len(self.noise_label_set) > 0 and self.noise_batch_size[0] > 0:
                # clean
                self.clean_world_size = min(int(self.world_size/2), self.clean_batch_size[0])
                assert(self.clean_batch_size[0] % self.clean_world_size == 0)
                self.clean_batch_ids_per_world = int(math.ceil(self.clean_batch_size[0] * 1.0 / self.clean_world_size))
                self.clean_total_batch_per_world = int(math.ceil(len(self.clean_label_set) * 1.0 / self.clean_batch_size[0]))
                # noise
                self.noise_world_size = self.world_size - self.clean_world_size
                assert(self.noise_batch_size[0] % self.noise_world_size == 0)
                self.noise_batch_ids_per_world = int(math.ceil(self.noise_batch_size[0] * 1.0 / self.noise_world_size))
                self.noise_total_batch_per_world = int(math.ceil(len(self.noise_label_set) * 1.0 / self.noise_batch_size[0]))
                self.total_batch_per_world = min(self.clean_total_batch_per_world, self.noise_total_batch_per_world)
            else:
                self.clean_batch_ids_per_world = int(math.ceil(self.clean_batch_size[0] * 1.0 / self.world_size))
                assert(self.clean_batch_size[0] % self.world_size == 0)
                self.total_batch_per_world = int(math.ceil(len(self.clean_label_set) * 1.0 / self.clean_batch_size[0]))
        else:
            self.noise_batch_ids_per_world = int(math.ceil(self.noise_batch_size[0] * 1.0 / self.world_size))
            assert(self.noise_batch_size[0] % self.world_size == 0)
            self.total_batch_per_world = int(math.ceil(len(self.noise_label_set) * 1.0 / self.noise_batch_size[0]))
        #############################################################
                                

    def __iter__(self):
        while (True):
            g = torch.Generator()
            g.manual_seed(self.random_seed)
            if len(self.clean_label_set) > 0 and self.clean_batch_size[0] > 0:
                if len(self.noise_label_set) > 0 and self.noise_batch_size[0] > 0:
                    world_size = self.clean_world_size
                    if self.rank < world_size:
                        label_set = self.clean_label_set
                        batch_ids_per_world = self.clean_batch_ids_per_world
                    else:
                        label_set = self.noise_label_set
                        batch_ids_per_world = self.noise_batch_ids_per_world
                else:
                    world_size = self.world_size
                    label_set = self.clean_label_set
                    batch_ids_per_world = self.clean_batch_ids_per_world
            else:
                world_size = self.world_size
                label_set = self.noise_label_set
                batch_ids_per_world = self.noise_batch_ids_per_world                    
            pid_index_all_world = torch.randperm(len(label_set), generator=g).tolist()
            pid_index_cur_world = pid_index_all_world[self.rank:len(label_set):world_size]
            # if self.rank == 0:
            #     print("random_seed={}".format(self.random_seed))
            #     print("pid_index_all_world={}, pid_index_cur_world={}".format(pid_index_all_world, pid_index_cur_world))
            #     print("batch_ids_per_world={}, total_batch_per_world={}".format(self.batch_ids_per_world, self.total_batch_per_world))
            
            sample_indices = list()
            # pid_index_cur_batch = random.sample(pid_index_cur_world, self.batch_ids_per_world)
            noise_hard =  (len(self.clean_label_set) > 0 and self.clean_batch_size[0] > 0) & \
                            (len(self.noise_label_set) > 0 and self.noise_batch_size[0] > 0) & \
                                (self.rank >= world_size) & (self.dataset.noise_class_sim is not None)
            if noise_hard:
                assert(label_set[0] == self.noise_label_set[0])
                assert(label_set[-1] == self.noise_label_set[-1])
                pid_index_cur_batch_s1 = np.random.choice(pid_index_cur_world, batch_ids_per_world//2, replace=False)
                pid_index_cur_batch_s2 = []
                for npid_index in pid_index_cur_batch_s1:
                    npid_class_sim = self.dataset.noise_class_sim[npid_index, :].data.cpu().numpy()
                    assert(npid_class_sim[npid_index] == 0)
                    npid_s2 = np.random.choice(self.noise_label_set, 1, replace=False, p=npid_class_sim)
                    npid_index_s2 = self.noise_label_set.index(npid_s2)
                    pid_index_cur_batch_s2.append(npid_index_s2)
                    # print('npid_s1={}, npid_s2={}'.format(label_set[npid_index], label_set[npid_index_s2]))
                pid_index_cur_batch = np.concatenate((pid_index_cur_batch_s1, np.asarray(pid_index_cur_batch_s2)))
            else: 
                pid_index_cur_batch = np.random.choice(pid_index_cur_world, batch_ids_per_world, replace=False)
            # print('rank={}, pid_list={}'.format(self.rank, [label_set[tmp] for tmp in pid_index_cur_batch]))
            for pid_index in pid_index_cur_batch:
                pid_name = label_set[pid_index]
                if pid_name in self.clean_label_set:
                    batch_seqs_per_id = self.clean_batch_size[1]
                elif pid_name in self.noise_label_set:
                    batch_seqs_per_id = self.noise_batch_size[1]
                _index = self.dataset.index_dict[pid_name]
                # _index = random.choices(_index, k=self.batch_size[1])
                if len(_index) >= batch_seqs_per_id:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=False).tolist()
                else:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=True).tolist() 
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

    def set_random_seed(self, seed):
        self.random_seed = seed