import math
import os
import os.path as osp
import random
import sys
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import TripletSampler, DistributedTripletSampler, build_data_transforms
from .loss import all_gather, GatherLayer, DistributedLossWrapper, DistributedLossWrapperWithLabelType, DistributedLossWrapperWithLabelTypeWithLabelOrigin
from .loss import PartTripletLoss, PartAdacontLoss, GlInfoNCELoss
from .loss import CenterLoss, CrossEntropyLabelSmooth
from .solver import WarmupMultiStepLR
from .network import GaitSet
from .network.sync_batchnorm import DataParallelWithCallback

class Model:
    def __init__(self, config):
        self.config = deepcopy(config)
        if self.config['DDP']:
            torch.cuda.set_device(self.config['local_rank'])
            dist.init_process_group(backend='nccl')
            self.config['encoder_infonce_weight'] *= dist.get_world_size()
            self.config['encoder_triplet_weight'] *= dist.get_world_size()
            self.config['encoder_adacont_weight'] *= dist.get_world_size()
            self.random_seed = self.config['random_seed'] + dist.get_rank()
        else:
            self.random_seed = self.config['random_seed']
        
        self.config.update({'num_id': len(self.config['train_source'].label_set)})
        self.encoder = GaitSet(self.config).float().cuda()
        if self.config['DDP']:
            self.encoder = DDP(self.encoder, device_ids=[self.config['local_rank']], output_device=self.config['local_rank'], find_unused_parameters=True)
        else:
            self.encoder = DataParallelWithCallback(self.encoder)
        self.build_data()
        self.build_loss()
        self.build_loss_metric()
        self.build_optimizer()

        if self.config['DDP']:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def build_data(self):
        # data augment
        if self.config['dataset_augment']:
            self.data_transforms = build_data_transforms(random_erasing=True, random_rotate=False, \
                                        random_horizontal_flip=False, random_pad_crop=False, \
                                        resolution=self.config['resolution'], random_seed=self.random_seed) 
        
        #triplet sampler
        if self.config['DDP']:
            self.triplet_sampler = DistributedTripletSampler(self.config['train_source'], \
                                        self.config['clean_batch_size'], self.config['noise_batch_size'], random_seed=self.random_seed)
        else:
            self.triplet_sampler = TripletSampler(self.config['train_source'], self.config['clean_batch_size'], self.config['noise_batch_size'])

    def update_noise_center(self, feature, label):
        if self.config['DDP']:
            feature = torch.cat(all_gather(feature), dim=0)
            label = torch.cat(all_gather(label), dim=0)
        feature = feature.detach()
        label = label.detach()
        assert(feature.size(0) == label.size(0))
        # print('feature size={}, label size={}'.format(feature.size(), label.size()))
        # print('label set={}, label set size={}'.format(torch.unique(label), torch.unique(label).size()))

        if self.config['train_source'].noise_class_center is None:
            self.config['train_source'].noise_class_center = torch.zeros((len(self.config['train_source'].noise_label_set), \
                                                                                self.config['hidden_dim']*sum(self.config['bin_num']))).to(feature.device)
        noise_class_center = self.config['train_source'].noise_class_center
        label_set = torch.unique(label)
        for l in label_set:
            l_key = int(l.data.cpu().numpy())
            if l_key in self.config['train_source'].all2noise_index_map.keys():
                l_mask = (label == l).bool()
                l_feature = feature[l_mask, :]
                l_mean_feature = torch.mean(l_feature, dim=0)
                noise_index = self.config['train_source'].all2noise_index_map[l_key]
                # if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                #     print('all_index={}, noise_index={}, num={}'.format(l, noise_index, l_feature.size(0)))
                noise_class_center[noise_index, :] = \
                    self.config['noise_momentum'] * noise_class_center[noise_index, :] + (1-self.config['noise_momentum']) * l_mean_feature
        self.config['train_source'].noise_class_center = noise_class_center

        norm_class_center = F.normalize(noise_class_center, dim=1)
        noise_class_sim = torch.matmul(norm_class_center, norm_class_center.transpose(0, 1))*self.config['noise_temperature']
        diag_mask = (torch.eye(len(self.config['train_source'].noise_label_set)) == 1).bool().to(feature.device)
        noise_class_sim = noise_class_sim.masked_fill(diag_mask, float('-inf'))
        noise_class_sim = F.softmax(noise_class_sim, dim=1)
        self.config['train_source'].noise_class_sim = noise_class_sim
        # print('noise_class_sim={}'.format(noise_class_sim))
        # print('max_noise_class_sim={}'.format(torch.max(noise_class_sim, dim=1)[0]))
        # print('min_noise_class_sim={}'.format(torch.min(noise_class_sim, dim=1)[0]))
        # print('sum_noise_class_sim={}'.format(torch.sum(noise_class_sim, dim=1)))

    def build_loss(self):
        if self.config['encoder_infonce_weight'] > 0:
            if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):              
                self.encoder_infonce_loss = GlInfoNCELoss(temperature=self.config['encoder_infonce_temperature'], \
                                                            pos_hard_mining=self.config['encoder_infonce_poshard'], neg_hard_mining=self.config['encoder_infonce_neghard'], \
                                                                weakshot=self.config['encoder_infonce_weakshot'], print_info=True).float().cuda()
            else:
                self.encoder_infonce_loss = GlInfoNCELoss(temperature=self.config['encoder_infonce_temperature'], \
                                                            pos_hard_mining=self.config['encoder_infonce_poshard'], neg_hard_mining=self.config['encoder_infonce_neghard'], \
                                                                weakshot=self.config['encoder_infonce_weakshot'], print_info=False).float().cuda() 
            if self.config['DDP']:
                self.encoder_infonce_loss = DistributedLossWrapperWithLabelTypeWithLabelOrigin(self.encoder_infonce_loss, dim=0)

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss = PartTripletLoss(self.config['encoder_triplet_margin'], \
                                                            weakshot=self.config['encoder_triplet_weakshot']).float().cuda()
            if self.config['DDP']:
                self.encoder_triplet_loss = DistributedLossWrapperWithLabelType(self.encoder_triplet_loss, dim=1)

        if self.config['encoder_adacont_weight'] > 0:
            self.encoder_adacont_loss = PartAdacontLoss(pos_thres=self.config['encoder_adacont_posthres'], neg_thres=self.config['encoder_adacont_negthres']).float().cuda() 
            if self.config['DDP']:
                self.encoder_adacont_loss = DistributedLossWrapperWithLabelType(self.encoder_adacont_loss, dim=1)
        

    def build_loss_metric(self):
        if self.config['encoder_infonce_weight'] > 0:
            self.encoder_infonce_loss_metric = [[]]

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss_metric = [[], []]

        if self.config['encoder_adacont_weight'] > 0:
            self.encoder_adacont_loss_metric = [[], [], [], []]

        self.total_loss_metric = []
    
    def build_optimizer(self):
        #lr and weight_decay
        base_lr = self.config['lr']
        base_weight_decay = self.config['weight_decay'] if base_lr > 0 else 0
        gl_lr = self.config['gl_lr'] if self.config['gl_lr'] is not None else base_lr
        gl_weight_decay = self.config['weight_decay'] if gl_lr > 0 else 0

        #params
        if self.config['encoder_infonce_weight'] > 0:
            gl_params_id = list()
            gl_params_id.extend(list(map(id, self.encoder.module.gl_norm.parameters())))
            base_params = filter(lambda p: id(p) not in gl_params_id, self.encoder.parameters())
            gl_params = filter(lambda p: id(p) in gl_params_id, self.encoder.parameters())
            tg_params =[{'params': base_params, 'lr': base_lr, 'weight_decay': base_weight_decay}, \
                        {'params': gl_params, 'lr': gl_lr, 'weight_decay': gl_weight_decay}]
        else:
            tg_params =[{'params': self.encoder.parameters(), 'lr': base_lr, 'weight_decay': base_weight_decay}]
 
        #optimizer
        if self.config['optimizer_type'] == 'SGD':
            self.optimizer = optim.SGD(tg_params, lr=self.config['lr'], weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
        elif self.config['optimizer_type'] == 'ADAM': #if ADAM set the first stepsize equal to total_iter
            self.optimizer = optim.Adam(tg_params, lr=self.config['lr'])
        if self.config['warmup']:
            self.scheduler = WarmupMultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])

        #AMP
        if self.config['AMP']:
            self.scaler = GradScaler()

    def fit(self):
        self.encoder.train()
        if self.config['restore_iter'] > 0:
            self.load(self.config['restore_iter'])
        else:
            self.config['restore_iter'] = 0

        train_loader = tordata.DataLoader(
            dataset=self.config['train_source'],
            batch_sampler=self.triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        train_label_set = list(self.config['train_source'].label_set)
        clean_train_label_set = list(self.config['train_source'].clean_label_set)
        #############################################################
        origin_train_label_set = list(set([l.split('_')[0] for l in train_label_set]))
        #############################################################
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, label, batch_frame in train_loader:
            #############################################################
            if self.config['DDP'] and self.config['restore_iter'] > 0 and \
                self.config['restore_iter'] % self.triplet_sampler.total_batch_per_world == 0:
                self.triplet_sampler.set_random_seed(self.triplet_sampler.random_seed+1)
            #############################################################
            self.optimizer.zero_grad()

            seq = self.np2var(seq).float()
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.asarray(target_label)).long()
            target_label_type = [1 if l in clean_train_label_set else -1 for l in label]
            target_label_type = self.np2var(np.asarray(target_label_type)).long()
            #############################################################
            target_label_origin = [origin_train_label_set.index(l.split('_')[0]) for l in label]
            target_label_origin = self.np2var(np.asarray(target_label_origin)).long()
            #############################################################
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            with autocast(enabled=self.config['AMP']):
                encoder_feature, encoder_gl_feature = self.encoder(seq, batch_frame, target_label)

            if self.config['noise_hard']:
                self.update_noise_center(encoder_gl_feature.view(encoder_gl_feature.size(0), -1).float().detach(), target_label.detach())

            loss = torch.zeros(1).to(encoder_feature.device)

            if self.config['encoder_infonce_weight'] > 0:
                infonce_loss_metric, gl_sim = \
                                self.encoder_infonce_loss(encoder_gl_feature.view(encoder_gl_feature.size(0), -1).float(), target_label, target_label_type, target_label_origin)
                loss += infonce_loss_metric.mean() * self.config['encoder_infonce_weight']
                self.encoder_infonce_loss_metric[0].append(infonce_loss_metric.mean().data.cpu().numpy())

            if self.config['encoder_triplet_weight'] > 0:
                encoder_triplet_feature = encoder_feature.float().permute(1, 0, 2).contiguous()
                triplet_label = target_label.unsqueeze(0).repeat(encoder_triplet_feature.size(0), 1)
                triplet_label_type = target_label_type.unsqueeze(0).repeat(encoder_triplet_feature.size(0), 1)
                triplet_loss_metric, nonzero_num = self.encoder_triplet_loss(encoder_triplet_feature, triplet_label, triplet_label_type)
                loss += triplet_loss_metric.mean() * self.config['encoder_triplet_weight']
                self.encoder_triplet_loss_metric[0].append(triplet_loss_metric.mean().data.cpu().numpy())
                self.encoder_triplet_loss_metric[1].append(nonzero_num.mean().data.cpu().numpy())

            if self.config['encoder_adacont_weight'] > 0:
                adacont_hp_loss_metric, adacont_hn_loss_metric, hp_nonzero_num, hn_nonzero_num = \
                                self.encoder_adacont_loss(encoder_triplet_feature, triplet_label, triplet_label_type, gl_sim.detach())
                loss += (adacont_hp_loss_metric.mean() + adacont_hn_loss_metric.mean()) * self.config['encoder_adacont_weight']
                self.encoder_adacont_loss_metric[0].append(adacont_hp_loss_metric.mean().data.cpu().numpy())
                self.encoder_adacont_loss_metric[1].append(adacont_hn_loss_metric.mean().data.cpu().numpy())
                self.encoder_adacont_loss_metric[2].append(hp_nonzero_num.mean().data.cpu().numpy())
                self.encoder_adacont_loss_metric[3].append(hn_nonzero_num.mean().data.cpu().numpy())

            self.total_loss_metric.append(loss.data.cpu().numpy())

            if loss > 1e-9:
                if self.config['AMP']:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:  
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                
            if self.config['restore_iter'] % 100 == 0:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    print(datetime.now() - _time1)
                    _time1 = datetime.now()
                    self.print_info()
                self.build_loss_metric()
            if self.config['restore_iter'] % 10000 == 0 or self.config['restore_iter'] == self.config['total_iter']:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    self.save()
            if self.config['restore_iter'] == self.config['total_iter']:
                break
            self.config['restore_iter'] += 1

    def print_info(self):
        print('iter {}:'.format(self.config['restore_iter']))

        def print_loss_info(loss_name, loss_metric, loss_weight, loss_info):
            print('{:#^30}: loss_metric={:.6f}, loss_weight={:.6f}, {}'.format(loss_name, np.mean(loss_metric), loss_weight, loss_info))

        if self.config['encoder_infonce_weight'] > 0:
            loss_name = 'Encoder InfoNCE'
            loss_metric = self.encoder_infonce_loss_metric[0]
            loss_weight = self.config['encoder_infonce_weight']
            loss_info = 'temperature={}, pos_hard_mining={}, neg_hard_mining={}, weakshot={}'.format( \
                self.config['encoder_infonce_temperature'], self.config['encoder_infonce_poshard'], self.config['encoder_infonce_neghard'], self.config['encoder_infonce_weakshot'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['encoder_triplet_weight'] > 0:
            loss_name = 'Encoder Triplet'
            loss_metric = self.encoder_triplet_loss_metric[0]
            loss_weight = self.config['encoder_triplet_weight']
            loss_info = 'nonzero_num={:.6f}, margin={}, weakshot={}'.format(np.mean(self.encoder_triplet_loss_metric[1]), \
                            self.config['encoder_triplet_margin'], self.config['encoder_triplet_weakshot'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['encoder_adacont_weight'] > 0:
            loss_name = 'Encoder Adacont'
            loss_metric = np.mean(self.encoder_adacont_loss_metric[0]) + np.mean(self.encoder_adacont_loss_metric[1])
            loss_weight = self.config['encoder_adacont_weight']
            loss_info = 'loss_metric_hp={:.6f}, loss_metric_hn={:.6f}, hp_nonzero_num={:.6f}, hn_nonzero_num={:.6f}, pos_thres={}, neg_thres={}'.format( \
                np.mean(self.encoder_adacont_loss_metric[0]), np.mean(self.encoder_adacont_loss_metric[1]), \
                np.mean(self.encoder_adacont_loss_metric[2]), np.mean(self.encoder_adacont_loss_metric[3]), \
                self.config['encoder_adacont_posthres'], self.config['encoder_adacont_negthres'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        print('{:#^30}: total_loss_metric={:.6f}'.format('Total Loss', np.mean(self.total_loss_metric)))
        
        #optimizer
        if self.config['encoder_infonce_weight'] > 0:
            print('{:#^30}: type={}, base_lr={:.6f}, base_weight_decay={:.6f}, gl_lr={:.6f}, gl_weight_decay={:.6f}'.format( \
                'Optimizer', self.config['optimizer_type'], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay'], \
                self.optimizer.param_groups[1]['lr'], self.optimizer.param_groups[1]['weight_decay']))
        else:
            print('{:#^30}: type={}, base_lr={:.6f}, base_weight_decay={:.6f}'.format( \
                'Optimizer', self.config['optimizer_type'], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay']))
        print('{:#^30}: num_label_set={}, num_clean_label_set={}, num_noise_label_set={}'.format( \
            'TrainDataSet', len(self.config['train_source'].label_set), len(self.config['train_source'].clean_label_set), len(self.config['train_source'].noise_label_set))) 
        print('{:#^30}: num_label_set={}, num_clean_label_set={}, num_noise_label_set={}'.format( \
            'TestDataSet', len(self.config['test_source'].label_set), len(self.config['test_source'].clean_label_set), len(self.config['test_source'].noise_label_set))) 
        print('{:#^30}: pid_fname={}, clean_batch_size={}, noise_batch_size={}, noise_split={}'.format( \
            'TrainDataLoader', self.config['pid_fname'], self.config['clean_batch_size'], self.config['noise_batch_size'], self.config['noise_split']))
        print('{:#^30}: noise_hard={}, noise_momentum={:.2f}, noise_temperature={:.6f}'.format( \
            'TrainDataLoader', self.config['noise_hard'], self.config['noise_momentum'], self.config['noise_temperature']))              
        sys.stdout.flush()

    def transform(self, flag, batch_size=1, feat_idx=0):
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source']
        self.config['sample_type'] = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        feature_list = list()
        view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
        seq_type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, label, batch_frame = x
            seq = self.np2var(seq).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            output = self.encoder(seq, batch_frame)
            feature = output[feat_idx]
            feature_list.append(feature.detach())             
            label_list += label

        return torch.cat(feature_list, 0), view_list, seq_type_list, label_list

    def collate_fn(self, batch):
        batch_size = len(batch)
        seqs = [batch[i][0] for i in range(batch_size)]
        label = [batch[i][1] for i in range(batch_size)]
        batch = [seqs, label, None]
        batch_frames = []
        if self.config['DDP']:
            gpu_num = 1
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)

        # generate batch_frames for next step
        for gpu_id in range(gpu_num):
            batch_frames_sub = []
            for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                if i < batch_size:
                    if self.config['sample_type'] == 'random':
                        batch_frames_sub.append(self.config['frame_num'])
                    elif self.config['sample_type'] == 'all':
                        batch_frames_sub.append(seqs[i].shape[0])
                    elif self.config['sample_type'] == 'random_fn':
                        frame_num = np.random.randint(self.config['min_frame_num'], self.config['max_frame_num'])
                        batch_frames_sub.append(frame_num)
            batch_frames.append(batch_frames_sub)
        if len(batch_frames[-1]) != batch_per_gpu:
            for i in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)

        # select frames from each seq 
        def select_frame(index):
            sample = seqs[index]
            frame_set = np.arange(sample.shape[0])
            frame_num = batch_frames[int(index / batch_per_gpu)][int(index % batch_per_gpu)]
            if len(frame_set) >= frame_num:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=False))
            else:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=True))
            return sample[frame_id_list, :, :]
        seqs = list(map(select_frame, range(len(seqs))))        

        # data augmentation
        def transform_seq(index):
            sample = seqs[index]
            return self.data_transforms(sample)
        if self.config['dataset_augment']:
            seqs = list(map(transform_seq, range(len(seqs))))  

        # concatenate seqs for each gpu if necessary
        if self.config['sample_type'] == 'random':
            seqs = np.asarray(seqs)                      
        elif self.config['sample_type'] == 'all' or self.config['sample_type'] == 'random_fn':
            max_sum_frames = np.max([np.sum(batch_frames[gpu_id]) for gpu_id in range(gpu_num)])
            new_seqs = []
            for gpu_id in range(gpu_num):
                tmp = []
                for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                    if i < batch_size:
                        tmp.append(seqs[i])
                tmp = np.concatenate(tmp, 0)
                tmp = np.pad(tmp, \
                    ((0, max_sum_frames - tmp.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
                new_seqs.append(np.asarray(tmp))
            seqs = np.asarray(new_seqs)  

        batch[0] = seqs
        if self.config['sample_type'] == 'all' or self.config['sample_type'] == 'random_fn':
            batch[-1] = np.asarray(batch_frames)
        
        return batch

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x)) 

    def save(self):
        os.makedirs(osp.join('checkpoint', self.config['model_name']), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], self.config['restore_iter'])))
        torch.save([self.optimizer.state_dict(), self.scheduler.state_dict()],
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], self.config['restore_iter'])))

    def load(self, restore_iter):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.encoder.load_state_dict(encoder_ckp)
        optimizer_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.optimizer.load_state_dict(optimizer_ckp[0])
        self.scheduler.load_state_dict(optimizer_ckp[1])  

    def init_model(self, init_model):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_state_dict = self.encoder.state_dict()
        ckp_state_dict = torch.load(init_model, map_location=map_location)
        init_state_dict = {k: v for k, v in ckp_state_dict.items() if k in encoder_state_dict}
        drop_state_dict = {k: v for k, v in ckp_state_dict.items() if k not in encoder_state_dict}
        print('#######################################')
        if init_state_dict:
            print("Useful Layers in Init_model for Initializaiton:\n", init_state_dict.keys())
        else:
            print("None of Layers in Init_model is Used for Initializaiton.")
        print('#######################################')
        if drop_state_dict:
            print("Useless Layers in Init_model for Initializaiton:\n", drop_state_dict.keys())
        else:
            print("All Layers in Init_model are Used for Initialization.")
        encoder_state_dict.update(init_state_dict)
        none_init_state_dict = {k: v for k, v in encoder_state_dict.items() if k not in init_state_dict}
        print('#######################################')
        if none_init_state_dict:
            print("The Layers in Target_model that Are *Not* Initialized:\n", none_init_state_dict.keys())
        else:
            print("All Layers in Target_model are Initialized")  
        print('#######################################')      
        self.encoder.load_state_dict(encoder_state_dict)    
