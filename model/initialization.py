import os
from copy import deepcopy
import numpy as np
from .data import load_data
from .model import Model

def initialize_data(config):
    print("Initializing data source...")
    train_source, test_source = load_data(config)
    print("Data initialization complete.")
    return train_source, test_source

def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    config['train_source'] = train_source
    config['test_source'] = test_source
    config['save_name'] = '{}_{}_{}_{}'.format(config['model_name'], '_'.join(config['dataset']), config['pid_num'], config['pid_shuffle'])
    m = Model(config)
    print("Model initialization complete.")
    return m

def initialization(config):
    print("Initialzing...")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    train_source, test_source = initialize_data(config)
    return initialize_model(config, train_source, test_source)