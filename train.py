import torch
import numpy as np
import random

from model.initialization import initialization

from config import *
config.update({'phase':'train'})
m = initialization(config)
print('#######################################')
print("Network Structures:", m.encoder)
print('#######################################')

if config['init_model']:
    m.init_model(config['init_model'])
else:
    print('#######################################')
    print("Init Model is None")
    print('#######################################')
print("Training START")
m.fit()
print("Training COMPLETE")
