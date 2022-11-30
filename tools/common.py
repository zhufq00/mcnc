import random
import numpy as np
import torch
import yaml
import os
import datetime
from pathlib import Path

def format_time(elapsed):    
    elapsed_rounded = int(round((elapsed)))    
    return str(datetime.timedelta(seconds=elapsed_rounded))   

def split_fn(fn_list,_fn):
    flag = 0
    res = []
    for fn in fn_list:
        if flag == 1:
            res.append(fn)
        if fn == _fn:
            flag = 1
    return res

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

class Args():
    def __init__(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.args_dict = config
            for k, v in config.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
    
    def to_str(self):
        mstr = '\n'
        for k,v in self.args_dict.items():
            mstr += k +': '+str(v)+'\n'
        return mstr

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
