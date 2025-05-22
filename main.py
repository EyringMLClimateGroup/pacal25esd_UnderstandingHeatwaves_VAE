import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import time
import pandas as pd
from itertools import product
from utils.misc import read_config, get_args, start_logger

import train
import val
import test
import grid_search

 
    
def main():
    
    global DEVICE, logger
    args = get_args()
    
    cfg = read_config(args.config)
    if cfg['mode'] == 'test':
        cfg = read_config(os.path.join(cfg['test']['checkpoint_dir'],  cfg['test']['load_model'], 'main.yaml'))
        cfg['mode'] = 'test'
        
    if cfg["train"]["load_checkpoint"]:
        current_time = cfg["train"]["checkpoint_dir"].split("/")[-2]
        logger = start_logger(cfg['logger']['name'], log_file=cfg['logger']['file'].format(current_time))
        logger.info("Loaded checkpoint from {}".format(cfg["train"]["checkpoint_dir"]))
    
    elif cfg['mode'] == 'test':
        current_time = cfg['test']['load_model']
        logger = start_logger(cfg['logger']['name'], log_file=cfg['logger']['file'].format(current_time))
        logger.info("Loaded checkpoint from {}".format(cfg['test']['checkpoint_dir']))
    else:
        logger = start_logger(cfg['logger']['name'], log_file=cfg['logger']['file'].format(time.strftime("%y%m%d_%H%M%S")))
    
    for module in [train, test, val, grid_search]:
        module.set_logger(logger)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("✅ Configuration loaded and logger started.")
    logger.info("Device: {}".format(DEVICE))

    # run the selected mode    
    if cfg['mode'] == 'train':
        train.main(cfg, DEVICE)
    elif cfg['mode'] == 'val':
        val.main(cfg, DEVICE)
    elif cfg['mode'] == 'test':
        test.main(cfg, DEVICE)
    elif cfg['mode'] == 'grid_search':
        grid_search.main(cfg, DEVICE)
    else:
        raise ValueError("Invalid mode. Expected one of ['train', 'val', 'test', 'grid_search'] but got {}".format(cfg['mode']))
        
    logger.info("🏅 Done!")
    
        
if __name__ == "__main__":
    main()