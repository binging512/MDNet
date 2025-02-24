import os
from shutil import copyfile
import argparse
import yaml
from yacs.config import CfgNode
from train_2d import train_2d
from train_3d import train_3d
from test_2d import test_2d
from test_3d import test_3d
from utils.logger import Logger
from mmengine.config import Config
import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告

def str2bool(v):
    if v == "True":
        return True
    else:
        return False


def run(args):
    cfg = Config.fromfile(args.config)
    cfg.dump(os.path.join(cfg.workspace, os.path.basename(args.config)))
    
    os.makedirs(cfg.workspace,exist_ok=True)
    os.makedirs(os.path.join(cfg.workspace, cfg.results_val), exist_ok=True)
    os.makedirs(os.path.join(cfg.workspace, cfg.results_test), exist_ok=True)
    # copyfile(args.config, os.path.join(cfg.workspace, os.path.basename(args.config)))
    logger =Logger(cfg)
    logger.info(cfg)
    if args.train_2d_pass == True:
        logger.info("Starting training 2d pass....")
        train_2d(cfg, logger)

    if args.test_2d_pass == True:   # Not available
        logger.info("Starting testing 2d pass....")
        test_2d(cfg, logger)
        
    if args.train_3d_pass == True:
        logger.info("Starting training 3d pass....")
        train_3d(cfg, logger)
        
    if args.test_3d_pass == True:
        logger.info("Starting testing 3d pass....")
        test_3d(cfg, logger)
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser("Pancls training argument parser.")
    parser.add_argument('--config', default='configs/debug_3d.yaml', type=str)
    parser.add_argument('--base_config', default='configs/base_3d.yaml', type=str)
    parser.add_argument("--pretrain_2d_pass", default=False, type=str2bool)
    parser.add_argument("--train_2d_pass", default=False, type=str2bool)
    parser.add_argument("--test_2d_pass", default=False, type=str2bool)
    parser.add_argument("--train_3d_pass", default=False, type=str2bool)
    parser.add_argument("--test_3d_pass", default=False, type=str2bool)
    args = parser.parse_args()
    run(args)