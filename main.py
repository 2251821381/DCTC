"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from models.Transformers import SCCLBert
import dataloader.dataloader as dataloader
from training import SCCLvTrainer
from utils.kmeans import get_kmeans_centers
from utils.logger import setup_path, set_global_random_seed
from utils.optimizer import get_optimizer, get_bert
import numpy as np


def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader(args)

    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)
    
    # initialize cluster centers
    # cluster_centers = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes+1, args.max_length)
    
    model = SCCLBert(bert, tokenizer,args.num_classes, alpha=args.alpha)
    model = model.cuda()

    # optimizer 
    optimizer = get_optimizer(model, args)
    
    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, args)
    trainer.train()
    
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local') 
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=10, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')
    parser.add_argument('--datapath', type=str, default='/home/janko/Code/sccl-main/AugData/augmented-datasets')
    parser.add_argument('--dataname', type=str, default='tweet-original-order_trans_subst_10', help="")
    parser.add_argument('--bert', type=str, default='BERT', help="")
    parser.add_argument('--use_pretrain', type=str, default='BERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    
    # Dataset

    parser.add_argument('--num_classes', type=int, default=89, help="")
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=2000)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='contrastive')
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=1, help="")
    
    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args



if __name__ == '__main__':
    import subprocess
    result=[]
    args = get_args(sys.argv[1:])
    lr =[1,2,3,45,6]
    for i in lr:

        args.lr=i
        main(args)
    data_path=[
        "F:\sccl-main\AugData\\augmented-datasets\\agnewsdataraw-8000_trans_subst_10.csv",
               # "F:\sccl-main\AugData\\augmented-datasets\\biomedical_trans_subst_10.csv",
               # "F:\sccl-main\AugData\\augmented-datasets\S_trans_subst_10.csv",
               # "F:\sccl-main\AugData\\augmented-datasets\searchsnippets_trans_subst_10.csv",
               # "F:\sccl-main\AugData\\augmented-datasets\stackoverflow_trans_subst_10.csv",
               # "F:\sccl-main\AugData\\augmented-datasets\T_trans_subst_10.csv",
               # "F:\sccl-main\AugData\\augmented-datasets\TS_trans_subst_10.csv",
               # "F:\sccl-main\AugData\\augmented-datasets\\tweet-original-order_trans_subst_10.csv"
               ]
    # class_num=[152,152,152]
    class_num = [ 4]

    # data_name=["S","T","TS"]
    data_name = ["ag"]
    for path,class_num ,name in zip(data_path,class_num,data_name):
        args.data_path=path
        args.num_classes=class_num
        args.dataname = name
        if args.train_instance == "sagemaker":
            run(args)
            subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_resdir])
        else:
            run(args)




    
