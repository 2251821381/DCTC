"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""
import csv
import os
import time
import numpy as np
from sklearn import cluster
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader
from models.Transformers import DCTC

from learner.losses import ContrastiveLoss

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss,ClusterLoss
from learner.losses import SupConLoss
from loss import CCLoss,DivClustLoss
class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        self.DivLoss = DivClustLoss()
        self.class_num=args.num_classes
        device=torch.device("cuda")
        self.cluster_loss =  ClusterLoss(self.class_num, 0.5, device).to(device)
        self.current_step=0
        self.gstep = 0
        self.ce_loss = nn.CrossEntropyLoss()
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()

    def target_l2(self,q):
        return ((q ** 2).t() / (q ** 2).sum(1)).t()
    def train_step_virtual(self, input_ids, attention_mask,target,i):
        
        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        G1=ContrastiveLoss().kernel_affinity(z=feat1,label=target)
        D_model = DCTC(self.model)
        G_loss= D_model((input_ids,attention_mask))

        G_loss=ContrastiveLoss()(feat1,feat2,G)
        clu1,clu2,clu3= self.model.cluster_logits(embd1, embd2,embd2)
        clustering_loss=self.cluster_loss(clu1,clu2)
        losses = self.contrast_loss(feat1, feat2)
        fea=torch.cat((feat1, feat2),dim=1)
        y=self.model.cluster_head(embd1)
        loss = G_loss * (1-self.eta)*clustering_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    

    def train_step_explicit(self, input_ids, attention_mask):
        
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        result=[]
        for i in np.arange(self.args.max_iter+1):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses = self.train_step_virtual(input_ids, attention_mask,batch["label"],i) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                # statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                print("loss:", losses)

                self.evaluate_embedding(i,result)
                self.model.train()

                # with open(self.args.dataname+".txt","a+",encoding="utf8") as f1:
                #     for item in result:
                #         line=",".join(str(x) for x in item  )+"\n"
                #         f1.writelines(line)

        return None   

    
    def evaluate_embedding(self, step,re):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))
        
        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
        # if self.args.dataname=="ag" and step%2==0:
            # with open ("ag"+str(step)+".csv“", 'w', newline='') as csvfile:
            #         emd=torch.cat([all_embeddings,all_labels.view(all_embeddings.size(0),-1).cuda()],dim=1)
            #         emb=emd.cpu().numpy()
            #         writer = csv.writer(csvfile)
            #         writer.writerows(emb)

        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes+2), Confusion(self.args.num_classes+2)
        
        all_pred = all_prob




        metric=self.clustering_accuracy_metrics(all_pred,   all_labels)


        return None

    def clustering_accuracy_metrics(self,cluster_labels, ground_truth):
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        if len(cluster_labels.shape) == 1:
            cluster_labels = np.expand_dims(cluster_labels, 0)

        cluster_labels = cluster_labels.astype(np.int64)
        ground_truth = ground_truth.astype(np.int64)
        assert cluster_labels.shape[-1] == ground_truth.shape[-1]

        metrics = {}
        cluster_accuracies, cluster_nmis, cluster_aris = [], [], []
        interclustering_nmi = []
        clusterings = len(cluster_labels)
        for k in range(clusterings):
            for j in range(clusterings):
                if j>k:
                    interclustering_nmi.append(np.round(normalized_mutual_info_score(cluster_labels[k], cluster_labels[j]), 5))
            cluster_accuracies.append(self.clustering_acc(cluster_labels[k], ground_truth))
            cluster_nmis.append(np.round(normalized_mutual_info_score(cluster_labels[k], ground_truth), 5))
            cluster_aris.append(np.round(adjusted_rand_score(ground_truth, cluster_labels[k]), 5))
            metrics["cluster_acc_" + str(k)] = cluster_accuracies[-1]
            metrics["cluster_nmi_" + str(k)] = cluster_nmis[-1]
            metrics["cluster_ari_" + str(k)] = cluster_aris[-1]
        metrics["max_cluster_acc"], metrics["mean_cluster_acc"], metrics["min_cluster_acc"] = np.max(
            cluster_accuracies), np.mean(cluster_accuracies), np.min(cluster_accuracies)
        metrics["max_cluster_nmi"], metrics["mean_cluster_nmi"], metrics["min_cluster_nmi"] = np.max(
            cluster_nmis), np.mean(cluster_nmis), np.min(cluster_nmis)
        metrics["max_cluster_ari"], metrics["mean_cluster_ari"], metrics["min_cluster_ari"] = np.max(
            cluster_aris), np.mean(cluster_aris), np.min(cluster_aris)
        if clusterings>1:
            metrics["interclustering_nmi"] = sum(interclustering_nmi)/len(interclustering_nmi)
        return metrics

    def clustering_acc(self,y_pred, y_true):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 100.0 / y_pred.size


             