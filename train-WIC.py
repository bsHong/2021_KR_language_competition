# CUDA_VISIBLE_DEVICES=7 python3 train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
# from time import time
import time
from tqdm import tqdm
import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from transformers import T5Tokenizer, T5EncoderModel

import model

import easydict

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

model_config = { 'num_labels' : 2}
MAX_LEN = 512

config_path = 'config/train_WIC.json'
with open(config_path) as json_data:
    args = easydict.EasyDict(
                            json.load(json_data)
    )
    print("[Parameter]\n {}".format(args))
json_data.close()

USE_CUDA = torch.cuda.is_available()

device = torch.device('cuda' if USE_CUDA else 'cpu')
print('Use device : {}'.format(device))
model_name = 'KETI-AIR/ke-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)

df_train=pd.read_csv('/root/WIC/00-data/mark_NIKL_SKT_WiC_Train.csv')
df_val=pd.read_csv('/root/WIC/00-data/mark_NIKL_SKT_WiC_Dev.csv')

def T5_tokenizer(tokenizer, sent1, sent2, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(
                        sent1, sent2,
                        add_special_tokens=True, 
                        max_length=MAX_LEN, 
                        pad_to_max_length=True, 
                        return_attention_mask=True,
                        truncation = True)
#     print(encoded_dict['input_ids'])
#     print(tokenizer.decode(encoded_dict['input_ids']))
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    return input_id, attention_mask

def clean_text(sent):
    sent_clean=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]", " ", sent)
    return sent_clean

train_input_ids =[]
train_attention_masks =[]
train_data_labels = []

for idx, data in tqdm(df_train.iterrows()):
    train_input_id, train_attention_mask = T5_tokenizer(tokenizer, data['SENTENCE1'], data['SENTENCE2'], MAX_LEN)
    train_input_ids.append(train_input_id)
    train_attention_masks.append(train_attention_mask)
    
train_data_labels = df_train['ANSWER'] * 1
val_input_ids =[]
val_attention_masks =[]
val_data_labels = []

for idx, data in tqdm(df_val.iterrows()):
    val_input_id, val_attention_mask = T5_tokenizer(tokenizer, data['SENTENCE1'], data['SENTENCE2'], MAX_LEN)

    val_input_ids.append(val_input_id)
    val_attention_masks.append(val_attention_mask)
    
val_data_labels = df_val['ANSWER'] * 1

train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_attention_masks = torch.tensor(train_attention_masks, dtype=torch.long)
train_data_labels = torch.tensor(train_data_labels.values)
# train_data_labels = torch.tensor(train_data_labels, dtype=torch.long)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_data_labels)
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset,
#                         sampler=train_sampler, 
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        pin_memory = False,
                        num_workers=7)

val_input_ids = torch.tensor(val_input_ids, dtype=torch.long)
val_attention_masks = torch.tensor(val_attention_masks, dtype=torch.long)
val_data_labels = torch.tensor(val_data_labels.values)

val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_data_labels)
val_sampler = RandomSampler(val_dataset)
val_loader = DataLoader(val_dataset,
                        sampler=val_sampler, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        pin_memory = False,
                        num_workers=7)

# model = Model.AnomalyDetectionAE(len(df_concat.columns)).to(device)
model = model.T5_WIC.from_pretrained(model_name, **model_config)
model = model.to(device)

for idx, (name, param) in enumerate(model.named_parameters()):
    if idx >65:
        param.requires_grad = True
    else:
        param.requires_grad = False
#     print(name, param.requires_grad)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

model_PATH = 'save_model/'

def train(model, train_loader, val_loader, optimizer, args):
# def train(model, train_loader, criterion, optimizer, args):
    best_val_acc = 0
    ''' AutoEncoder for Anomaly Detection  '''
    print("Start training...\n")
    print(f"{'Elapsed':^9} | {'Epoch':^7} | {'Step':^7} | {'Train loss':^12} | {'Train acc':^12} | {'Val loss':^10} | {'Val Acc':^9} ")
    print("-"*85)
    best_val_loss = 10000
    early_stop_cnt = 0
    n_batch_check = len(train_loader) // 5
    for epoch in range(args.epochs):
        tr_acc = 0
        tr_recall = 0
        tr_precision = 0
        tr_f1 = 0
        nb_train_steps = 0 
        total_loss, batch_loss, batch_counts, batch_tr_acc = 0, 0, 0, 0
#         batch_time = Meters.AverageMeter()

        model.train()
        end_time = time.time()
        epoch_time = 0 
        for step, batch in enumerate(train_loader):
            batch_counts +=1
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask':batch[1].to(device),
                      'labels':batch[2].to(device)
                     }
            # optimizer gradient initialization
            model.zero_grad()
            
            # compute model : outputs = model(**inputs)
            outputs = model(**inputs)
            # loss function 
#             loss = model.loss_function(*outputs)
            loss = outputs[0]
            batch_loss += loss.item()
            total_loss += loss.item() # epoch loss..
            
            # update loss
            loss.backward()
            
            # gradient cliping
            # Adam과 같은 동적인 학습률을 가진 optim은 사용할 필요 X
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.5)

            optimizer.step()
    #         scheduler.step()
                
            logits = outputs[1]
        
            preds = torch.argmax(torch.nn.functional.log_softmax(logits,dim=1), dim=1)
            preds_list = preds.detach().cpu().numpy().tolist() # prediction result 1 array

            true_np = batch[2].detach().cpu().numpy()
            true_list = batch[2].detach().cpu().numpy().tolist()
            
            tr_acc += accuracy_score(true_list,preds_list)
            batch_tr_acc += accuracy_score(true_list,preds_list)
            tr_f1 += f1_score(true_list,preds_list, average='micro', zero_division=1,labels=np.unique(preds_list))
            tr_recall += recall_score(true_list,preds_list, average='micro', zero_division=1,labels=np.unique(preds_list))
            tr_precision += precision_score(true_list,preds_list, average='micro', zero_division=1,labels=np.unique(preds_list))
            nb_train_steps +=1
        
            if (step % n_batch_check == 0 and step != 0) or (step == len(train_loader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - end_time
                epoch_time += time_elapsed
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
                # Print training results
                print(f" {time_elapsed} | {epoch + 1:^7} | {step:^8}| {batch_loss / batch_counts:^12.6f} | {batch_tr_acc / batch_counts:^12.6f} | {'-':^11}| {'-':^11}")
                batch_loss, batch_counts, batch_tr_acc = 0, 0, 0
                end_time = time.time()
        
        
        epoch_time = time.strftime("%H:%M:%S", time.gmtime(epoch_time))
        avg_train_loss = total_loss / len(train_loader)
        
        ############################################
        #################EVALUATE###################
        ############################################
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_acc = 0
            val_recall = 0
            val_precision = 0
            val_f1 = 0
            nb_val_steps = 0
            for step, val_batch in enumerate(val_loader):
                val_inputs = {'input_ids': val_batch[0].to(device),
                              'attention_mask':val_batch[1].to(device),
                              'labels':val_batch[2].to(device)
                             }
                # compute model : outputs = model(**inputs)
                outputs = model(**val_inputs)

                loss = outputs[0]
                val_loss += loss.item()
                
                val_logits = outputs[1]
                
                val_preds = torch.argmax(torch.nn.functional.log_softmax(val_logits,dim=1), dim=1)
                val_preds_list = val_preds.detach().cpu().numpy().tolist() # prediction result 1 array

                val_true_np = val_batch[2].detach().cpu().numpy()
                val_true_list = val_batch[2].detach().cpu().numpy().tolist()

                val_acc += accuracy_score(val_true_list,val_preds_list)
                val_f1 += f1_score(val_true_list,val_preds_list, average='micro', zero_division=1,labels=np.unique(val_preds_list))
                val_recall += recall_score(val_true_list,val_preds_list, average='micro', zero_division=1,labels=np.unique(val_preds_list))
                val_precision += precision_score(val_true_list,val_preds_list, average='micro', zero_division=1,labels=np.unique(val_preds_list))
                nb_val_steps +=1
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc/len(val_loader)
#         avg_val_loss = 0
        print("-"*85)
        print(f" {time_elapsed} | {epoch + 1:^7} | {'-':^8}| {avg_train_loss:^12.6f} | {tr_acc/nb_train_steps:^12.6f} | {avg_val_loss:^11.6f}| {val_acc/nb_val_steps:^11.6}")
        print("-"*85)
        
#         print('tr_acc {}, tr_recall {}, tr_precision {}, tr_f1 {}'.format(tr_acc/nb_train_steps, tr_recall/nb_train_steps, tr_precision/nb_train_steps, tr_f1/nb_train_steps))
#         print('val_acc {}, val_f1 {}, val_precision {}, val_recall {}'.format(val_acc/nb_val_steps, val_f1/nb_val_steps, val_precision/nb_val_steps, val_recall/nb_val_steps))
        ############################################
        #################Early stop#################
        ############################################
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            early_stop_cnt = 0 
            
            # save best model
            # best_model = model
            # ... save weight..........
            print(avg_val_acc)
            torch.save(model, model_PATH + 'model_WIC.pt')
            
        else:
            early_stop_cnt +=1 
#             if args.early_stop and early_stop_cnt == 5:
#                 print('early stop condition   best_val_loss[{}]  avg_val_loss[{}]'.format(best_val_loss, avg_val_loss))
# #                 break
        
    print("Training complete!")
    print("Best Val acc = {}".format(best_val_acc))
    
    
train(model, train_loader, val_loader, optimizer, args)
