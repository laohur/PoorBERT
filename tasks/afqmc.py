import sys

sys.path.append("..")
sys.path.append(".")
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler

from callback.progressbar import ProgressBar
from configs import Constants
from tasks.utils import truncate_pair, collate_fn
from tasks.task import TaskPoor

logger = logging.getLogger(__name__)
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)

class Task(TaskPoor):
    def __init__(self,config):
        super().__init__(config)

    def load_model(self, model_path ):
        return super().load_model_seq(model_path)

    def predict(self):
        preds=self.infer()
        # 保存标签结果
        label_map = {i: label for i, label in enumerate(self.labels)}
        with open(self.config.output_submit_file, "w") as writer:
            for i, pred in enumerate(preds):
                json_d = { "id":i, "label":str(label_map[pred])  }
                writer.write(json.dumps(json_d) + '\n')

        logger.info(f" test : {len(preds)}  examples ")
        # model.train()

class TaskDataset(Dataset):
    def __init__(self,input_file,labels,tokenizer,max_tokens,config):
        super(TaskDataset, self).__init__()
        self.config=config
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.labels=labels
        self.label2idx =   {label:i for i, label in enumerate(labels)}
        self.input_file = input_file
        self.doc = self.load_file(input_file)
        self.total_lines =len(self.doc)
        logger.info(f"  装载{input_file}完毕{self.total_lines}")

    def load_file(self,input_file):
        with open(input_file) as f:
            doc0=f.readlines()
        doc=[]
        label_prob={}
        lens=[]
        for line in doc0:
            # sents=line.split('\t')
            item=json.loads(line.strip())
            a, b=item["sentence1"],item["sentence2"]
            l=item.get("label",self.labels[0])
            a, b, l = a.strip(), b.strip(), l.strip()
            doc.append([a,b,l])
            label_prob[l]=label_prob.get(l,0)+1
            lens.append(len(a)+len(b))
        long=max(lens)  #157
        print(f" longest {long} ")
        if "train" in input_file:
            doc+=self.rebanlance(doc,label_prob)
        label_prob1 = {}
        for item in doc:
            l = item[-1]
            label_prob1[l] = label_prob1.get(l, 0) + 1
        label_prob1 = {}
        for item in doc:
            l = item[-1]
            label_prob1[l] = label_prob1.get(l, 0) + 1
        return doc

    def rebanlance(self, doc, label_prob):
        for k in label_prob.keys():
            label_prob[k] = label_prob[k] * len(label_prob) / len(doc)
        expand = []
        for item in doc:
            label = item[-1]
            for i in range(4):
                if random.random() > label_prob[label]:
                    expand.append(item)
        return expand

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        item=self.doc[idx]
        a,b,l=item
        if random.random()<0.5:
            a,b=b,a

        senta = self.tokenizer.tokenize(a,noise=self.config.noise)
        sentb = self.tokenizer.tokenize(b,noise=self.config.noise)
        label=self.label2idx[l]

        a,b=truncate_pair(senta,sentb,max_len=self.max_tokens-5)
        tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + b + [Constants.TOKEN_EOS]
        length=len(tokens)
        tokens+=[Constants.TOKEN_PAD]*(self.max_tokens-length)
        type_ids = []
        k = 0
        for j, c in enumerate(tokens):
            type_ids.append(k)
            if c in [Constants.TOKEN_EOS]:
                k += 1
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask=[1]*length+[0]*(self.max_tokens-length)
        return  (tokens), (input_mask) ,(type_ids) , length,label


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model_name', default="none")
    args = parser.parse_args()
    model_name=args.model_name

    task_name="afqmc"
    description="句子相似度"
    # outputs = '/media/u/t1/dataset/Poor/outputs/'
    # data_dir = '/media/u/t1/dataset/CLUEdatasets/'
    config = {
        "task_name": task_name,
        "model_name":model_name,
        # "model_dir":outputs + model_name,
        # "model_name_or_path": outputs + model_name,
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        # "output_dir": outputs + f"{model_name}/task_output",
        "max_len": 256,
        "batch_size":50,
        # "learning_rate": 0.9e-5,
        # "n_epochs": 10,
        # "logging_steps": 100,
        # "save_steps": 1000,
        "train_file":"train.json",
        "valid_file":"dev.json",
        "test_file":"test.json",
        "TaskDataset":TaskDataset,
        "labels":["0", "1"]
        # "per_gpu_train_batch_size": 16,
        # "per_gpu_eval_batch_size": 16,
    }
    task=Task(config)
    task.train()
    task.predict()

