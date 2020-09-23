import sys
sys.path.append("..")
sys.path.append(".")
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler

from callback.progressbar import ProgressBar
from configs import Constants
from tasks.utils import truncate_pair, TaskConfig, collate_fn, truncate_one, index_of, find_span
from tasks.task import TaskPoor

logger = logging.getLogger(__name__)
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)

class Task(TaskPoor):
    def __init__(self,config):
        super().__init__(config)


    def load_model(self,model_path):
        return super().load_model_token(model_path)

    def predict(self):

        preds=self.infer()
        result={}
        for i, pred in enumerate(preds):
            id, a, q, c = self.test_dataset.doc[i]
            context = self.tokenizer.tokenize(c)
            # start, end = pred
            start, end = find_span(pred)

            label = context[start - 2:end - 1]
            result[id]=''.join(label)

        with open(self.config.output_submit_file, "w") as writer:
            writer.write(json.dumps(result, indent=4, ensure_ascii=False) + "\n")

        logger.info(f" test : {len(preds)}  examples  --> {self.config.output_submit_file}")

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
            item=json.loads(line.strip())
            id,a,q,c=item['id'] ,item["answer"],item["question"],item["context"]
            doc.append([ id,a,q,c])
            lens.append(len(q)+len(c))
        long=max(lens)  #1016
        print(f" longest {long} ")
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        items=self.doc[idx]
        if self.config.task_name=="cmrc":
            id, a, q, c =items
            [a,q,c]=[ self.tokenizer.tokenize(x) for x in (a,q,c) ]
            start=index_of(c,a)+2
            q = [Constants.TOKEN_BOQ] + q + [Constants.TOKEN_EOQ]
            c = [Constants.TOKEN_BOC] + c + [Constants.TOKEN_EOC]
            tokens=[Constants.TOKEN_CLS]+c+q
            # start+=len(q)+2
            label=[0]*(self.max_tokens)
            for i in range(len(a)):
                label[start+i]=1
            label=np.array(label)

        # label=self.label2idx[l]

        length=len(tokens)

        tokens+=[Constants.TOKEN_PAD]*(self.max_tokens-length)
        type_ids = []
        k = 0
        for j, c in enumerate(tokens):
            type_ids.append(k)
            if c in [Constants.TOKEN_EOQ,Constants.TOKEN_EOA]:
                k += 1
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask=[1]*length+[0]*(self.max_tokens-length)
        # return  np.array(tokens) , np.array(input_mask),np.array(type_ids) , length,label
        return  (tokens) , (input_mask), (type_ids) , length,label

def preprocess0(src1,src2,target):
    with open(src1) as f:
        data=json.load(f)["data"]
    if src2:
        with open(src2) as f:
            data+=json.load(f)["data"]
    doc=[]
    for item in data:
        item=item["paragraphs"][0]
        context=item["context"]
        questions=item["qas"]

        for qa in questions:
            id = qa["id"]
            q=qa["question"]
            answer=""
            for a in qa["answers"]:
                if len(a["text"])>len(answer):
                    answer=a["text"]
            example={ "id":id,"answer":answer,"question":q,"context":context}
            doc.append(json.dumps(example,ensure_ascii=False))

    with open(target,"w") as f:
        f.writelines('\n'.join(doc))
    logger.info(f" -->{target} ")

def preprocess(datadir):
    src1=os.path.join(datadir,"train.json")
    src2 = os.path.join(datadir, "trial.json")
    target=datadir+'/train.txt'
    preprocess0(src1,src2,target)

    src1=os.path.join(datadir,"dev.json")
    src2 = ""
    target=datadir+'/dev.txt'
    preprocess0(src1,src2,target)

    src1=os.path.join(datadir,"test.json")
    src2 =""
    target=datadir+'/test.txt'
    preprocess0(src1,src2,target)


if __name__ == "__main__":
    task_name="cmrc"
    description="抽取式阅读理解"
    labels =  ["0", "1"]
    config = {
        "output_mode": "TokenClassification" ,
        # "model_type": "albert",
        # "model_name_or_path": outputs + model_name,
        "task_name": task_name,
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        "max_len": 1024,
        "batch_size":7,
        # "output_dir": outputs + f"{model_name}/task_output",
        # "learning_rate": 5e-5,
        # "n_epochs": 10,
        # "logging_steps": 100,
        # "save_steps": 1000,
        # "num_workers" : 0,
        "train_file":"train.txt",
        "valid_file":"dev.txt",
        "test_file":"test.txt",
        # "output_submit_file":"cmrc2018_prediction.json",
        "TaskDataset":TaskDataset,
        "labels":labels
        # "per_gpu_train_batch_size": 16,
        # "per_gpu_eval_batch_size": 16,
    }
    # preprocess(data_dir + task_name)

    task=Task(config)
    task.train()
    task.predict()
