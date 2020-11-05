import sys
sys.path.append(".")
import numpy as np
import json
import logging
import os
import random
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler
from configs import Constants
from tasks.utils import truncate_pair, truncate_one, TaskConfig
from tasks.task import TaskPoor

logger = logging.getLogger(__name__)
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)

class Task(TaskPoor):
    def __init__(self,config):
        super().__init__(config)

    # def load_model(self, model_path ):
    #     return super().load_model_seq(model_path)

    def predict(self):
        preds=self.infer()
        result={}
        for i in range(len(self.test_dataset)):
            id,l,a,choice,q,context = self.test_dataset.doc[i]
            result[id]=preds[i]

        # label_map = {i: label for i, label in enumerate(self.labels)}
        with open(self.config.output_submit_file, "w") as writer:
            for id,l in result.items():
                json_d = { "id":id, "label":l  }
                writer.write(json.dumps(json_d,ensure_ascii=False) + '\n')

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
        self.items=None

    def load_file(self,input_file):
        with open(input_file) as f:
            doc0=f.readlines()
        doc=[]
        label_prob={}
        lens=[]
        for line in doc0:
            item=json.loads(line.strip())
            id,answer,choice,question,context=item['id'],item["answer"],item["choice"],item["question"],' '.join(item["context"])
            if "train" in input_file:
                random.shuffle(choice)
            label = 0
            for i in range(self.config.num_labels):
                if i >= len(choice):
                    choice.append("无效答案")
                if answer == choice[i]:
                    label = i
            # label_prob[l] = label_prob.get(l, 0) + 1
            doc.append([id,answer,choice,question,context,label])
            lens.append(len(question)+len(answer)+len(context))
        long=max(lens)  #1577
        print(f" longest {long} ")
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        tokenses,input_maskes,type_idses,lengthes=[],[],[],[]
        if self.config.task_name=="c3":
            id,answer,choice,question,context,label= self.doc[idx]
            for i in range(self.config.num_labels):
                option=choice[i]
                a,q,c=self.tokenizer.tokenize(option),self.tokenizer.tokenize(question),self.tokenizer.tokenize(context)
                q=[Constants.TOKEN_BOQ] + q + [Constants.TOKEN_EOQ]
                a=[Constants.TOKEN_BOA] + a + [Constants.TOKEN_EOA]
                c=truncate_one(c,max_len=self.max_tokens-3-len(a)-len(q))
                c=[Constants.TOKEN_BOC] + c + [Constants.TOKEN_EOC]
                tokens=[Constants.TOKEN_CLS]+q+a+c
                length=len(tokens)

                tokens+=[Constants.TOKEN_PAD]*(self.max_tokens-length)
                type_ids = []
                k = 0
                for j, c in enumerate(tokens):
                    type_ids.append(k)
                    if c in [Constants.TOKEN_EOQ, Constants.TOKEN_EOA]:
                        k += 1
                tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask=[1]*length+[0]*(self.max_tokens-length)
                tokenses.append(tokens)
                input_maskes.append(input_mask)
                type_idses.append(type_ids)
                lengthes.append(length)
        tokenses, input_maskes, type_idses, lengthes= [np.array(x) for x in [tokenses,input_maskes,type_idses,lengthes]]
        return  tokenses,input_maskes,type_idses,lengthes,label

def preprocess0(src1,src2,target):
    with open(src1) as f:
        data=json.load(f)
    if src2:
        with open(src2) as f:
            data+=json.load(f)
    doc=[]
    for item in data:
        context=item[0]
        questions=item[1]
        for qa in questions:
            q=qa["question"]
            choice=qa["choice"]
            answer=qa.get("answer","")
            # id=qa.get('id',q+'_'+c)
            id=qa.get('id','id_'+q)

            example={ "id":id,"answer":answer,"choice":choice,"question":q,"context":context }
            doc.append(json.dumps(example,ensure_ascii=False))

    with open(target,"w") as f:
        f.writelines('\n'.join(doc))
    logger.info(f" -->{target} ")

def preprocess(datadir):
    src1=os.path.join(datadir,"m-train.json")
    src2 = os.path.join(datadir, "d-train.json")
    target=datadir+'/train.txt'
    preprocess0(src1,src2,target)

    src1=os.path.join(datadir,"m-dev.json")
    src2 = os.path.join(datadir, "d-dev.json")
    target=datadir+'/dev.txt'
    preprocess0(src1,src2,target)

    src1=os.path.join(datadir,"test.json")
    src2 =""
    target=datadir+'/test.txt'
    preprocess0(src1,src2,target)


if __name__ == "__main__":

    description="多选阅读理解"
    # labels =  ["0", "1"]
    labels =  ["0", "1","2","3"]

    config = {
        # "model_type": "albert",
        # "model_name_or_path": outputs + model_name,
        "task_name": "c3",
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        # "output_dir": outputs + f"{model_name}/task_output",
        # "max_len": 1024,
        "batch_size":2,
        # "num_workers":0,
        # "learning_rate": 5e-5,
        # "logging_steps": 100,
        # "save_steps": 1000,
        "train_file":"train.txt",
        "valid_file":"dev.txt",
        "test_file":"test.txt",
        "TaskDataset":TaskDataset,
        "num_labels":len(labels),
        "labels":labels
        # "per_gpu_train_batch_size": 16,
        # "per_gpu_eval_batch_size": 16,
    }
    # taskConfig=TaskConfig(config)
    # preprocess(taskConfig.data_dir)

    task=Task(config)
    task.train()
    task.predict()
