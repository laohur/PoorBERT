import json
import logging
import os
import random
import sys

sys.path.append("..")
sys.path.append(".")
import numpy as np
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler
from configs import Constants
from tasks.utils import truncate_pair, TaskConfig, truncate_one, index_of, find_span
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
            id, left, right, candidates, label= self.test_dataset.doc[i]
            result[id]=preds[i]

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
        logger.info(f" 装载{input_file}完毕{self.total_lines}")

    def load_file(self,input_file):
        with open(input_file) as f:
            doc0=f.readlines()
        doc=[]
        label_prob={}
        lens=[]
        for line in doc0:
            item=json.loads(line.strip())
            id, answer, left, right, candidates= item["id"],item["answer"],item["left"],item["right"], item["candidates"]
            if "train" in input_file:
                random.shuffle(candidates)
            label=0
            for i in range(self.config.num_labels):
                if i >= len(candidates):
                    candidates.append("无效答案")
                if candidates[i]==answer:
                    label=i
            doc.append([id, left, right, candidates,label ])
            lens.append(len(left)+len(right)+len(candidates[label]))
        long=max(lens)  #606
        print(f" longest {long} ")
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        tokenses,input_maskes,type_idses,lengthes=[],[],[],[]
        if self.config.task_name=="chid":
            id, left, right, candidates,label=self.doc[idx]
            for i in range(self.config.num_labels):
                [ idiom, l, r]=[ self.tokenizer.tokenize(x) for x in ( candidates[i], left, right) ]
                tokens=[Constants.TOKEN_CLS]+[Constants.TOKEN_BOS]+idiom+[Constants.TOKEN_EOS,Constants.TOKEN_BOS]+l+[Constants.TOKEN_MASK]*4+r+[Constants.TOKEN_EOS]
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
                tokenses.append(tokens)
                input_maskes.append(input_mask)
                type_idses.append(type_ids)
                lengthes.append(length)
        tokenses, input_maskes, type_idses, lengthes= [np.array(x) for x in [tokenses,input_maskes,type_idses,lengthes]]
        return  tokenses,input_maskes,type_idses,lengthes,label

def seg(line):
    segs={}
    for i in range(len(line)):
        if line[i:i+6]=="#idiom":
            segs[line[i:i+13]]=( line[:i], line[i+13:]  )
    return segs

def preprocess0(src1,src2,target):
    with open(src1) as f:
        doc1=f.readlines()
    if src2:
        with open(src2) as f:
            ans=json.load(f)
    doc=[]
    for line in doc1:
        item=json.loads(line)
        candidates=item["candidates"]
        content=item["content"]
        for sent in content:
            segs=seg(sent)
            for k, v in segs.items():
                answer = ""
                if src2:
                    answer = candidates[ans[k]]
            example={ "id":k,'answer':answer,"left":v[0],"right":v[1],"candidates":candidates}
            doc.append(json.dumps(example,ensure_ascii=False))

    with open(target,"w") as f:
        f.writelines('\n'.join(doc))
    logger.info(f" {len(doc)} examples -->{target} ")

def preprocess(datadir):
    src1=os.path.join(datadir,"train.json")
    src2 = os.path.join(datadir, "train_answer.json")
    target=datadir+'/train.txt'
    preprocess0(src1,src2,target)

    src1=os.path.join(datadir,"dev.json")
    src2 = os.path.join(datadir,"dev_answer.json")
    target=datadir+'/dev.txt'
    preprocess0(src1,src2,target)

    src1=os.path.join(datadir,"test.json")
    src2 =""
    target=datadir+'/test.txt'
    preprocess0(src1,src2,target)


if __name__ == "__main__":

    task_name="chid"
    description="成语阅读理解填空"
    labels =  ["0", "1"]
    config = {
        # "model_type": "albert",
        # "model_name_or_path": outputs + model_name,
        "task_name": task_name,
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        # "output_dir": outputs + f"{model_name}/task_output",
        # "max_len": 1024,
        "batch_size":1,
        # "num_workers":0,
        # "learning_rate": 5e-5,
        # "logging_steps": 50000,
        # "save_steps": 50000,
        "train_file":"train.txt",
        "valid_file":"dev.txt",
        "test_file":"test.txt",
        "TaskDataset":TaskDataset,
        "num_labels": 10,
        "labels":labels
        # "per_gpu_train_batch_size": 16,
        # "per_gpu_eval_batch_size": 16,
    }

    # taskConfig=TaskConfig(config)
    # preprocess(taskConfig.data_dir)

    task=Task(config)
    task.train()
    task.predict()
