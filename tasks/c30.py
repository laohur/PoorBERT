import sys

sys.path.append("..")
sys.path.append(".")
import json
import logging
import os
import random
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler
from configs import Constants
from tasks.utils import truncate_pair, truncate_one
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
        for i,pred in enumerate(preds):
            id, a, q, c, l = self.test_dataset.doc[i]
            # label=str(label_map[preds[i]])
            if id not in result:
                result[id]=[]
            result[id].append(pred)

        label_map = {i: label for i, label in enumerate(self.labels)}
        with open(self.config.output_submit_file, "w") as writer:
            for k,vs in result.items():
                l=0
                for i in range(len(vs)):
                    if vs[i]==1:
                        l=i
                json_d = { "id":k, "label":l  }
                writer.write(json.dumps(json_d) + '\n')

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
            id,l,a,q,c=item['id'],item["label"],item["answer"],item["question"],item["context"]
            c= "`".join(c)

            l=item.get("label",self.labels[0])
            if l not in self.labels:
                logger.warn(f" error label {line} ")
                continue
            doc.append([id,a,q,c,l])
            label_prob[l] = label_prob.get(l, 0) + 1
            lens.append(len(q)+len(a)+len(c))
        long=max(lens)  #1578
        print(f" longest {long} ")
        if "train" in input_file:
            doc+=self.rebanlance(doc,label_prob)
        label_prob1 = {}
        for item in doc:
            l = item[-1]
            label_prob1[l] = label_prob1.get(l, 0) + 1
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        items=self.doc[idx]
        if self.config.task_name=="c30":
            id, a, q, c, l=items
            a,q,c=self.tokenizer.tokenize(a),self.tokenizer.tokenize(q),self.tokenizer.tokenize(c)
            q=[Constants.TOKEN_BOQ] + q + [Constants.TOKEN_EOQ]
            a=[Constants.TOKEN_BOA] + a + [Constants.TOKEN_EOA]
            c=truncate_one(c,max_len=self.max_tokens-3-len(a)-len(q))
            c=[Constants.TOKEN_BOC] + c + [Constants.TOKEN_EOC]
            tokens=[Constants.TOKEN_CLS]+q+a+c

        label=self.label2idx[l]
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
        return  (tokens) , (input_mask), (type_ids) , length,label

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
            for c in choice:
                # id=qa.get('id',q+'_'+c)
                id=qa.get('id',q+'_')

                if c==answer:
                    label='1'
                else:
                    label='0'
                example={ "id":id,"label":label,"answer":c,"question":q,"context":context }
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
    labels =  ["0", "1"]
    # labels =  ["0", "1","2","3"]

    config = {
        # "model_type": "albert",
        # "model_name_or_path": outputs + model_name,
        "task_name": "c30",
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        # "output_dir": outputs + f"{model_name}/task_output",
        # "max_len": 1024,
        # "batch_size":8,
        # "learning_rate": 5e-5,
        # "logging_steps": 100,
        # "save_steps": 1000,
        "train_file":"train.txt",
        "valid_file":"dev.txt",
        "test_file":"test.txt",
        "TaskDataset":TaskDataset,
        "labels":labels
        # "per_gpu_train_batch_size": 16,
        # "per_gpu_eval_batch_size": 16,
    }
    # taskConfig=TaskConfig(config)
    # preprocess(taskConfig.data_dir)

    task=Task(config)
    task.train()
    task.predict()
