import json
import logging
import random
import sys
sys.path.append("..")
sys.path.append(".")
from tasks import utils
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler
from configs import Constants
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
        label_map = {i: label for i, label in enumerate(self.labels)}
        with open(self.config.output_submit_file, "w") as writer:
            for i, pred in enumerate(preds):
                label=str(label_map[pred])
                json_d = { "id":i, "label":label  }
                # json_d = { "id":i, "label":label,"label_desc":label_descs[label]  }
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
            text=item["text"]
            # a, b=item["text"],item["keywords"]
            # tokens=list(a)
            segs=[]
            for i in ["1","2"]:
                m = item["target"][f"span{i}_index"]
                n = m + len(item["target"][f"span{i}_text"])
                # tokens[m]="《"+tokens[m]
                # tokens[n]= tokens[n]+"》"
                segs+=[m,n]
            if segs[0]>segs[2]:
                segs=segs[2:]+segs[:2]

            a,b,c,d,e=text[:segs[0]],text[segs[0]:segs[1]],text[segs[1]:segs[2]],text[segs[2]:segs[3]],text[segs[3]:]
            # a=''.join(tokens)

            l=item.get("label",self.labels[0])
            if l not in self.labels:
                logger.warn(f" error label {line} ")
                continue
            # a, l = a.strip(), l.strip()
            doc.append([a,b,c,d,e,l])
            # a, b, l = a.strip(), b.strip(), l.strip()
            # doc.append([a,b,l])
            label_prob[l] = label_prob.get(l, 0) + 1
            lens.append(len(a)+len(b)+len(c)+len(d)+len(e))
        long=max(lens)  #142
        print(f"longest {long}")
        if "train" in input_file:
            doc += utils.rebanlance(doc, label_prob)
        label_prob1={}
        for item in doc:
            l=item[-1]
            label_prob1[l]=label_prob1.get(l,0)+1
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        if self.config.task_name=="wsc":
            a,b,c,d,e,l=self.doc[idx]
            a,b,c,d,e = self.tokenizer.tokenize(a,noise=self.config.noise),self.tokenizer.tokenize(b),self.tokenizer.tokenize(c,noise=self.config.noise),self.tokenizer.tokenize(d),self.tokenizer.tokenize(e,noise=self.config.noise)
            tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a +["unsued1"]+b+["unsued1"]+c+["unsued3"]+d+["unsued3"]+e+ [Constants.TOKEN_EOS]

        label=self.label2idx[l]

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
        return  tokens , input_mask,type_ids , length,label


if __name__ == "__main__":

    task_name="wsc"
    description="代词消歧"
    labels =  ["false","true"]

    config = {
        # "model_type": "albert",
        # "model_name_or_path": outputs + model_name,
        "task_name": task_name,
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        # "output_dir": outputs + f"{model_name}/task_output",
        # "max_len": 256,
        "batch_size":16,
        "n_epochs":50,
        # "num_workers":4,
        # "learning_rate": 2e-5,
        # "logging_steps": 100,
        # "save_steps": 1000,
        "train_file":"train.json",
        "valid_file":"dev.json",
        "test_file":"test.json",
        "TaskDataset":TaskDataset,
        "labels":labels
        # "per_gpu_train_batch_size": 16,
        # "per_gpu_eval_batch_size": 16,
    }
    # torch.multiprocessing.set_start_method('spawn')
    task=Task(config)
    task.train()
    task.predict()