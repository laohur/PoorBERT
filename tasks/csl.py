import random
import sys
sys.path.append(".")
from tasks import utils
import json
import logging
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler
from configs import Constants
from tasks.utils import truncate_pair, TaskConfig, truncate_one
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
            keywords=item["keyword"]  # acc:0.5598404255319149
            random.shuffle(keywords) # 0.6505984042553191+
            a= "|".join(keywords)
            b=item["abst"].strip()

            l=item.get("label",self.labels[0])
            # if l not in self.labels:
            #     logger.warn(f" error label {line} ")
            #     continue
            # a, b, l = a.strip(), b.strip(), l.strip()
            doc.append([a,b,l])
            label_prob[l] = label_prob.get(l, 0) + 1
            lens.append(len(a)+len(b))
        long=max(lens)  #1545
        print(f" longest {long} label_prob {label_prob}")
        if "train" in input_file:
            doc += utils.rebanlance(doc, label_prob)
        label_prob1 = {}
        for item in doc:
            l = item[-1]
            label_prob1[l] = label_prob1.get(l, 0) + 1
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        if self.config.task_name=="csl":
            a,b,l=self.doc[idx]
            senta = self.tokenizer.tokenize(a)
            sentb = self.tokenizer.tokenize(b,noise=self.config.noise)
            a,b=truncate_pair(senta,sentb,max_len=self.max_tokens-5)
            # tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + b + [Constants.TOKEN_EOS]  #acc:0.5113031914893617 acc:0.5478723404255319
            tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a + [Constants.TOKEN_EOS] + ["unsued3"] + b + ["unsued3"]  # acc:0.515625 acc:0.5598404255319149

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
        return  (tokens) , (input_mask),(type_ids) , length,label


if __name__ == "__main__":
    task_name="csl"
    description="论文关键词识别 "
    labels =  ["0", "1"]
    config = {
        # "model_type": "albert",
        # "model_name_or_path": outputs + model_name,
        "task_name": task_name,
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        # "max_len": 1024,
        # "batch_size":8,
        # "num_workers":4,
        # "learning_rate": 5e-5,
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