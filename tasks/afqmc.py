import sys
sys.path.append(".")
from torch.utils.data.dataset import Dataset
import json
import logging
import random
# from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler
from configs import Constants
from tasks.task import TaskPoor
from tasks import utils

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
            doc+=utils.rebanlance(doc,label_prob)
        label_prob1 = {}
        for item in doc:
            l = item[-1]
            label_prob1[l] = label_prob1.get(l, 0) + 1
        label_prob1 = {}
        for item in doc:
            l = item[-1]
            label_prob1[l] = label_prob1.get(l, 0) + 1
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        item=self.doc[idx]
        a,b,l=item
        if random.random()<0.5:  #  # acc:0.6902006172839505
            a,b=b,a
        # acc:0.6103395061728395
        senta = self.tokenizer.tokenize(a,noise=self.config.noise)
        sentb = self.tokenizer.tokenize(b,noise=self.config.noise)
        label=self.label2idx[l]

        a,b=utils.truncate_pair(senta,sentb,max_len=self.max_tokens-5)
        tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + b + [Constants.TOKEN_EOS]  # acc:0.6902006172839505
        # tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a + [Constants.TOKEN_EOS] + ["unsued3"] + b + ["unsued3"]  #  acc:0.5960648148148148
        # tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOQ] + a + [Constants.TOKEN_EOQ] + [Constants.TOKEN_BOA] + b + [Constants.TOKEN_EOA]  # acc:0.5899691358024691

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
        # "max_len": 256,
        # "batch_size":16,
        # "num_workers":4,
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

