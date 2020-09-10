import json
import logging
import os
import random
import sys


sys.path.append("..")
sys.path.append(".")
import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler

from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.optimization.adamw import AdamW
from callback.progressbar import ProgressBar
from configs import Constants
from model.configuration_albert import AlbertConfig
from model.modeling_poor import AlbertForPreTraining, AlbertForSequenceClassification
from model.tokenization_shang import ShangTokenizer, Sentence
from tasks.utils import truncate_pair, TaskConfig, collate_fn, truncate_one
from tasks.task import TaskPoor

logger = logging.getLogger(__name__)
# FORMAT = '%(pathname)s %(filename)s  %(funcName)s %(lineno)d %(asctime)-15s  %(message)s'
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)

class Task(TaskPoor):
    def __init__(self,config):
        super().__init__(config)
        # self.config=TaskConfig(config)
        # self.task_name=self.config.task_name
        # self.dataset=TaskDataset
        # self.labels=self.config.labels
        #
        # if not os.path.exists(self.config.output_dir):
        #     os.makedirs(self.config.output_dir)
        #
        # self.tokenizer = ShangTokenizer(vocab_path=self.config.vocab_file, bujian_path=self.config.bujian_file)
        # self.model = self.load_model()
        # self.valid_dataset=self.load_valid()
    def load_model(self, model_path ):
        bert_config = AlbertConfig.from_pretrained(model_path,num_labels=len(self.labels),finetuning_task=self.task_name)
        logger.info(f" loadding {model_path} ")
        model = AlbertForSequenceClassification.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)
        model.to(self.config.device)
        return model

    def predict(self):
        args=self.config
        model= self.load_model(self.config.output_dir)
        model.eval()
        # dataset=self.valid_dataset
        input_file=os.path.join(self.config.data_dir,self.config.test_file)
        dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer,labels=self.labels, max_tokens=self.config.max_len,config=self.config)
        # msg={ "n_examples": len(dataset),  }
        # logger.info("  Num examples = %d", len(dataset))
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size,collate_fn=collate_fn,num_workers=self.config.num_workers)

        # Eval!
        # logger.info("***** Running evaluation {} *****".format(self.task_name))
        # logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(dataloader), desc="Evaluating")
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                input_ids, type_ids, attention_mask, label_ids = batch
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'type_ids': type_ids, 'labels': label_ids}
                outputs = model(**inputs)
                _, logits = outputs[:2]
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        # 保存标签结果
        # label_descs={}
        # with open(os.path.join(self.config.data_dir, "labels.json")) as f:
        #     for line in f.readlines():
        #         item=json.loads(line.strip())
        #         label,desc=item["label"],item["label_desc"]
        #         label_descs[label]=desc

        label_map = {i: label for i, label in enumerate(self.labels)}
        with open(self.config.output_submit_file, "w") as writer:
            for i, pred in enumerate(preds):
                label=str(label_map[pred])
                json_d = { "id":i, "label":label  }
                # json_d = { "id":i, "label":label,"label_desc":label_descs[label]  }
                writer.write(json.dumps(json_d) + '\n')

        logger.info(f" test : {len(preds)}  examples ")
        model.train()

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
            doc += self.rebanlance(doc, label_prob)
        label_prob1={}
        for item in doc:
            l=item[-1]
            label_prob1[l]=label_prob1.get(l,0)+1
        return doc

    def rebanlance(self, doc, label_prob):
        for k in label_prob.keys():
            label_prob[k] /= len(doc)
        expand = []
        for item in doc:
            label = item[-1]
            for i in range(5):
                if random.random() > label_prob[label]:
                    expand.append(item)
        return expand

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        if self.config.task_name=="wsc":
            a,b,c,d,e,l=self.doc[idx]
            a,b,c,d,e = self.tokenizer.tokenize(a,noise=0.5),self.tokenizer.tokenize(b),self.tokenizer.tokenize(c,noise=0.5),self.tokenizer.tokenize(d),self.tokenizer.tokenize(e,noise=0.5)

            # a=truncate_one(senta,max_len=self.max_tokens-3)
            tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a +["unsued1"]+b+["unsued2"]+c+["unsued3"]+d+["unsued4"]+e+ [Constants.TOKEN_EOS]
        else:
            a,b,l=self.doc[idx]
            if random.random()<0.5:
              a,b=b,a
            senta = self.tokenizer.tokenize(a)
            sentb = self.tokenizer.tokenize(b)
            a,b=truncate_pair(senta,sentb,max_len=self.max_tokens-5)
            tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + b + [Constants.TOKEN_EOS]

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
        return  np.array(tokens) , np.array(input_mask),np.array(type_ids) , length,label


if __name__ == "__main__":
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument('--model_name', default="none")
    # args = parser.parse_args()
    # model_name=args.model_name

    # outputs = '/media/u/t1/dataset/Poor/outputs/'
    # data_dir = '/media/u/t1/dataset/CLUEdatasets/'

    task_name="wsc"
    description="代词消歧"
    labels =  ["true", "false"]

    config = {
        "model_type": "albert",
        # "model_name_or_path": outputs + model_name,
        "task_name": task_name,
        # "data_dir": data_dir + task_name,
        # "vocab_file": outputs + f"{model_name}/vocab.txt",
        # "bujian_file": outputs + f"{model_name}/bujian.txt",
        # "model_config_path": outputs + f"{model_name}/config.json",
        # "output_dir": outputs + f"{model_name}/task_output",
        "max_len": 256,
        "batch_size":50,
        "n_epochs":10,
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
    # task.predict()