import json
import logging
import os
import random
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, SequentialSampler, RandomSampler

from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.optimization.adamw import AdamW
from callback.progressbar import ProgressBar
from configs import Constants
from model.configuration_albert import AlbertConfig
from model.modeling_peng import AlbertForPreTraining, AlbertForSequenceClassification
from model.tokenization_shang import ShangTokenizer, Sentence
from tasks.utils import truncate_pair, TaskConfig, collate_fn, truncate_one

logger = logging.getLogger(__name__)
# FORMAT = '%(pathname)s %(filename)s  %(funcName)s %(lineno)d %(asctime)-15s  %(message)s'
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)

class Task:
    def __init__(self,config):
        # super(Task, self).__init__(config)
        self.config=TaskConfig(config)
        self.task_name=self.config.task_name
        self.dataset=TaskDataset
        self.labels=self.config.labels

        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        self.tokenizer = ShangTokenizer(vocab_path="../configs/vocab_shang.txt", split_path="../configs/spliter_cn.txt")
        self.model = self.load_model()
        # self.valid_dataset=self.load_valid()

    def load_model(self):
        bert_config = AlbertConfig.from_pretrained(self.config.model_config_path,num_labels=len(self.labels),finetuning_task=self.task_name)
        # model = AlbertForSequenceClassification(config=bert_config)
        # config = AlbertConfig.from_pretrained(args.config_path, num_labels=num_labels, finetuning_task=args.task_name)
        model_path = self.config.model_name_or_path
        # if os.path.exists(model_path):
        logger.info(f" loadding {model_path} ")
            # model = AlbertForPreTraining.from_pretrained(model_path,config=bert_config)
        model = AlbertForSequenceClassification.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)
        model.to(self.config.device)
        return model

    def train(self):
        args=self.config
        model=self.model
        input_file=os.path.join(args.data_dir,self.config.train_file)
        dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer, labels=self.labels, max_tokens=self.config.max_len)
        num_training_steps=self.config.n_epochs*len(dataset)
        warmup_steps = int(num_training_steps * args.warmup_proportion)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
        model.train()
        self.global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        # global_step, tr_loss = self.train_batch(self.config, train_dataloader, model, self.tokenizer, optimizer, scheduler)
        for epoch in range(self.config.n_epochs):
            dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer, labels=self.labels, max_tokens=self.config.max_len)
            sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, collate_fn=collate_fn, num_workers=self.config.num_workers)
            pbar = ProgressBar(n_total=len(dataloader), desc=f"{input_file[-20:]}")
            for step, batch in enumerate(dataloader):
                loss=self.train_batch(batch,args,optimizer,scheduler,step)
                msg={ "epoch":float(epoch), "global_step":float(self.global_step),"loss": loss ,"lr": float(scheduler.get_last_lr()[0]),"seq_len":batch[0].shape[1]  }
                pbar(step, msg)
                tr_loss += loss
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and self.global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        self.evaluate(epoch)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and self.global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(self.global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,  'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
            print(" ")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            msg = {"epoch": float(epoch), "global_step": float(self.global_step), "loss": loss, "average loss":tr_loss, "lr": float(scheduler.get_last_lr()[0])}

            logger.info(   f" {msg}")
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    def train_batch(self, batch,args,optimizer,scheduler,step):
        model=self.model
        batch = tuple(t.to(self.config.device) for t in batch)
        input_ids, attention_mask, label_ids = batch
        inputs = {'input_ids': input_ids,  'attention_mask': attention_mask,  'labels': label_ids }
        # outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=label_ids)
        outputs = self.model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (step + 1) % args.gradient_accumlengthulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            self.global_step += 1

        return loss.item()

    def evaluate(self,epoch):
        args=self.config
        model=self.model
        model.eval()
        # dataset=self.valid_dataset
        input_file=os.path.join(self.config.data_dir,self.config.valid_file)
        dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer,labels=self.labels, max_tokens=self.config.max_len)
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
                input_ids, attention_mask, label_ids = batch
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_ids}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
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
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        # result = compute_metrics(eval_task, preds, out_label_ids)
        acc=(preds == out_label_ids).mean()
        result={"acc": acc,"epoch":epoch,"step":self.global_step}

        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        line=json.dumps(result,ensure_ascii=False)
        with open(output_eval_file, "a") as writer:
            writer.write(line)
        logger.info(f" valid : {line} ")
        # for key in sorted(result.keys()):
        #     logger.info(" dev: %s = %s", key, str(result[key]))
        model.train()
    def predict(self):
        args=self.config
        model=self.model
        model.eval()
        # dataset=self.valid_dataset
        input_file=os.path.join(self.config.data_dir,self.config.test_file)
        dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer,labels=self.labels, max_tokens=self.config.max_len)
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
                input_ids, attention_mask, label_ids = batch
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_ids}
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

        output_submit_file = os.path.join(self.config.output_dir, f"{self.task_name}_prediction.json")
        # output_logits_file = os.path.join(self.config.output_dir, "test_logits")
        # 保存标签结果
        # label_descs={}
        # with open(os.path.join(self.config.data_dir, "labels.json")) as f:
        #     for line in f.readlines():
        #         item=json.loads(line.strip())
        #         label,desc=item["label"],item["label_desc"]
        #         label_descs[label]=desc

        label_map = {i: label for i, label in enumerate(self.labels)}
        with open(output_submit_file, "w") as writer:
            for i, pred in enumerate(preds):
                label=str(label_map[pred])
                json_d = { "id":i, "label":label  }
                # json_d = { "id":i, "label":label,"label_desc":label_descs[label]  }
                writer.write(json.dumps(json_d) + '\n')

        logger.info(f" test : {len(preds)}  examples ")
        model.train()

class TaskDataset(Dataset):
    def __init__(self,input_file,labels,tokenizer,max_tokens):
        super(TaskDataset, self).__init__()
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
        doc1={}
        for line in doc0:
            item=json.loads(line.strip())
            a =item["abst"].strip()
            if a not in doc1:
                doc1[a]=[]
            doc1[a].append(line)

        doc=[]
        for k,v in doc1.items():
            real,all=set(),set()
            for line in v:
                item=json.loads(line.strip())
                keywords=item["keyword"]
                l=item.get("label",self.labels[0]).strip()
                if l not in self.labels:
                    logger.warn(f" error label {line} ")
                    continue
                all|=set(keywords)
                if l==  self.labels[1]:
                    real|=set(keywords)
            # fake=all-real
            for k in all:
                if k in real:
                    l=self.labels[1]
                else:
                    l=self.labels[0]
                doc.append([k,a,l])
            # a, b, l = a.strip(), b.strip(), l.strip()
            # doc.append([a,b,l])
        return doc

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        if len(self.doc[idx])==2:
            a,l=self.doc[idx]
            senta = self.tokenizer.tokenize(a)
            a=truncate_one(senta,max_len=self.max_tokens-3)
            tokens = [Constants.TOKEN_CLS,Constants.TOKEN_BOS] + a + [Constants.TOKEN_EOS]
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
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask=[1]*length+[0]*(self.max_tokens-length)
        return  np.array(tokens) , np.array(input_mask) , length,label


if __name__ == "__main__":
    outputs = '/media/u/t1/dataset/Peng/outputs/'
    data_dir = '/media/u/t1/dataset/CLUEdatasets/'
    task_name="csl"
    labels =  ["0", "1"]

    config = {
        "model_type": "albert",
        "model_name_or_path": outputs + "lm-checkpoint",
        "task_name": task_name,
        "data_dir": data_dir + task_name,
        "vocab_file": outputs + "lm-checkpoint/vocab_shang.txt",
        "model_config_path": outputs + "lm-checkpoint/config.json",
        "max_seq_length": 512,
        "batch_size":80,
        "output_dir": outputs + f"{task_name}_output",
        "learning_rate": 5e-5,
        "n_epochs": 1,
        "logging_steps": 100,
        "save_steps": 1000,
        "train_file":"train.json",
        "valid_file":"dev.json",
        "test_file":"test.json",
        "labels":labels
        # "per_gpu_train_batch_size": 16,
        # "per_gpu_eval_batch_size": 16,
    }
    # torch.multiprocessing.set_start_method('spawn')
    task=Task(config)
    task.train()
    task.predict()