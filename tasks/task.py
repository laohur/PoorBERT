import copy
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
from torch.optim import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
# from callback.optimization.adamw import AdamW
from callback.progressbar import ProgressBar
from configs import Constants
from model.configuration_bert import BertConfig
from model.modeling_poor import BertForPreTraining, BertForSequenceClassification, BertForTokenClassification
from model.tokenization_shang import ShangTokenizer, Sentence
from tasks.utils import truncate_pair, TaskConfig, collate_fn
from tools.common import logger, init_logger

# logger = logging.getLogger(__name__)
# # FORMAT = '%(pathname)s %(filename)s  %(funcName)s %(lineno)d %(asctime)-15s  %(message)s'
# FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
# logging.basicConfig(filename="tasks.log",filemode='a',format=FORMAT,level=logging.INFO)



class TaskPoor:
    def __init__(self,config):
        # super(Task, self).__init__(config)
        self.config=TaskConfig(config)
        init_logger(log_file=f"{self.config.output_dir}/train.log")
        self.task_name=self.config.task_name
        self.dataset=self.config.TaskDataset
        self.labels=self.config.labels

        self.tokenizer = ShangTokenizer(vocab_path=self.config.vocab_file, bujian_path=self.config.bujian_file)
        # self.valid_dataset=self.load_valid()
        self.acc=0
        self.model = self.load_model(self.config.model_name_or_path)

        self.valid_dataset = None
        self.test_dataset=None

    def load_model_seq(self, model_path ):
        bert_config = BertConfig.from_pretrained(model_path,num_labels=len(self.labels),finetuning_task=self.task_name)
        logger.info(f" loadding {model_path} ")
        model = BertForSequenceClassification.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)
        model.to(self.config.device)
        return model

    def load_model_token(self,model_path):
        bert_config = BertConfig.from_pretrained(self.config.model_config_path,num_labels=len(self.labels),finetuning_task=self.task_name)
        logger.info(f" loadding {model_path} ")
        model = BertForTokenClassification.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)
        model.to(self.config.device)
        return model

    def train(self):
        input_file=os.path.join(self.config.data_dir,self.config.valid_file)
        self.valid_dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer,labels=self.labels, max_tokens=self.config.max_len,config=self.config)
        self.config.save_steps=max(self.config.save_steps,len(self.valid_dataset)//self.config.batch_size)
        self.config.logging_steps=max(self.config.logging_steps,len(self.valid_dataset)//self.config.batch_size)
        args=self.config
        model=self.model
        input_file=os.path.join(args.data_dir,self.config.train_file)
        dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer, labels=self.labels, max_tokens=self.config.max_len,config=self.config)
        num_training_steps=self.config.n_epochs*len(dataset)
        warmup_steps = int(num_training_steps * args.warmup_proportion)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bert"])],'lr': self.config.learning_rate},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ["bert"])], 'lr': self.config.learning_rate/5}
        # ]
        # # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        self.global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        for epoch in range(self.config.n_epochs):
            model.train()
            dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer, labels=self.labels, max_tokens=self.config.max_len,config=self.config)
            sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, collate_fn=collate_fn,pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)
            pbar = ProgressBar(n_total=len(dataloader), desc=f"{input_file[-20:]}")
            for step, batch in enumerate(dataloader):
                loss=self.train_batch(batch,args,optimizer,scheduler,step)
                msg={ "epoch":float(epoch), "global_step":float(self.global_step),"loss": loss ,"lr": float(scheduler.get_last_lr()[0]),"seq_len":batch[0].shape[1]   }
                pbar(step, msg)
                tr_loss += loss
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and self.global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        print()
                        acc=self.evaluate(epoch)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and (self.global_step % args.save_steps == 0 or step+1==len(dataloader))and acc>self.acc:
                    logger.info(f"Saving best model acc:{self.acc} -->{acc}")
                    self.acc=acc
                    output_dir = args.output_dir
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,  'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    # break
            # break
            print("\n ")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            msg = {"epoch": float(epoch), "global_step": float(self.global_step), "loss": loss, "average loss":tr_loss, "lr": float(scheduler.get_last_lr()[0])}
            logger.info(   f" {msg}")

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
        input_ids,token_type_ids, attention_mask, label_ids = batch
        inputs = {'input_ids': input_ids,  'attention_mask': attention_mask, 'token_type_ids':token_type_ids, 'labels': label_ids }
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
        dataset=self.valid_dataset
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(dataloader), desc="Evaluating")
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask,token_type_ids, label_ids = batch
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids':token_type_ids,'labels': label_ids}
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
            writer.write(line+"\n")
        logger.info(f"\n valid : {line} ")
        # for key in sorted(result.keys()):
        #     logger.info(" dev: %s = %s", key, str(result[key]))
        model.train()
        return acc

    def infer(self):
        args=self.config
        model= self.load_model(self.config.output_dir)
        logger.info(f"selected best model acc:{self.acc}")
        # model=self.model
        model.eval()
        # dataset=self.valid_dataset
        input_file=os.path.join(self.config.data_dir,self.config.test_file)
        dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer,labels=self.labels, max_tokens=self.config.max_len,config=self.config)
        self.test_dataset=dataset
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)

        nb_eval_steps = 0
        preds = None
        pbar = ProgressBar(n_total=len(dataloader), desc="Testing")
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask, token_type_ids, label_ids = batch
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, "token_type_ids": token_type_ids, 'labels': label_ids}
                outputs = model(**inputs)
                _, logits = outputs[:2]
            nb_eval_steps += 1
            prob=logits.detach().cpu().numpy()
            if args.output_mode=="TokenClassification":
                if preds is None:
                    preds=[]
                preds+=[x  for x in prob]
            else:
                if preds is None:
                    preds = prob
                else:
                    preds = np.append(preds, prob, axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        logger.info(f"inffered {len(preds)}")

        return preds