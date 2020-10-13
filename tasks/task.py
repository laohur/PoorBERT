import json
import math
import os
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
from model.configuration_bert import BertConfig
from model.modeling_poor import BertForPreTraining, BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering, BertForMultipleChoice
from model.tokenization_shang import ShangTokenizer, Sentence
from tasks.utils import truncate_pair, TaskConfig, find_span, cal_acc
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

    def load_model(self, model_path ):
        bert_config = BertConfig.from_pretrained(model_path, num_labels=self.config.num_labels, finetuning_task=self.task_name)
        logger.info(f" loadding {model_path} ")
        if self.config.task_name in ["c3", "chid"]:
            model = BertForMultipleChoice.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)
        elif self.config.output_mode == "span":
            model = BertForTokenClassification.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)
        elif self.config.output_mode == "qa":
            model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)
        elif   self.config.output_mode == "classification":
            model = BertForSequenceClassification.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=bert_config)

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
            dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer, labels=self.labels, max_tokens=self.config.max_len,config=self.config)
            sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, collate_fn=self.config.collate_fn,pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)
            pbar = ProgressBar(n_total=len(dataloader), desc=f"{input_file[-20:]}")
            for step, batch in enumerate(dataloader):
                loss=self.train_batch(batch,args,optimizer,scheduler,step)
                msg={ "epoch":float(epoch), "global_step":float(self.global_step),"loss": loss ,"lr": float(scheduler.get_last_lr()[0]),"seq_len":batch[0].shape[-1]   }
                pbar(step, msg)
                tr_loss += loss
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and (self.global_step % args.logging_steps == 0 or step+1==len(dataloader)  ):
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
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
        model.train()
        batch = tuple(t.to(self.config.device) for t in batch)
        if self.config.output_mode == "qa":
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'start_positions': start_positions, "end_positions": end_positions}
        else:
            input_ids, attention_mask, token_type_ids, label_ids = batch
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': label_ids}

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

        if (step + 1) % args.gradient_accumulation_steps == 0:
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
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, collate_fn=self.config.collate_fn, pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)
        print(' ')
        nb_eval_steps = 0
        scores=[]
        pbar = ProgressBar(n_total=len(dataloader), desc="Evaluating")
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                batch = tuple(t.to(args.device) for t in batch)
                if self.config.output_mode=="qa":
                    input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
                    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
                else:
                    input_ids, attention_mask,token_type_ids, label_ids = batch
                    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids':token_type_ids,'labels': label_ids}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if self.config.output_mode == "qa":
                    start_logits, end_logits=tmp_eval_loss, logits
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = start_logits.size(1)
                    start_positions.clamp_(0, ignored_index)
                    end_positions.clamp_(0, ignored_index)

                    score1 = cal_acc(start_logits, start_positions)
                    score2 = cal_acc(end_logits, end_positions)
                    scores.append((score1+ score2)/2)
                elif self.config.output_mode == "span" :
                    for i in range(len(logits)):
                        score = cal_acc(logits[i], label_ids[i])
                        scores.append((score))
                elif self.config.output_mode == "classification":
                    score = cal_acc(logits, label_ids)
                    scores.append(score)
            nb_eval_steps += 1
            pbar(step)
            # break
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        acc = np.array(scores).mean()
        result={"acc": acc,"epoch":epoch,"step":self.global_step}

        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        line=json.dumps(result,ensure_ascii=False)
        with open(output_eval_file, "a") as writer:
            writer.write(line+"\n")
        logger.info(f"\n valid : {line} ")
        model.train()
        return acc

    def infer(self):
        args=self.config
        logger.info(f"selected best model acc:{self.acc}")
        model= self.load_model(self.config.output_dir)
        # model=self.model
        model.eval()
        # dataset=self.valid_dataset
        input_file=os.path.join(self.config.data_dir,self.config.test_file)
        dataset = self.dataset(input_file=input_file, tokenizer=self.tokenizer,labels=self.labels, max_tokens=self.config.max_len,config=self.config)
        self.test_dataset=dataset
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, collate_fn=self.config.collate_fn, pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)

        nb_eval_steps = 0
        preds = []
        pbar = ProgressBar(n_total=len(dataloader), desc="Testing")
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                if self.config.output_mode == "qa":
                    input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
                    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
                else:
                    input_ids, attention_mask, token_type_ids, label_ids = batch
                    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': label_ids}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if self.config.output_mode == "qa":
                    start_logits, end_logits=tmp_eval_loss, logits
                    start = torch.argmax(start_logits, 1).tolist()
                    end = torch.argmax(end_logits, 1).tolist()
                    preds+=zip(start,end)
                elif args.output_mode=="span":
                    prob = logits.detach().cpu().numpy()
                    preds+=[x  for x in prob]
                elif args.output_mode == "classification":
                    preds+=torch.argmax(logits, 1).tolist()
            nb_eval_steps += 1
            pbar(step)
            # break
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        logger.info(f"infered {len(preds)}")
        return preds