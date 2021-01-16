import gc
import os
import random
import traceback
from argparse import ArgumentParser
import numpy as np
import torch
import json
import time
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
import glob
from pathlib import Path

from callback.progressbar import ProgressBar
from tools.common import AverageMeter
from metrics.custom_metrics import LMAccuracy
from model.modeling_poor import BertForPreTraining, BertConfig
from model.file_utils import CONFIG_NAME
from callback.lr_scheduler import PoorLR,get_linear_schedule_with_warmup
from configs import Constants
# from model.data_parallel import BalancedDataParallel
from model.PretrainQaDataset import PretrainQaDataset
from model.PretrainSelfDataset import PretrainSelfDataset
from model.PretrainTokenedDataset import PretrainTokenedDataset
from model.tokenization_shang import ShangTokenizer
from tools.common import logger, init_logger


class PretrainConfig:
    use_relation=True
    use_bujian=True
    use_stair=False

    seed=42
    num_workers: int = 64
    timeout=10
    pin_memory = True

    noise = 0
    epochs = 1
    do_lower_case=True
    learning_rate=5e-4 
    warmup_proportion=0.1
    num_eval_steps=1000
    num_save_steps=5000
    adam_epsilon=1e-6
    weight_decay=0.01
    max_grad_norm=1

    max_len=512
    train_batch_size=22
    gradient_accumulation_steps=1
    now_stair=0

    global_step:int =0
    last_time = time.time()

    no_cuda=False
    local_rank=-1  # local_rank for distributed training on gpus   ## from gpu_id
    fp16=False
    fp16_opt_level="01"  #  Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'
    loss_scale=0   #    "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True."   "0 (default value): dynamic loss scaling."  "Positive power of 2: static loss scaling value."
    config_path=""
    output_dir=""

    bert_config_path=""
    # log_file=""
    # trained_log=""
    # total_train_examples = 0.2*10e7  # 5m~=8000lines   1g~=E7  30  150  *0.28
    vocab_path = "configs/vocab.txt"
    bujian_path = "configs/bujian.txt"
    corpus_dir=""
    corpus=[]
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)
        # self.total_train_examples*=self.epochs
        self.output_dir=self.outputs
        self.log_file = str(self.output_dir / "train.log")
        self.trained_log = self.output_dir / "trained.log"

        if "poornone" in str(self.output_dir).lower():
            self.use_relation = False
            self.use_bujian = False
            # self.use_stair = False
        if "relation" in str(self.output_dir).lower():
            self.use_relation = True
            self.use_bujian = False
            # self.use_stair = False
        elif "bujian" in str(self.output_dir).lower():
            self.use_relation = False
            self.use_bujian = True
            # self.use_stair = False
        # elif "gradation" in str(self.output_dir).lower():
        #     self.use_relation = False
        #     self.use_bujian = False
        #     self.use_stair = False
        elif "poorbert" in str(self.output_dir).lower():
            self.use_relation = True
            self.use_bujian = True
            # self.use_stair = True
        
        def get_corpus(path):
            files=[]
            files+=glob.glob(rf"{path}/tokened/*/*.txt",recursive=True)
            files+=glob.glob(rf"{path}/self/*/*.txt",recursive=True)
            files+=glob.glob(rf"{path}/qa/*/*.txt",recursive=True)
            # files=np.random.choice(files,100,replace=False)
            logger.info(  f" \n\n ==== training {len(files)} files ==== \n")  #13000
            # files.sort()
            random.shuffle(files) 
            return files        
        self.corpus=get_corpus(self.corpus_dir)   


class Trainer:
    now_stair=0

    def __init__(self,config):
        self.config=config
        if not config.output_dir.exists():
            config.output_dir.mkdir()
        log_file=str(config.output_dir/ "train.log")
        init_logger(log_file=log_file)
        self.trained_log=config.output_dir / "trained.log"
        if not self.trained_log.exists():
            with open(self.trained_log, "a") as f:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"  {now}  start train")

        if args.local_rank == -1 or config.no_cuda:
            device = torch.device(f"cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
            config.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            config.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info(        f"device: {device} , distributed training: {args.local_rank }, 16-bits training: {config.fp16}")
        config.device=device
        if config.gradient_accumulation_steps < 1:
            raise ValueError(    f"Invalid gradient_accumulation_steps parameter: {config.gradient_accumulation_steps}, should be >= 1")

        bert_config = BertConfig.from_pretrained(config.bert_config_path)
        # bert_config.use_stair=config.use_stair

        model = BertForPreTraining(config=bert_config)
        config.model_path = str((config.output_dir).absolute())
        if os.path.exists(config.model_path+"/bujian.txt"):
            logger.info(f" loadding {config.model_path} ")
            model = BertForPreTraining.from_pretrained(config.model_path)
        model.to(device)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        for name,param in param_optimizer:
            logger.info(f" param size {name}-->{param.size()} ")
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_train_optimization_steps)
        scheduler = PoorLR(optimizer,base_lr=config.learning_rate,total_stage=len(config.corpus),warm_up=config.warmup_proportion)
        if os.path.exists(config.model_path+"/optimizer.bin"):
            optimizer.load_state_dict(torch.load(config.model_path + "/optimizer.bin"), map_location=torch.device('cpu'))
        if config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16_opt_level)
        if config.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            # model = BalancedDataParallel(2, model, dim=4).to(device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
        
        self.bert_config=bert_config
        self.tokenizer = ShangTokenizer(vocab_path=config.vocab_path, bujian_path=config.bujian_path,use_bujian=config.use_bujian)
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler

        self.mask_metric = LMAccuracy()
        self.sop_metric = LMAccuracy()
        self.tr_mask_acc = AverageMeter()
        self.tr_sop_acc = AverageMeter()
        self.tr_loss = AverageMeter()
        self.tr_mask_loss = AverageMeter()
        self.tr_sop_loss = AverageMeter()
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)        
    

    def train_tokened(self,batch):
        config=self.config
        bert_config=self.bert_config
        model=self.model
        config=self.config
        loss_fct=self.loss_fct
        mask_metric=self.mask_metric
        sop_metric=self.sop_metric

        input_ids, attention_mask,token_type_ids, lm_label_ids, char_label_ids, word_label_ids,relation_ids = batch
        tasks=[Constants.SCORE_CHAR,Constants.SCORE_WORD,Constants.SCORE_MASK,Constants.SCORE_RELATION ]
        outputs, scores= model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,tasks=tasks)
        prediction_scores,  char_score, word_score,seq_relationship_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_CHAR],scores[Constants.SCORE_WORD],scores[Constants.SCORE_RELATION]

        masked_lm_loss = loss_fct(prediction_scores.reshape(-1, bert_config.vocab_size), lm_label_ids.reshape(-1))
        char_struct_loss = loss_fct(char_score.view(-1, 2), char_label_ids.reshape(-1))
        word_struct_loss = loss_fct(word_score.view(-1, 2), word_label_ids.reshape(-1))
        sentence_relation_loss = loss_fct(seq_relationship_score.view(-1, 5), relation_ids.view(-1))
        loss = masked_lm_loss + char_struct_loss + word_struct_loss + sentence_relation_loss

        if config.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        if config.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        with torch.no_grad():
            mask_metric(logits=prediction_scores.reshape(-1, bert_config.vocab_size), target=lm_label_ids.reshape(-1))
            sop_metric(logits=seq_relationship_score.reshape(-1, 5), target=relation_ids.reshape(-1))
            loss_batch=[loss,masked_lm_loss, sentence_relation_loss, char_struct_loss , word_struct_loss ]
            loss_batch=[ x.item() for x in loss_batch ]

        msg={'总':loss_batch[0],'掩':loss_batch[1],'系':loss_batch[2],'字':loss_batch[3],'词':loss_batch[4]}
        return msg

    def train_self(self,batch):
        config=self.config
        bert_config=self.bert_config
        model=self.model
        config=self.config
        loss_fct=self.loss_fct
        mask_metric=self.mask_metric
        sop_metric=self.sop_metric

        input_ids, attention_mask,token_type_ids, lm_label_ids,relation_ids = batch
        tasks=[Constants.SCORE_MASK,Constants.SCORE_RELATION ]
        outputs, scores= model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,tasks=tasks)
        prediction_scores,seq_relationship_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_RELATION]

        masked_lm_loss = loss_fct(prediction_scores.view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
        sentence_relation_loss = loss_fct(seq_relationship_score.view(-1, 5), relation_ids.view(-1))
        loss = masked_lm_loss + sentence_relation_loss

        if config.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        if config.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        with torch.no_grad():
            mask_metric(logits=prediction_scores.view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
            sop_metric(logits=seq_relationship_score.view(-1, 5), target=relation_ids.view(-1))
            loss_batch = [loss,masked_lm_loss,   sentence_relation_loss]
            loss_batch = [x.item() for x in loss_batch]

        msg={'总':loss_batch[0],'掩':loss_batch[1],'系':loss_batch[2]}
        return msg

    def train_qa(self,batch):
        config=self.config
        bert_config=self.bert_config
        model=self.model
        config=self.config
        loss_fct=self.loss_fct
        mask_metric=self.mask_metric
        sop_metric=self.sop_metric
        scheduler=self.scheduler

        input_ids, input_mask,token_type_ids,lm_label_ids, fake_ids = batch
        tasks=[ Constants.SCORE_MASK,Constants.SCORE_QA ]
        outputs, scores= model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=input_mask,tasks=tasks)
        prediction_scores,qa_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_QA]

        masked_lm_loss = loss_fct(prediction_scores.view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
        qa_loss = loss_fct(qa_score.view(-1, 2), fake_ids.view(-1))
        loss = masked_lm_loss + qa_loss

        if config.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        if config.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        with torch.no_grad():
            mask_metric(logits=prediction_scores.view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
            sop_metric(logits=qa_score.view(-1, 2), target=fake_ids.view(-1))
            loss_batch = [loss,masked_lm_loss,   qa_loss]
            loss_batch = [x.item() for x in loss_batch]

        msg={'总':loss_batch[0],'lr': scheduler.get_lr(),'掩':loss_batch[1], '系':loss_batch[2]}
        return msg

    def train(self,file_path):
        config=self.config
        bert_config=self.bert_config
        model=self.model
        config=self.config
        loss_fct=self.loss_fct
        mask_metric=self.mask_metric
        sop_metric=self.sop_metric
        scheduler=self.scheduler
        tokenizer=self.tokenizer
        
        if "tokened" in file_path:
            train_epoch=self.train_tokened
            tag="tokened"
            epoch_dataset = PretrainTokenedDataset(input_file=file_path, tokenizer=tokenizer, task='self', max_tokens=config.max_len,use_relation=config.use_relation)
        elif "self" in file_path:
            train_epoch=self.train_self
            tag="self"
            epoch_dataset = PretrainSelfDataset(input_file=file_path, tokenizer=tokenizer, task='self', max_tokens=config.max_len,use_relation=config.use_relation)
        elif "qa" in file_path:
            train_epoch=self.train_qa
            tag="qa"
            epoch_dataset = PretrainQaDataset(input_file=file_path,tokenizer=tokenizer,max_tokens=config.max_len,use_relation=config.use_relation)

        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler,collate_fn=epoch_dataset.collate_fn, batch_size=config.train_batch_size,num_workers=config.num_workers,timeout=config.timeout,pin_memory=config.pin_memory)

        model.train()
        msg={}
        start_time = time.time()
        pbar = ProgressBar(n_total=len(train_dataloader), desc=f"{tag} {file_path[-15:]}")
        self.scheduler.feed(len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            try:
                batch = tuple(t.to(self.config.device) for t in batch)
                msg = train_epoch( batch)
                msg["时"]=time.time()-start_time
                msg["lr"]= self.scheduler.get_lr()
                msg["长"]= batch[0].shape[1]
                pbar(step, msg)
                self.train_after(step, start_time,len(batch),msg,len(train_dataloader))
            except Exception as e:
                gc.collect()
                torch.cuda.empty_cache()
                logger.error(f"argconfigs.max_len:{config.max_len} config.train_batch_size:{config.train_batch_size}  error:{e} \n")
        msg['file'] = file_path
        return msg

    def train_after(self,step,start_time, batch_size,msg,n_total):
        config=self.config
        bert_config=self.bert_config
        model=self.model
        config=self.config
        loss_fct=self.loss_fct
        mask_metric=self.mask_metric
        sop_metric=self.sop_metric
        scheduler=self.scheduler
        optimizer=self.optimizer

        tr_mask_acc=self.tr_mask_acc 
        tr_sop_acc=self.tr_sop_acc
        tr_loss=self.tr_loss
        tr_mask_loss=self.tr_mask_loss
        tr_sop_loss=self.tr_sop_loss

        config.global_step += 1
        p,p1,p2= scheduler.get_progress()
        if (config.global_step + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16:
                torch.nn.utils.clip_grad_norm_(model.parameters(),  config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            if bert_config.use_stair:
                next_stair=int(p*20)
                if self.now_stair<next_stair<bert_config.num_hidden_layers:
                    model.upstair(next_stair)
                    self.now_stair+=1

        with torch.no_grad():
            loss,masked_lm_loss,next_sentence_loss=msg['总'],msg['掩'],msg['系']
            tr_mask_acc.update(mask_metric.value(), n=batch_size)
            tr_sop_acc.update(sop_metric.value(), n=batch_size)
            tr_loss.update(loss, n=1)
            tr_mask_loss.update(masked_lm_loss, n=1)
            tr_sop_loss.update(next_sentence_loss, n=1)

            if config.num_eval_steps>0 and config.global_step % config.num_eval_steps == 0:
                print()
                now = time.time()
                eta = now - start_time
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta
                train_logs = {}
                train_logs['loss'] = tr_loss.avg
                train_logs['mask_acc'] = tr_mask_acc.avg
                train_logs['sop_acc'] = tr_sop_acc.avg
                train_logs['mask_loss'] = tr_mask_loss.avg
                train_logs['sop_loss'] = tr_sop_loss.avg
                show_info = f'[Training]:[progress:{p}] cost:{time.time()-config.last_time}s ' \
                            f'- ETA: {eta_format}' + "-".join([f' {key}: {value:.4f} ' for key, value in train_logs.items()])
                logger.info(show_info)
                tr_mask_acc.reset()
                tr_sop_acc.reset()
                tr_loss.reset()
                tr_mask_loss.reset()
                tr_sop_loss.reset()
                start_time = now
                config.last_time=time.time()

        if args.local_rank in [-1, 0] and config.num_save_steps > 0  and  ( step+1==n_total  ):
        # if config.local_rank in [-1, 0] and config.num_save_steps > 0 and (config.global_step % config.num_save_steps == 0 or fid + 1 == len(files)):
            print()
            # Save model checkpoint
            output_dir = config.output_dir
            if not output_dir.exists():
                output_dir.mkdir()
            # save model
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(str(output_dir))
            # torch.save(config, str(output_dir / 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            # torch.save(optimizer.state_dict(), str(output_dir / "optimizer.bin"))
            # save config
            output_config_file = output_dir / CONFIG_NAME
            with open(str(output_config_file), 'w') as f:
                f.write(model_to_save.config.to_json_string())
            # save vocab
            self.tokenizer.save_vocabulary(output_dir)
        return start_time

    def pretrain(self):
        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {config.total_train_examples}")
        # logger.info(f"  Batch size = {config.train_batch_size}")
        # logger.info(f"  Num steps = {num_train_optimization_steps}")
        # logger.info(f"  warmup_steps = {config.warmup_steps}")
        # base
        # probs = [0.1, 0.5, 0.3, 0.1]
        # lens = [1024, 512, 256, 128]
        # sizes = [10, 29, 78, 172]

        # small
        probs = [ 0.5, 0.3, 0.2] 
        lens = [ 512, 256, 128]
        sizes = [ 70, 160, 320]

        with open(self.trained_log) as f:
            self.trained_logs=f.readlines()

        for epoch in range(config.epochs):
            random.shuffle(config.corpus)
            for fid,file_path in enumerate(config.corpus):
                file_path=str(file_path)
                trained=False
                for line in self.trained_logs:
                    if file_path in line:
                        trained=True
                        break
                if trained:
                    logger.info(f" {file_path}  already trained")
                    self.scheduler.feed(1)
                    continue
                if os.path.getsize(file_path) <100:
                    logger.info(f" {file_path} file size {os.path.getsize(file_path)}  too small")
                    self.scheduler.feed(1)
                    continue
                idx = np.random.choice(a=len(probs), size=1, replace=False, p=probs)[0]
                # idx=0
                config.max_len = lens[idx]
                config.train_batch_size = sizes[idx]
                # config.train_batch_size = int(config.train_batch_size * 0.5)

                t0=time.time()
                logger.info(f" \n\n fid {fid} folder {len(config.corpus)}  max_len {config.max_len} batch_size {config.train_batch_size} global_step {config.global_step}")
                trace=""
                gc.collect()
                torch.cuda.empty_cache()
                trace=""
                try:
                    trace=self.train(file_path)
                except Exception as e:
                    logger.error(f"{file_path} failed {e}")
                cost = time.time() - t0
                msg={     "fid":fid,   "file":file_path, "cost":cost, "success":"False", "now_stair":self.now_stair    }
                # logger.info(f"  fid{fid} {file_path} trainned {cost}s {trace}")
                if trace:
                    msg["success"] = trace
                    with open(self.trained_log,"a") as f:
                        now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        f.write(f"time:{now} \t cost:{cost} \t file:{file_path} \t  trained \n")
                logger.info(json.dumps(msg,ensure_ascii=False))
            logger.info(f"trained {len(config.corpus)} files")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,  help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    # BASE_DIR = Path('/mfs/entropy/nlp/')
    OUTPUTS_DIR = Path('../../dataset/PoorBERT') / 'pretrained'  #  PoorBERT>
    config_dict = {
        'corpus_dir':"../../data",  # ->
        'outputs': OUTPUTS_DIR,
        'figure_dir': OUTPUTS_DIR / "figure",
        'checkpoint_dir': OUTPUTS_DIR,
        'bert_config_path': 'configs/poorbert_config.json',
        'vocab_path':  'configs/vocab.txt',
        'local_rank':args.local_rank
    }
    config = PretrainConfig(config_dict)    

    trainer=Trainer(config)
    trainer.pretrain()

'''
CUDA_VISIBLE_DEVICES=6,7,8,9 python3  pretrain.py  
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch  pretrain.py
none  
bujian      
stair  
Relation
all 

'''