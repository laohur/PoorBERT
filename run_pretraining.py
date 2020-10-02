import gc
import os
import random
import traceback

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
from callback.lr_scheduler import get_linear_schedule_with_warmup
from configs import Constants
from model.PretrainQaDataset import PretrainQaDataset
from model.PretrainSelfDataset import PretrainSelfDataset
from model.PretrainTokenedDataset import PretrainTokenedDataset
from model.tokenization_shang import ShangTokenizer
from tools.common import logger, init_logger

def train_tokened(model,batch,config):
    input_ids, attention_mask,token_type_ids, lm_label_ids, char_label_ids, word_label_ids,relation_ids = batch
    tasks=[Constants.SCORE_CHAR,Constants.SCORE_WORD,Constants.SCORE_MASK,Constants.SCORE_RELATION ]
    outputs, scores= model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,tasks=tasks)
    prediction_scores,  char_score, word_score,seq_relationship_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_CHAR],scores[Constants.SCORE_WORD],scores[Constants.SCORE_RELATION]

    masked_lm_loss0 = loss_fct(prediction_scores[0].reshape(-1, bert_config.vocab_size), lm_label_ids.reshape(-1))
    masked_lm_loss1 = loss_fct(prediction_scores[1].reshape(-1, bert_config.vocab_size), lm_label_ids.reshape(-1))
    char_struct_loss = loss_fct(char_score.view(-1, 2), char_label_ids.reshape(-1))
    word_struct_loss = loss_fct(word_score.view(-1, 2), word_label_ids.reshape(-1))
    sentence_relation_loss0 = loss_fct(seq_relationship_score[0].view(-1, 5), relation_ids.view(-1))
    sentence_relation_loss1 = loss_fct(seq_relationship_score[1].view(-1, 5), relation_ids.view(-1))
    loss = 2*masked_lm_loss0+masked_lm_loss1 + char_struct_loss + word_struct_loss + 2*sentence_relation_loss0+sentence_relation_loss1   # bert

    if config.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if config.gradient_accumulation_steps > 1:
        loss = loss / config.gradient_accumulation_steps
    if config.fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    with torch.no_grad():
        mask_metric(logits=prediction_scores[0].reshape(-1, bert_config.vocab_size), target=lm_label_ids.reshape(-1))
        sop_metric(logits=seq_relationship_score[0].reshape(-1, 5), target=relation_ids.reshape(-1))
        loss_batch=[loss,masked_lm_loss0,masked_lm_loss1 , sentence_relation_loss0,sentence_relation_loss1, char_struct_loss , word_struct_loss ]
        loss_batch=[ x.item() for x in loss_batch ]

    msg={'总':loss_batch[0],'lr': scheduler.get_last_lr()[0],'掩0':loss_batch[1],'掩1':loss_batch[2],'系0':loss_batch[3],'系1':loss_batch[4],'字':loss_batch[5],'词':loss_batch[6],'len':batch[0].shape[1]}
    return msg

def train_self(model,batch,config):
    input_ids, attention_mask,token_type_ids, lm_label_ids,relation_ids = batch
    tasks=[Constants.SCORE_MASK,Constants.SCORE_RELATION ]
    outputs, scores= model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,tasks=tasks)
    prediction_scores,seq_relationship_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_RELATION]

    masked_lm_loss0 = loss_fct(prediction_scores[0].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
    masked_lm_loss1 = loss_fct(prediction_scores[1].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
    sentence_relation_loss0 = loss_fct(seq_relationship_score[0].view(-1, 5), relation_ids.view(-1))
    sentence_relation_loss1 = loss_fct(seq_relationship_score[1].view(-1, 5), relation_ids.view(-1))
    loss = 2*masked_lm_loss0+masked_lm_loss1 + 2*sentence_relation_loss0+sentence_relation_loss1

    if config.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if config.gradient_accumulation_steps > 1:
        loss = loss / config.gradient_accumulation_steps
    if config.fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    with torch.no_grad():
        mask_metric(logits=prediction_scores[0].view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
        sop_metric(logits=seq_relationship_score[0].view(-1, 5), target=relation_ids.view(-1))
        loss_batch = [loss,masked_lm_loss0, masked_lm_loss1,   sentence_relation_loss0, sentence_relation_loss1]
        loss_batch = [x.item() for x in loss_batch]

    msg={'总':loss_batch[0],'lr': scheduler.get_last_lr()[0],'掩0':loss_batch[1],'掩1':loss_batch[2],'系0':loss_batch[3],'系1':loss_batch[4],'len':batch[0].shape[1]}
    return msg

def train_qa(model,batch,config):
    input_ids, input_mask,token_type_ids,lm_label_ids, fake_ids = batch
    tasks=[ Constants.SCORE_MASK,Constants.SCORE_QA ]
    outputs, scores= model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=input_mask,tasks=tasks)
    prediction_scores,qa_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_QA]

    masked_lm_loss0 = loss_fct(prediction_scores[0].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
    masked_lm_loss1 = loss_fct(prediction_scores[1].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
    qa_loss0 = loss_fct(qa_score[0].view(-1, 2), fake_ids.view(-1))
    qa_loss1 = loss_fct(qa_score[1].view(-1, 2), fake_ids.view(-1))
    loss = 2*masked_lm_loss0+masked_lm_loss1 + 2*qa_loss0+qa_loss1

    if config.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if config.gradient_accumulation_steps > 1:
        loss = loss / config.gradient_accumulation_steps
    if config.fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    with torch.no_grad():
        mask_metric(logits=prediction_scores[0].view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
        sop_metric(logits=qa_score[0].view(-1, 2), target=fake_ids.view(-1))
        loss_batch = [loss,masked_lm_loss0, masked_lm_loss1,   qa_loss0, qa_loss1]
        loss_batch = [x.item() for x in loss_batch]

    msg={'总':loss_batch[0],'lr': scheduler.get_last_lr()[0],'掩0':loss_batch[1],'掩1':loss_batch[2],'系0':loss_batch[3],'系1':loss_batch[4],'len':batch[0].shape[1]}
    return msg

def train(model,tokenizer,file_path,config):
    if "tokened" in file_path:
        train_epoch=train_tokened
        tag="tokened"
        epoch_dataset = PretrainTokenedDataset(input_file=file_path, tokenizer=tokenizer, task='self', max_tokens=config.max_len)
    elif "self" in file_path:
        train_epoch=train_self
        tag="self"
        epoch_dataset = PretrainSelfDataset(input_file=file_path, tokenizer=tokenizer, task='self', max_tokens=config.max_len)
    elif "qa" in file_path:
        train_epoch=train_qa
        tag="qa"
        epoch_dataset = PretrainQaDataset(input_file=file_path,tokenizer=tokenizer,max_tokens=config.max_len)

    if config.local_rank == -1:
        train_sampler = RandomSampler(epoch_dataset)
    else:
        train_sampler = DistributedSampler(epoch_dataset)
    train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler,collate_fn=epoch_dataset.collate_fn, batch_size=config.train_batch_size,num_workers=config.num_workers,timeout=config.timeout,pin_memory=config.pin_memory)

    model.train()
    msg={}
    start_time = time.time()
    pbar = ProgressBar(n_total=len(train_dataloader), desc=f"{tag} pretrain {file_path[-20:]}")
    for step, batch in enumerate(train_dataloader):
        try:
            batch = tuple(t.to(device) for t in batch)
            msg = train_epoch(model, batch, config)
            msg["时"]=time.time()-start_time
            pbar(step, msg)
            train_after(step, start_time,tokenizer,len(batch),msg,len(train_dataloader),config)
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            # e = traceback.format_exc()
            print(f"argconfigs.max_len:{config.max_len} config.train_batch_size:{config.train_batch_size}  error:{e} \n")
    msg['file'] = file_path
    return msg

def train_after(step,start_time,tokenizer, batch_size,msg,n_total,config):
    config.global_step += 1
    if (config.global_step + 1) % config.gradient_accumulation_steps == 0:
        if config.fp16:
            torch.nn.utils.clip_grad_norm_(model.parameters(),  config.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    with torch.no_grad():
        loss,masked_lm_loss,next_sentence_loss=msg['总'],msg['掩0'],msg['系0']
        tr_mask_acc.update(mask_metric.value(), n=batch_size)
        tr_sop_acc.update(sop_metric.value(), n=batch_size)
        tr_loss.update(loss, n=1)
        tr_mask_loss.update(masked_lm_loss, n=1)
        tr_sop_loss.update(next_sentence_loss, n=1)

        if config.global_step % config.num_eval_steps == 0  or step+1==n_total :
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
            show_info = f'[Training]:[{epoch}/{config.epochs}]{config.global_step}/{num_train_optimization_steps} cost:{time.time()-config.last_time}  ' \
                        f'- ETA: {eta_format}' + "-".join([f' {key}: {value:.4f} ' for key, value in train_logs.items()])
            logger.info(show_info)
            tr_mask_acc.reset()
            tr_sop_acc.reset()
            tr_loss.reset()
            tr_mask_loss.reset()
            tr_sop_loss.reset()
            start_time = now
            config.last_time=time.time()

    if config.global_step % config.num_save_steps == 0 or (step+1==n_total and step>config.num_eval_steps ) :
        if config.local_rank in [-1, 0] and config.num_save_steps > 0:
            print()
            # Save model checkpoint
            output_dir = config.output_dir
            if not output_dir.exists():
                output_dir.mkdir()
            # save model
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(str(output_dir))
            torch.save(config, str(output_dir / 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), str(output_dir / "optimizer.bin"))
            # save config
            output_config_file = output_dir / CONFIG_NAME
            with open(str(output_config_file), 'w') as f:
                f.write(model_to_save.config.to_json_string())
            # save vocab
            tokenizer.save_vocabulary(output_dir)
    return start_time

class PretrainConfig:
    seed=42
    num_workers: int = 4
    timeout=1
    pin_memory = True

    noise = 0
    epochs = 1
    do_lower_case=True
    learning_rate=1e-4
    warmup_proportion=0.1
    num_eval_steps=1000
    num_save_steps=10000
    adam_epsilon=1e-6
    weight_decay=0.01
    max_grad_norm=1

    max_len=128
    train_batch_size=32
    total_train_examples=33e7  # 5m~=8000lines   1g~=E7
    gradient_accumulation_steps=1

    global_step:int =0
    last_time = time.time()

    no_cuda=False
    local_rank=-1  # local_rank for distributed training on gpus
    fp16=False
    fp16_opt_level="01"  #  Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'
    loss_scale=0   #    "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True."   "0 (default value): dynamic loss scaling."  "Positive power of 2: static loss scaling value."
    config_path=""
    vocab_path=""
    output_dir=""

    bert_config_path=""
    # log_file=""
    # trained_log=""
    vocab_path = "configs/vocab.txt"
    bujian_path = "configs/bujian.txt"
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)
        self.total_train_examples*=self.epochs
        self.output_dir=self.outputs
        self.log_file = str(self.output_dir / "train.log")
        self.trained_log = self.output_dir / "trained.log"

if __name__ == '__main__':
    BASE_DIR = Path('.')
    OUTPUTS_DIR = Path('/media/u/t1/dataset/PoorBERT') / 'pretrained'
    config_dict = {
        'outputs': OUTPUTS_DIR,
        'figure_dir': OUTPUTS_DIR / "figure",
        'checkpoint_dir': OUTPUTS_DIR,
        'bert_config_path': 'configs/poorbert_config.json',
        'vocab_path':  'configs/vocab.txt'
    }
    config = PretrainConfig(config_dict)

    if not config.output_dir.exists():
        config.output_dir.mkdir()
    log_file=str(config.output_dir/ "train.log")
    trained_log=config.output_dir / "trained.log"
    if not trained_log.exists():
        with open(trained_log, "a") as f:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"  {now}  start train")
    init_logger(log_file=log_file)

    if config.local_rank == -1 or config.no_cuda:
        device = torch.device(f"cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        config.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", config.local_rank)
        config.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(        f"device: {device} , distributed training: {bool(config.local_rank != -1)}, 16-bits training: {config.fp16}")

    if config.gradient_accumulation_steps < 1:
        raise ValueError(            f"Invalid gradient_accumulation_steps parameter: {config.gradient_accumulation_steps}, should be >= 1")

    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    lens = [1024, 512, 256, 128, 64]
    sizes = [7, 18, 44, 97, 208]
    steps=0
    for i, p in enumerate(probs):
        steps+=probs[i]/sizes[i]
    num_train_optimization_steps=int(config.total_train_examples*steps)
    config.train_batch_size=int(config.total_train_examples/num_train_optimization_steps)
    # config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    # num_train_optimization_steps = int( config.total_train_examples / config.train_batch_size / config.gradient_accumulation_steps)
    if config.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    config.warmup_steps = int(num_train_optimization_steps * config.warmup_proportion)

    bert_config = BertConfig.from_pretrained(config.bert_config_path)
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_train_optimization_steps)

    if os.path.exists(config.model_path+"/optimizer.bin"):
        optimizer.load_state_dict(torch.load(config.model_path + "/optimizer.bin"))
    if config.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16_opt_level)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)

    mask_metric = LMAccuracy()
    sop_metric = LMAccuracy()
    tr_mask_acc = AverageMeter()
    tr_sop_acc = AverageMeter()
    tr_loss = AverageMeter()
    tr_mask_loss = AverageMeter()
    tr_sop_loss = AverageMeter()
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {config.total_train_examples}")
    logger.info(f"  Batch size = {config.train_batch_size}")
    logger.info(f"  Num steps = {num_train_optimization_steps}")
    logger.info(f"  warmup_steps = {config.warmup_steps}")

    with open(trained_log) as f:
        trained_logs=f.readlines()

    tokenizer = ShangTokenizer(vocab_path=config.vocab_path, bujian_path=config.bujian_path)
    for epoch in range(config.epochs):
        files=[]
        files+=glob.glob(r"/media/u/t1/data/tokened/*/*.txt")
        # files+=glob.glob(r"/media/u/t1/data/self/*/*.txt")
        files+=glob.glob(r"/media/u/t1/data/qa/*/*.txt")
        # files=np.random.choice(files,100,replace=False)
        logger.info(  f" \n\n ==== training {len(files)} files ==== \n")  #13000
        # files.sort()
        random.shuffle(files)
        for fid,file_path in enumerate(files):
            file_path=str(file_path)
            trained=False
            for line in trained_logs:
                if file_path in line:
                    trained=True
                    break
            if trained:
                logger.info(f" {file_path}  already trained")
                continue
            if os.path.getsize(file_path) <100:
                logger.info(f" {file_path} file size {os.path.getsize(file_path)}  too small")
                continue
            idx = np.random.choice(a=len(probs), size=1, replace=False, p=probs)[0]
            config.max_len = lens[idx]
            config.train_batch_size = sizes[idx]
            # gradient_accumulation_steps=[16,12,8,4,2,1]
            # config.gradient_accumulation_steps=gradient_accumulation_steps[i]
            # config.train_batch_size = int(config.train_batch_size * 0.7)

            t0=time.time()
            logger.info(f" fid {fid} folder {len(files)}  max_len {config.max_len} batch_size {config.train_batch_size} gradient_accumulation_steps {config.gradient_accumulation_steps}")
            msg={                "fid":fid,                "file":file_path,            }
            trace=""
            gc.collect()
            torch.cuda.empty_cache()
            trace=""
            try:
                trace=train(model,tokenizer,file_path,config)
            except Exception as e:
                logger.error(f"{file_path} failed {e}")
            cost = time.time() - t0
            logger.info(f"  fid{fid} {file_path} trainned {cost}s {trace}")
            if trace:
                msg["success"] = trace
                with open(trained_log,"a") as f:
                    now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    f.write(f"time:{now} \t cost:{cost} \t file:{file_path} \t  trained \n")
            logger.info(json.dumps(msg,ensure_ascii=False))
        logger.info(f"trained {len(files)} files")

# if __name__ == '__main__':
    # main()
'''
 python3  run_pretraining.py  

none  
bujian      
gradation  
sentRelation
all 

'''