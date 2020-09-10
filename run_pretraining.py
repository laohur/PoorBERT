import os
import random
import traceback

import torch
import json
import time
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
from tempfile import TemporaryDirectory

from tensorboardX import SummaryWriter

from callback.progressbar import ProgressBar
from configs import Constants
from model.PretrainQaDataset import PretrainQaDataset
from model.PretrainSelfDataset import PretrainSelfDataset
from model.PretrainTokenedDataset import PretrainTokenedDataset
from model.tokenization_shang import ShangTokenizer
from tools.common import logger, init_logger
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tools.common import AverageMeter
from metrics.custom_metrics import LMAccuracy
from torch.nn import CrossEntropyLoss
# from model.modeling_albert import AlbertForPreTraining, AlbertConfig
# from model.modeling_shang import AlbertForPreTraining, AlbertConfig
# from model.modeling_peng import AlbertForPreTraining, AlbertConfig
from model.modeling_poor import AlbertForPreTraining, AlbertConfig
from model.file_utils import CONFIG_NAME
from model.tokenization_bert import BertTokenizer
from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from tools.common import seed_everything
# from  torch.cuda.amp import autocast
# from torch.cuda.amp import  GradScaler
import glob

global_step = 0
last_time = time.time()
# scaler=GradScaler()

def train_tokened(model,tokenizer,file_path):
    global global_step
    start_time = time.time()
    t0=time.time()
    epoch_dataset = PretrainTokenedDataset(input_file=file_path,tokenizer=tokenizer,task='self',max_tokens=args.max_len)

    if args.local_rank == -1:
        train_sampler = RandomSampler(epoch_dataset)
    else:
        train_sampler = DistributedSampler(epoch_dataset)
    train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler,collate_fn=epoch_dataset.collate_fn, batch_size=args.train_batch_size,num_workers=6)
    model.train()
    pbar = ProgressBar(n_total=len(train_dataloader), desc=f"tokened {file_path[-20:]}")
    msg={}
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask,type_ids, lm_label_ids, char_label_ids, word_label_ids,relation_ids = batch
        tasks=[Constants.SCORE_CHAR,Constants.SCORE_WORD,Constants.SCORE_MASK,Constants.SCORE_RELATION ]
        outputs, scores= model(input_ids=input_ids, type_ids=type_ids, attention_mask=attention_mask,tasks=tasks)
        prediction_scores,  char_score, word_score,seq_relationship_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_CHAR],scores[Constants.SCORE_WORD],scores[Constants.SCORE_RELATION]

        # print(f" shape {prediction_scores[0].shape} {lm_label_ids.shape}")
        masked_lm_loss0 = loss_fct(prediction_scores[0].reshape(-1, bert_config.vocab_size), lm_label_ids.reshape(-1))
        masked_lm_loss1 = loss_fct(prediction_scores[1].reshape(-1, bert_config.vocab_size), lm_label_ids.reshape(-1))
        masked_lm_loss=2*masked_lm_loss0+masked_lm_loss1
        char_struct_loss = loss_fct(char_score.view(-1, 2), char_label_ids.reshape(-1))
        word_struct_loss = loss_fct(word_score.view(-1, 2), word_label_ids.reshape(-1))
        sentence_relation_loss0 = loss_fct(seq_relationship_score[0].view(-1, 5), relation_ids.view(-1))
        sentence_relation_loss1 = loss_fct(seq_relationship_score[1].view(-1, 5), relation_ids.view(-1))
        sentence_relation_loss=2*sentence_relation_loss0+sentence_relation_loss1
        # loss = masked_lm_loss0 + char_struct_loss + word_struct_loss + sentence_relation_loss0  # bert512
        loss = masked_lm_loss + char_struct_loss + word_struct_loss + sentence_relation_loss   # bert

        # mask_metric(logits=scores[Constants.SCORE_MASK].view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
        mask_metric(logits=prediction_scores[0].reshape(-1, bert_config.vocab_size), target=lm_label_ids.reshape(-1))
        # sop_metric(logits=seq_relationship_score.view(-1, 2), target=is_next.view(-1))
        sop_metric(logits=seq_relationship_score[0].reshape(-1, 5), target=relation_ids.reshape(-1))

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
        else:
            loss.backward()

        loss_batch=[loss,masked_lm_loss0,masked_lm_loss1 , sentence_relation_loss0,sentence_relation_loss1, char_struct_loss , word_struct_loss ]
        loss_batch=[ x.cpu().item() for x in loss_batch ]
        msg={'总':loss_batch[0],'时':time.time()-t0 ,'lr': scheduler.get_last_lr()[0],'掩0':loss_batch[1],'掩1':loss_batch[2],'系0':loss_batch[3],'系1':loss_batch[4],'字':loss_batch[5],'词':loss_batch[6],'len':batch[0].shape[1]}
        pbar(step, msg )
        start_time=train_after(step, start_time,tokenizer,len(batch),msg,len(epoch_dataset))
    msg['file'] = file_path
    return msg

def train_self(model,tokenizer,file_path):
    global global_step
    start_time = time.time()
    t0=time.time()
    # tokenizer = ShangTokenizer(vocab_path="configs/vocab_shang.txt", split_path="configs/spliter_cn.txt")
    epoch_dataset = PretrainSelfDataset(input_file=file_path,tokenizer=tokenizer,task='self',max_tokens=args.max_len,noise=min(0.01,global_step/args.warmup_steps))

    if args.local_rank == -1:
        train_sampler = RandomSampler(epoch_dataset)
    else:
        train_sampler = DistributedSampler(epoch_dataset)
    train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler,collate_fn=epoch_dataset.collate_fn, batch_size=args.train_batch_size,num_workers=6)
    model.train()
    msg={}
    pbar = ProgressBar(n_total=len(train_dataloader), desc=f"self pretrain {file_path[-20:]}")
    for step, batch in enumerate(train_dataloader):
        # def train_batch(batch):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask,type_ids, lm_label_ids,relation_ids = batch
        tasks=[Constants.SCORE_MASK,Constants.SCORE_RELATION ]
        # with autocast():
        outputs, scores= model(input_ids=input_ids, type_ids=type_ids, attention_mask=attention_mask,tasks=tasks)
        # sequence_output,pooled_output=outputs[:2]
        prediction_scores,seq_relationship_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_RELATION]

        masked_lm_loss0 = loss_fct(prediction_scores[0].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
        masked_lm_loss1 = loss_fct(prediction_scores[1].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
        masked_lm_loss=2*masked_lm_loss0+masked_lm_loss1
        sentence_relation_loss0 = loss_fct(seq_relationship_score[0].view(-1, 5), relation_ids.view(-1))
        sentence_relation_loss1 = loss_fct(seq_relationship_score[1].view(-1, 5), relation_ids.view(-1))
        sentence_relation_loss=2*sentence_relation_loss0+sentence_relation_loss1
        loss = masked_lm_loss + sentence_relation_loss


        mask_metric(logits=prediction_scores[0].view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
        sop_metric(logits=seq_relationship_score[0].view(-1, 5), target=relation_ids.view(-1))

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_batch = [loss,masked_lm_loss0, masked_lm_loss1,   sentence_relation_loss0, sentence_relation_loss1]
        loss_batch = [x.cpu().item() for x in loss_batch]
        msg={'总':loss_batch[0],'时':time.time()-t0 ,'lr': scheduler.get_last_lr()[0],'掩0':loss_batch[1],'掩1':loss_batch[2],'系0':loss_batch[3],'系1':loss_batch[4],'len':batch[0].shape[1]}
        pbar(step, msg )
        start_time=train_after(step, start_time,tokenizer,len(batch),msg,len(epoch_dataset))
    msg['file'] = file_path
    return msg

def train_qa(model,tokenizer,file_path):
    global global_step
    start_time = time.time()
    t0=time.time()
    # tokenizer = ShangTokenizer(vocab_path="configs/vocab_shang.txt", split_path="configs/spliter_cn.txt")
    epoch_dataset = PretrainQaDataset(input_file=file_path,tokenizer=tokenizer,max_tokens=args.max_len,noise=min(0.01,global_step/args.warmup_steps))

    if args.local_rank == -1:
        train_sampler = RandomSampler(epoch_dataset)
    else:
        train_sampler = DistributedSampler(epoch_dataset)
    train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler,collate_fn=epoch_dataset.collate_fn, batch_size=args.train_batch_size,num_workers=6)
    model.train()
    msg={}
    pbar = ProgressBar(n_total=len(train_dataloader), desc=f"qa pretrain {file_path}")
    for step, batch in enumerate(train_dataloader):
        # def train_batch(batch):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask,type_ids,lm_label_ids, fake_ids = batch
        tasks=[ Constants.SCORE_MASK,Constants.SCORE_QA ]
        outputs, scores= model(input_ids=input_ids,type_ids=type_ids,attention_mask=input_mask,tasks=tasks)
                # with autocast():
        prediction_scores,qa_score = scores[Constants.SCORE_MASK],scores[Constants.SCORE_QA]

        masked_lm_loss0 = loss_fct(prediction_scores[0].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
        masked_lm_loss1 = loss_fct(prediction_scores[1].view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
        masked_lm_loss=2*masked_lm_loss0+masked_lm_loss1
        qa_loss0 = loss_fct(qa_score[0].view(-1, 2), fake_ids.view(-1))
        qa_loss1 = loss_fct(qa_score[1].view(-1, 2), fake_ids.view(-1))
        qa_loss=2*qa_loss0+qa_loss1

        loss = masked_lm_loss + qa_loss

        mask_metric(logits=prediction_scores[0].view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
        sop_metric(logits=qa_score[0].view(-1, 2), target=fake_ids.view(-1))

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            scaler.scale(loss).backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
        else:
            loss.backward()

        loss_batch = [loss,masked_lm_loss0, masked_lm_loss1,   qa_loss0, qa_loss1]
        loss_batch = [x.cpu().item() for x in loss_batch]
        msg={'总':loss_batch[0],'时':time.time()-t0 ,'lr': scheduler.get_last_lr()[0],'掩0':loss_batch[1],'掩1':loss_batch[2],'系0':loss_batch[3],'系1':loss_batch[4],'len':batch[0].shape[1]}
        pbar(step, msg)
        start_time=train_after(step, start_time,tokenizer,len(batch),msg,len(epoch_dataset))
    msg['file'] = file_path
    return msg

def train_after(step,start_time,tokenizer, batch_size,msg,n_total):
    global global_step,last_time
    loss,masked_lm_loss,next_sentence_loss=msg['总'],msg['掩0'],msg['系0']
    tr_mask_acc.update(mask_metric.value(), n=batch_size)
    tr_sop_acc.update(sop_metric.value(), n=batch_size)
    tr_loss.update(loss, n=1)
    tr_mask_loss.update(masked_lm_loss, n=1)
    tr_sop_loss.update(next_sentence_loss, n=1)
    # writer.add_scalars('data/scalar_group', msg, global_step)

    global_step += 1
    if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
            # scaler.step(optimizer)
            # scaler.update()
            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),  args.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if global_step % args.num_eval_steps == 0:
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
        show_info = f'[Training]:[{epoch}/{args.epochs}]{global_step}/{num_train_optimization_steps} cost:{time.time()-last_time}  ' \
                    f'- ETA: {eta_format}' + "-".join([f' {key}: {value:.4f} ' for key, value in train_logs.items()])
        logger.info(show_info)
        tr_mask_acc.reset()
        tr_sop_acc.reset()
        tr_loss.reset()
        tr_mask_loss.reset()
        tr_sop_loss.reset()
        start_time = now
        last_time=time.time()


    if global_step % args.num_save_steps == 0 or step+1==n_total :
        if args.local_rank in [-1, 0] and args.num_save_steps > 0:
            # Save model checkpoint
            # output_dir = args.output_dir / f'lm-checkpoint-{global_step}'
            output_dir = args.output_dir
            if not output_dir.exists():
                output_dir.mkdir()
            # save model
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            # model_to_save.eval()
            model_to_save.save_pretrained(str(output_dir))
            torch.save(args, str(output_dir / 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), str(output_dir / "optimizer.bin"))
            # save config
            output_config_file = output_dir / CONFIG_NAME
            with open(str(output_config_file), 'w') as f:
                f.write(model_to_save.config.to_json_string())
            # save vocab
            tokenizer.save_vocabulary(output_dir)
            # writer.export_scalars_to_json("./all_scalars.json")
            # writer.close()
    return start_time

if __name__ == '__main__':
    from pathlib import Path

    BASE_DIR = Path('.')
    OUTPUTS_DIR = Path('/media/u/t1/dataset/Poor_all') / 'pretrained'
    # OUTPUTS_DIR = Path("outputs")
    config = {
        # 'data_dir': Path('/media/u/t1/dataset/Poor') / 'dataset',
        'outputs': OUTPUTS_DIR,
        'figure_dir': OUTPUTS_DIR / "figure",
        # 'checkpoint_dir': OUTPUTS_DIR / "none",
        'checkpoint_dir': OUTPUTS_DIR ,
        # 'result_dir': OUTPUTS_DIR / "result",

        'albert_config_path': BASE_DIR / 'configs/albert_config_poor.json',
        'albert_vocab_path': BASE_DIR / 'configs/vocab.txt',
        'albert_spliter_path': BASE_DIR / 'configs/bujian.txt',
        'pretrain_dir': Path("/media/u/t1/data/tokened/rmrb")
    }

    parser = ArgumentParser()
    ## Required parameters
    # parser.add_argument("--data_dir", default='dataset', type=str,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--config_path", default='configs/albert_config_base.json', type=str)
    # parser.add_argument("--vocab_path",default="configs/vocab.txt",type=str)
    # parser.add_argument("--output_dir", default='outputs', type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--model_path", default='', type=str)
    parser.add_argument("--reduce_memory", action="store_true",                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=1,                        help="Number of epochs to train for")
    parser.add_argument("--do_lower_case", action='store_true',                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--num_eval_steps', default=500)
    parser.add_argument('--num_save_steps', default=2000)
    parser.add_argument("--local_rank", type=int, default=-1,                        help="local_rank for distributed training on gpus")
    parser.add_argument("--weight_decay", default=0.01, type=float,                        help="Weight deay if we apply some.")
    parser.add_argument("--no_cuda", action='store_true',                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size", default=128, type=int,                        help="Total batch size for training.")
    parser.add_argument('--loss_scale', type=float, default=0,                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"                             "0 (default value): dynamic loss scaling.\n"                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    # parser.add_argument("--learning_rate", default=0.000176, type=float,                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=42,                        help="random seed for initialization")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--fp16', action='store_true',  default=False,                      help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()

    args.vocab_path=config['albert_vocab_path']
    args.albert_spliter_path=config['albert_spliter_path']
    # args.data_dir=config['data_dir']
    args.output_dir=config['outputs']
    args.config_path=config['albert_config_path']

    # args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    args.pretrain_folders =["/media/u/t1/data/tokened", "/media/u/t1/data/self" ,"/media/u/t1/data/task"  ]
    # args.checkpoint_dir=config["checkpoint_dir"]
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    # writer = SummaryWriter(log_dir=args.checkpoint_dir)  # for tensorboardX
    log_file=str(args.output_dir/ "train.log")
    trained_log=args.output_dir / "trained.log"
    if not trained_log.exists():
        with open(trained_log, "a") as f:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"  {now}  start train")
    init_logger(log_file=log_file)

    # samples_per_epoch = 0
    # logger.info(f"samples_per_epoch: {samples_per_epoch}")
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(f"cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} , distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # seed_everything(args.seed)
    # total_train_examples = samples_per_epoch * args.epochs
    total_train_examples=1.1e7  # 5m~=8000lines   1g~=E7
    num_train_optimization_steps = int( total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    args.warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)

    bert_config = AlbertConfig.from_pretrained(args.config_path)
    args.max_len=bert_config.max_position_embeddings
    args.max_len=512
    model = AlbertForPreTraining(config=bert_config)
    args.model_path = str((args.output_dir / "lm-checkpoint").absolute())
    if os.path.exists(args.model_path):
        logger.info(f" loadding {args.model_path} ")
        model = AlbertForPreTraining.from_pretrained(args.model_path)
    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    for name,param in param_optimizer:
        logger.info(f" param size {name}-->{param.size()} ")
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = AdaFactor(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if os.path.exists(args.model_path):
        optimizer.load_state_dict(torch.load(args.model_path + "/optimizer.bin"))
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    mask_metric = LMAccuracy()
    sop_metric = LMAccuracy()
    tr_mask_acc = AverageMeter()
    tr_sop_acc = AverageMeter()
    tr_loss = AverageMeter()
    tr_mask_loss = AverageMeter()
    tr_sop_loss = AverageMeter()
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_examples}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = {num_train_optimization_steps}")
    logger.info(f"  warmup_steps = {args.warmup_steps}")
    # seed_everything(args.seed)  # Added here for reproducibility
    tokenizer = ShangTokenizer(vocab_path="configs/vocab.txt", bujian_path="configs/bujian.txt")
    for epoch in range(args.epochs):
        files=[]
        files+=glob.glob(r"/media/u/t1/data/tokened/*/*.txt")
        # files+=glob.glob(r"/media/u/t1/data/self/*/*.txt")
        files+=glob.glob(r"/media/u/t1/data/qa/*/*.txt")
        random.shuffle(files)
        # files=np.random.choice(files,100,replace=False)
        logger.info(  f" \n\n ==== training {len(files)} files ==== \n")  #13000
        # files.sort()
        for fid,file_path in enumerate(files):
            file_path=str(file_path)
            trained=False
            with open(trained_log) as f:
                for line in f.readlines():
                    if file_path in line:
                        trained=True
                        break
            if trained:
                logger.info(f" {file_path}  already trained")
                continue
            rand=random.random()

            if rand<0.1:
                args.max_len = 1024
                args.train_batch_size = 8
            if rand<0.2:
                args.max_len=512
                args.train_batch_size=20
            elif rand<0.4:
                args.max_len=256
                args.train_batch_size=52
            elif rand<0.7:
                args.max_len = 128
                args.train_batch_size = 115
            elif rand<0.9:
                args.max_len = 64
                args.train_batch_size = 236
            else :
                args.max_len=32
                args.train_batch_size=500

            # args.max_len = 512
            # args.train_batch_size = 20
            # args.train_batch_size = int(args.train_batch_size * 0.8)
            t0=time.time()
            logger.info(f" fid {fid} folder {len(files)}  max_len {args.max_len} batch_size {args.train_batch_size}")
            msg={
                "fid":fid,
                "file":file_path,
            }
            trace=""
            if "tokened" in file_path:
                try:
                    trace=train_tokened(model,tokenizer,file_path)
                except Exception as e:
                    try:
                        args.train_batch_size =int(args.train_batch_size*0.8)
                        trace=train_tokened(model,tokenizer,file_path)
                    except Exception as e:
                        print(f"args.max_len:{args.max_len} args.train_batch_size:{args.train_batch_size}  error:{traceback.format_exc()} ")

            elif "self" in file_path:
                trace=train_self(model,tokenizer,file_path)
            elif "qa" in file_path:
                # trace=train_qa(model,tokenizer,file_path)
                try:
                    trace=train_qa(model,tokenizer,file_path)
                except Exception as e:
                    try:
                        args.train_batch_size =int(args.train_batch_size*0.8)
                        trace=train_qa(model,tokenizer,file_path)
                    except Exception as e:
                        print(f"args.max_len:{args.max_len} args.train_batch_size:{args.train_batch_size}  error:{traceback.format_exc()} ")

            torch.cuda.empty_cache()
            cost = time.time() - t0
            logger.info(f"  fid{fid} {file_path} trainned {cost}s ")

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
 CUDA_VISIBLE_DEVICES="0"  python3  run_pretraining.py | tee pretrain.log

626 files ->100
14g->2.2g

self 1.2g
tokened zhwiki    1.5g
qa baike_qa  0.75g  150w 1.5e6
total 2g

expr1
none  
all bujian       only self  
all gradation  
all sentRelation
all bujian gradation sentRelation

'''