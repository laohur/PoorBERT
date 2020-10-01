import os
import json

import numpy as np
import torch
import sys
sys.path.append("..")
sys.path.append(".")

def cal_acc(preds, labels):
    p=torch.argmax(preds,1).detach().cpu().numpy()
    l=labels.detach().cpu().numpy()
    score = (p == l).mean()
    return score

def truncate_one( seq, max_len):
    seq_len=len(seq)
    if seq_len>max_len:
      # idx = random.randint(0,max_len-1)
      idx = max_len//2 # solid
      seq=seq[:idx] + seq[-(max_len-idx):]
    if len(seq)>max_len:
      assert len(seq)<=max_len
    return seq

def truncate_pair(tokensa,tokensb,max_len):
    if len(tokensa)+len(tokensb)>max_len:
      if len(tokensa)>=len(tokensb):
        if len(tokensb)>max_len//2:
          tokensa = truncate_one(tokensa,  max_len//2)
          tokensb = truncate_one(tokensb,  max_len//2)
        else:
          tokensa = truncate_one(tokensa,  max_len - len(tokensb))
      else:
        if len(tokensa)>max_len//2:
          tokensa = truncate_one(tokensa,  max_len//2)
          tokensb = truncate_one(tokensb,  max_len//2)
        else:
          tokensb = truncate_one(tokensb,  max_len - len(tokensa))
    assert len(tokensa)+len(tokensb)<=max_len
    return tokensa,tokensb


def collate_qa(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids,all_attention_mask,type_ids, all_lens, start_positions,end_positions=zip(*batch)
    max_len = max(all_lens)
    all_input_ids = np.array(all_input_ids)[:, :max_len]
    all_attention_mask = np.array(all_attention_mask)[:, :max_len]
    type_ids = np.array(type_ids)[:, :max_len]
    return (torch.LongTensor(all_input_ids) , torch.LongTensor(all_attention_mask), torch.LongTensor(type_ids), torch.LongTensor(start_positions),torch.LongTensor(end_positions))

def collate_cls(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids,all_attention_mask,type_ids, all_lens, all_labels=zip(*batch)
    max_len = max(all_lens)
    all_input_ids = np.array(all_input_ids)[:, :max_len]
    if isinstance(all_labels[0],np.ndarray):
        all_labels = np.array(all_labels)[:, :max_len]
    all_attention_mask = np.array(all_attention_mask)[:, :max_len]
    type_ids = np.array(type_ids)[:, :max_len]

    return (torch.LongTensor(all_input_ids) , torch.LongTensor(all_attention_mask), torch.LongTensor(type_ids), torch.LongTensor(all_labels))


class TaskConfig:
    """ Hyperparameters for training """
    # seed: int = 42 # random seed
    task_name="task"
    collate_fn=None
    model_name="none"
    # model_type="Poor"
    model_name_or_path=""
    model_config_path=""
    vocab_file="config/vocab.txt"
    bujian_file="config/bujian.txt"
    noise=0
    max_len=1024
    batch_size: int = 6
    gradient_accumlengthulation_steps=1
    learning_rate = 2.5e-5 # learning rate
    n_epochs: int = 5 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup_proportion: float = 0.1
    logging_steps:int=100
    save_steps: int = 100 # interval for saving model
    # total_steps: int = 100000 # total number of steps to train
    # max_seq_length=256
    data_dir=""
    output_dir=""
    local_rank=-1
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    gradient_accumulation_steps=1
    eval_all_checkpoints=0.0
    adam_epsilon=1e-6
    max_grad_norm=1.0
    weight_decay=0.01

    pin_memory = True
    num_workers=0
    timeout=1
    no_cuda=False
    n_gpu=1
    fp16=False

    output_mode = "classification"   # classification  span
    train_file="train.txt"
    valid_file="valid.txt"
    test_file="test.txt"
    output_submit_file=""
    TaskDataset=None
    labels= ["0", "1"]
    # def __init__(self,**kwargs):-
    #     for k,v in kwargs.items():
    #         setattr(self,k,v)
    def __init__(self,dic):
        for k,v in dic.items():
            setattr(self,k,v)
        if self.output_mode=="qa":
            self.collate_fn=collate_qa
        else:
            self.collate_fn=collate_cls
        model_dir='/media/u/t1/dataset/PoorBERT/'
        pretrained = model_dir+"pretrained/"
        data_dir = '/media/u/t1/dataset/CLUEdatasets/'
        dic2={
            "model_dir":pretrained,
            "model_name_or_path": pretrained,
            "data_dir": data_dir + self.task_name,
            "vocab_file": pretrained + f"vocab.txt",
            "bujian_file": pretrained + f"bujian.txt",
            "model_config_path": pretrained + f"config.json",
            "output_dir": model_dir + f"{self.task_name}",
        }
        for k,v in dic2.items():
            setattr(self,k,v)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        submit_dir=model_dir + f"clue_submit"
        if not os.path.exists(submit_dir):
            os.makedirs(submit_dir)
        self.output_submit_file = os.path.join(submit_dir, f"{self.task_name}_prediction.json")

    @classmethod
    def from_dict(cls, dic): # load config from json
        return cls(**dic)

    @classmethod
    def from_json(cls, jsons): # load config from json
        return cls(**json.loads(jsons))

    @classmethod
    def from_file(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

def index_of(context,patten):
    m,n=len(context),len(patten)
    for i in range(m-n+1):
        if ''.join(context[i:i+n])==''.join(patten):
            return i
    return 0

def find_span(logits,threshold=0.5):
    start,end=2,len(logits)-1
    for i in range(2,len(logits)):
        if logits[i][1] > logits[i][0]:
            start=i
            break
    for j in range(start,len(logits)):
        if logits[j][1]<logits[i][0]:
            end=j
            break
    return start,end
