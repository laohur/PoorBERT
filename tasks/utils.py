import csv
import os
import sys
import copy
import json

import numpy as np
import torch
import sys
sys.path.append("..")
sys.path.append(".")

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label,input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(line.strip().split("_!_"))
            return lines

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



def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids,all_attention_mask,type_ids, all_lens, all_labels=zip(*batch)
    max_len = max(all_lens)
    # all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = batch
    all_input_ids = np.array(all_input_ids)[:, :max_len]
    if isinstance(all_labels[0],np.ndarray):
        all_labels = np.array(all_labels)[:, :max_len]
    all_attention_mask = np.array(all_attention_mask)[:, :max_len]
    type_ids = np.array(type_ids)[:, :max_len]

    # all_token_type_ids = all_token_type_ids[:, :max_len]
    # return all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    # return (torch.Tensor(x) for x in (all_input_ids, all_attention_mask, all_labels))
    return (torch.LongTensor(all_input_ids) , torch.LongTensor(all_attention_mask), torch.LongTensor(type_ids), torch.LongTensor(all_labels))

class TaskConfig:
    """ Hyperparameters for training """
    # seed: int = 42 # random seed
    task_name="task"
    model_name="none"
    # model_type="Poor"
    model_name_or_path=""
    model_config_path=""
    vocab_file="config/vocab.txt"
    bujian_file="config/bujian.txt"
    noise=0
    batch_size: int = 50
    gradient_accumlengthulation_steps=1
    max_len=256
    learning_rate = 2e-5 # learning rate
    n_epochs: int = 5 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup_proportion: float = 0.1
    logging_steps:int=10
    save_steps: int = 10 # interval for saving model
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

    output_mode = "classification"
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

        model_dir='/media/u/t1/dataset/Poor_all/'
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
        prefix=self.task_name
        if "cmrc" in prefix:
            prefix="cmrc2018"
        self.output_submit_file = os.path.join(submit_dir, f"{prefix}_prediction.json")

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
    # n_class=2
    start,end=2,len(logits)-1
    for i in range(2,len(logits)):
        # if logits[i][1]>threshold:
        if logits[i][1] > logits[i][0]:
            start=i
            break
    for j in range(start,len(logits)):
        if logits[j][1]<logits[i][0]:
            end=j
            break
    return start,end
