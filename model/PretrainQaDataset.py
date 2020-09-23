"""Tokenization classes."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
import os
import sys
import unicodedata
import random
import six
import logging
import sentencepiece as spm
from torch.utils.data import Dataset
import torch
import numpy as np

from callback.progressbar import ProgressBar
from configs import Constants

# logger = logging.getLogger(__name__)
from model.tokenization_shang import Sentence

logger = logging.getLogger(__name__)
# FORMAT = '%(pathname)s %(filename)s  %(funcName)s %(lineno)d %(asctime)-15s  %(message)s'
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)


class PretrainQaDataset(Dataset):
  def __init__(self, input_file, tokenizer,task='answer', max_tokens=256,noise=0):
    task_dict = {'self': 0, 'answer': 1, 'question': 2}
    self.taskid=task_dict.get(task, Constants.TASK_SELF)
    self.tokenizer = tokenizer
    self.max_tokens = max_tokens
    self.input_file=input_file
    self.doc=self.load_file(input_file)
    logger.info(f"PretrainQaDataset {input_file}装载{len(self.doc)}完毕")

  def load_file(self,path):
    with open(path) as f:
      doc = f.readlines()
    folder = []
    for line in doc:
      line = line.strip()
      if  line:
        sents=line.split('\t')
        if len(sents)<2:
          continue
        q,a=sents[:2]
        c=sents[2] if len(sents)>2 else ''
        pair=[q,a,c]
        for i in range(len(pair)):
          if len(pair[i])>self.max_tokens-8:
            part=self.max_tokens//2-4
            pair[i]=pair[i][:part]+pair[i][-part:]
        if pair[0] and pair[1]:
          folder.append(pair)
    folder = [x for x in folder if x]
    return folder

  def __len__(self):
    return len(self.doc)
  def collate_fn(self,batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask,token_type_ids, token_label,all_lens, relation_lable = zip(*batch)
    max_len = max(all_lens)
    all_input_ids = np.array(all_input_ids)[:, :max_len]
    all_attention_mask = np.array(all_attention_mask)[:, :max_len]
    token_type_ids = np.array(token_type_ids)[:, :max_len]
    token_label = np.array(token_label)[:, :max_len]

    return (torch.LongTensor(all_input_ids), torch.LongTensor(all_attention_mask),torch.LongTensor(token_type_ids),torch.LongTensor(token_label), torch.LongTensor(relation_lable))

  def __getitem__(self, idx):
    wrong=self.doc[random.randint(0,len(self.doc)-1)]
    fake=0
    q, a, c = self.doc[idx//3]
    rand=random.random()
    # if False:
    if rand>0.5:
      fake=1
      if rand>0.75:
        q=wrong[0]
      else:
        a=wrong[1]

    qac, input_mask,token_type_ids,length, qac_label= self.qa_features([q,a,c],max_len=self.max_tokens)
    # qac =torch.LongTensor(np.array(qac))
    # input_mask=torch.LongTensor(np.array(input_mask))
    # qac_label=torch.LongTensor(np.array(qac_label))
    # task_type_id = torch.tensor(qac.shape).new_full(qac.shape, Constants.TASK_QA)

    # return qac , input_mask, qac_label,task_type_id,fake
    # return np.array(qac),np.array(input_mask),np.array(token_type_ids),np.array(qac_label),length,fake
    return  (qac, input_mask, token_type_ids, qac_label,length,fake)

  def qa_features(self,qac,max_len):
    sentq = Sentence(qac[0], tokenizer=self.tokenizer)
    senta = Sentence(qac[1], tokenizer=self.tokenizer)
    coupleq=self.plain_sents([sentq])
    couplea=self.plain_sents([senta])
    for i in range(len(coupleq)):
      coupleq[i],couplea[i]=self.truncate_pair(coupleq[i],couplea[i],max_len-5)
      tokenq,token_labelq=coupleq
      tokena,token_labela=couplea
      tokenq=[Constants.TOKEN_BOQ]+tokenq+[Constants.TOKEN_EOQ]
      token_labelq = [Constants.TOKEN_BOQ] + token_labelq + [Constants.TOKEN_EOQ]
      tokena=[Constants.TOKEN_BOA]+tokena+[Constants.TOKEN_EOA]
      token_labela = [Constants.TOKEN_BOA] + token_labela + [Constants.TOKEN_EOA]
    res = max_len - len(tokenq) - len(tokena) - 3
    tokenc,token_labelc=[],[]
    if res > 0  and qac[2]:
      sentc = Sentence(qac[2], tokenizer=self.tokenizer)
      couplec=self.plain_sents([sentc])
      for i in range(len(couplec)):
        couplec[i] = self.truncate_one(couplec[i], res)
      tokenc,token_labelc=couplec
      tokenc=[Constants.TOKEN_BOC]+tokenc+[Constants.TOKEN_EOC]
      token_labelc = [Constants.TOKEN_BOC] + token_labelc + [Constants.TOKEN_EOC]

    if random.random()>0.5:
      tokenq , tokena=tokenq,tokena
      token_labelq , token_labela=token_labela,token_labelq

    tokens=[Constants.TOKEN_CLS]+tokenq+tokena+tokenc
    length=len(tokens)
    tokens_label=[Constants.TOKEN_CLS]+token_labelq+token_labela+token_labelc
    input_mask=[1]*len(tokens)+[0]*(self.max_tokens-length)

    tokens += [Constants.TOKEN_PAD] *(self.max_tokens-length)

    tokens_label += [Constants.TOKEN_PAD]*(self.max_tokens-length)
    type_ids = []
    k=0
    for j, c in enumerate(tokens):
      type_ids.append(k)
      if c in [Constants.TOKEN_EOQ, Constants.TOKEN_EOA]:
        k+=1
    tokens=self.tokenizer.convert_tokens_to_ids(tokens)
    tokens_label=self.tokenizer.convert_tokens_to_ids(tokens_label)
    return tokens , input_mask,type_ids,length, tokens_label

  def truncate_one(self, seq, max_len):
    seq_len=len(seq)
    if seq_len>max_len:
      # idx = random.randint(0,max_len-1)
      idx = max_len//2 # solid
      seq=seq[:idx] + seq[-(max_len-idx):]
    if len(seq)>max_len:
      assert len(seq)<=max_len
    return seq

  def truncate_pair(self,tokensa,tokensb,max_len):
    if len(tokensa)+len(tokensb)>max_len:
      if len(tokensa)>=len(tokensb):
        if len(tokensb)>=max_len//2:
          tokensa = self.truncate_one(tokensa,  max_len//2)
          tokensb = self.truncate_one(tokensb,  max_len//2)
        else:
          tokensa = self.truncate_one(tokensa,  max_len - len(tokensb))
      else:
        if len(tokensa)>=max_len//2:
          tokensa = self.truncate_one(tokensa,  max_len//2)
          tokensb = self.truncate_one(tokensb,  max_len//2)
        else:
          tokensb = self.truncate_one(tokensb,  max_len - len(tokensa))
    assert len(tokensa)+len(tokensb)<=max_len
    return tokensa,tokensb

  def plain(self,mat):
    seq=[y  for x in mat  for y in x   ]
    return seq

  def plain_sents(self,sents):
    tokens,token_label=[],[]
    for sent in sents:
      if len(sent.get_features())!=2:
        c=0
      t, l=sent.get_features()
      tokens+=t
      token_label+=l
    assert len(tokens)==len(token_label)
    return [tokens,token_label]



