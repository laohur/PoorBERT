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
from model.tokenization_shang import Sentence, SentenceWorded

logger = logging.getLogger(__name__)
# FORMAT = '%(pathname)s %(filename)s  %(funcName)s %(lineno)d %(asctime)-15s  %(message)s'
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)



class PretrainTokenedDataset(Dataset):
  def __init__(self, input_file, tokenizer,noise=0.01,task='self', max_tokens=256):
    task_dict = {'self': 0, 'answer': 1, 'question': 2}
    self.taskid=task_dict.get(task, Constants.TASK_SELF)
    self.tokenizer = tokenizer
    self.max_tokens = max_tokens
    self.input_file=input_file
    self.folder=self.load_file(input_file)
    self.total_lines=sum( [ len(x) for x in self.folder  ] )
    self.doc_bounds=[]
    if self.folder:
      self.doc_bounds=[(0,len(self.folder[0]))]*len(self.folder)
    for i in range(1,len(self.folder)):
      start=self.doc_bounds[i-1][1]
      self.doc_bounds[i]=( start,  start+len(self.folder[i]))
    logger.info(f"PretrainTokenedDataset 装载{input_file}完毕{self.total_lines}")

  def load_file0(self,path):
    with open(path) as f:
      docs = f.readlines()
    folder = []
    doc = []
    for line in docs:
      line = line.strip()
      if not line:
        folder.append(doc)
        doc = []
      else:
        while len(line)>self.max_tokens-3:
          doc.append(line[:self.max_tokens-3])
          line=line[self.max_tokens-3:]
        if line:
          doc.append(line)
    if doc:
      folder.append(doc)
    folder = [x for x in folder if x]
    return folder

  def load_file(self,path):
    with open(path) as f:
      docs = f.readlines()
    folder = []
    doc = []
    line=""
    for l in docs:
      l = l.strip()
      if not l:
        doc.append(line)
        line = l
        folder.append(doc)
        doc = []
      else:
        if len(''.join(line.split(' ')))+len(''.join(l.split(' ')))<self.max_tokens*0.5:
          line+=l+' '
        else:
          doc.append(line)
          line=l
    if line:
      doc.append(line)
    if doc:
      folder.append(doc)
    folder = [x for x in folder if x]
    # lens=[len(x) for x in folder]
    return folder

  def __len__(self):
    return self.total_lines

  def collate_fn(self,batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask,type_ids,token_label,char_label,word_label, all_lens, relation_lable = zip(*batch)
    max_len = max(all_lens)
    # all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = batch
    all_input_ids = np.array(all_input_ids)[:, :max_len]
    all_attention_mask = np.array(all_attention_mask)[:, :max_len]
    type_ids = np.array(type_ids)[:, :max_len]
    token_label = np.array(token_label)[:, :max_len]
    char_label = np.array(char_label)[:, :max_len]
    word_label = np.array(word_label)[:, :max_len]

    return (torch.LongTensor(all_input_ids), torch.LongTensor(all_attention_mask),torch.LongTensor(type_ids),torch.LongTensor(token_label),torch.LongTensor(char_label), torch.LongTensor(word_label), torch.LongTensor(relation_lable))

  def __getitem__(self, idx):
    ( tokens,input_mask,type_ids,token_label,char_label,word_label,length,relation_lable) =self.self_feature(idx)
    # return (np.array(tokens),np.array(input_mask),np.array(type_ids),np.array(token_label),np.array(char_label),np.array(word_label),length,relation_lable)
    return  (tokens, input_mask, type_ids, token_label, char_label, word_label, length, relation_lable)


  def self_feature(self, idx):
    docid,lno=self.lineid2docid(idx)
    doc=self.folder[docid]
    rand=random.random()
    relation_lable=0
    if rand<=0.2 or len(doc)==1 :  # ->
    # if True:
      couplea, coupleb=self.grab1(doc,lno,max_len=self.max_tokens-5)
      if not coupleb:
        tokens= [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS]
        token_label= [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS]
        char_label= [0] + couplea[2]+[0]
        word_label=[0] + couplea[3]+[0]
      else:
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]
        char_label = [0] + couplea[2] + [0]+ [0] + coupleb[2] + [0]
        word_label = [0] + couplea[3] + [0]+ [0] + coupleb[3] + [0]

    elif rand<0.4 and lno>=1 : #<-
      couplea, coupleb=self.grab1(doc,lno,max_len=self.max_tokens-5)
      if not coupleb:
        tokens= [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS]
        token_label= [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS]
        char_label= [0] + couplea[2]+[0]
        word_label=[0] + couplea[3]+[0]
      else:
        couplea, coupleb= coupleb, couplea
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]
        char_label = [0] + couplea[2] + [0]+ [0] + coupleb[2] + [0]
        word_label = [0] + couplea[3] + [0]+ [0] + coupleb[3] + [0]
        relation_lable=1
    elif rand<0.6 and lno>=1:  # before
        lnob=random.randint(0,lno-1)
        couplea, coupleb=self.grab2(doc,lno,doc,lnob,max_len=self.max_tokens-5)
        # tokens, char_label, word_label=(  (couplea[x]+coupleb[x])   for x in range(3)  )
        relation_lable=2
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]
        char_label = [0] + couplea[2] + [0]+ [0] + coupleb[2] + [0]
        word_label = [0] + couplea[3] + [0]+ [0] + coupleb[3] + [0]

    elif rand<0.8 and lno<=len(doc)-2: # after
        lnob=random.randint(lno+1,len(doc)-1)
        couplea, coupleb=self.grab2(doc,lno,doc,lnob,max_len=self.max_tokens-5)
        # tokens, char_label, word_label=(  (couplea[x]+coupleb[x])   for x in range(3)  )
        relation_lable=3
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]
        char_label = [0] + couplea[2] + [0]+ [0] + coupleb[2] + [0]
        word_label = [0] + couplea[3] + [0]+ [0] + coupleb[3] + [0]
    else: # outer doc
      docidb=docid
      while docidb==docid:
        docidb=random.randint(0,len(self.folder)-1)
      docb=self.folder[docidb]
      lnob=random.randint(0,len(docb)-1)
      couplea, coupleb = self.grab2(doc, lno, docb, lnob,max_len=self.max_tokens-5)
      relation_lable = 4
      tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
      token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]
      char_label = [0] + couplea[2] + [0] + [0] + coupleb[2] + [0]
      word_label = [0] + couplea[3] + [0] + [0] + coupleb[3] + [0]

    tokens= [Constants.TOKEN_CLS] + tokens
    length=len(tokens)
    token_label= [Constants.TOKEN_CLS] + token_label
    char_label=[0]+char_label
    word_label=[0]+word_label

    input_mask=[1]*length+[0]*(self.max_tokens-length)
    tokens+= [Constants.TOKEN_PAD] *(self.max_tokens-length)
    type_ids=[]
    k=0
    for j,c in enumerate(tokens):
      type_ids.append(k)
      if c==Constants.TOKEN_EOS:
        k+=1
    tokens=self.tokenizer.convert_tokens_to_ids(tokens)

    token_label+= [Constants.TOKEN_PAD] *(self.max_tokens-length)
    token_label=self.tokenizer.convert_tokens_to_ids(token_label)

    char_label+=[0]*(self.max_tokens-length)
    word_label+=[0]*(self.max_tokens-length)

    return (tokens,input_mask,type_ids,token_label,char_label,word_label,length,relation_lable)

  def lineid2docid(self,lineid):
    for i in range(len(self.doc_bounds)):
      if self.doc_bounds[i][0] <= lineid < self.doc_bounds[i][1]:
        return i, lineid-self.doc_bounds[i][0]
    return -1,-1

  def get_line(self,doc,offset):
    return doc[offset]

  def grab1(self,doc,lno,max_len): # grab continous lines
    sent = SentenceWorded(doc[lno], tokenizer=self.tokenizer)
    lengths=[sent.length]
    sents=[sent]
    lno+=1
    while sum(lengths)<max_len and lno<len(doc):
      sent = SentenceWorded(doc[lno], tokenizer=self.tokenizer)
      if sum(lengths)+sent.length<=max_len:
        sents.append(sent)
        lengths.append(sent.length)
        lno+=1
      else:
        break

    if len(lengths)==1:
      couple=self.plain_sents(sents)  #tokens, char_label, word_label
      for i in range(len(couple)):
        couple[i]=self.truncate_one(couple[i], max_len)
      assert len(couple)==4
      assert  len(couple[0])== len(couple[1])== len(couple[2])== len(couple[3])
      return couple,None

    seg=random.randint(0,len(lengths)-1)
    sentsa,sentsb=sents[:seg],sents[seg:]
    couplea= self.plain_sents(sentsa)
    coupleb = self.plain_sents(sentsb)
    for i in range(len(couplea)):
      couplea[i],coupleb[i]=self.truncate_pair(couplea[i],coupleb[i],max_len=max_len)

    assert len(couplea)==len(coupleb) == 4
    assert len(couplea[0]+coupleb[0]) == len(couplea[1])+len(coupleb[1]) == len(couplea[2])+len(coupleb[2]) == len(couplea[3])+len(coupleb[3])
    return couplea,coupleb

  def grab2(self,doca, lnoa,docb,lnob,max_len):
      senta = SentenceWorded(doca[lnoa], tokenizer=self.tokenizer)
      sentb= SentenceWorded(docb[lnob], tokenizer=self.tokenizer)
      lengthsa,lengthsb = [senta.length], [sentb.length]
      sentsa,sentsb = [senta],[sentb]

      lnoa+=1
      lnob+=1
      while sum(lengthsa)+sum(lengthsb)<max_len and lnoa<len(doca) and lnob<len(docb):
        senta = SentenceWorded(doca[lnoa], tokenizer=self.tokenizer)
        sentb = SentenceWorded(docb[lnob], tokenizer=self.tokenizer)
        if sum(lengthsa)+sum(lengthsb)+senta.length+sentb.length<=max_len:
          sentsa.append(senta)
          sentsb.append(sentb)
          lengthsa.append(senta.length)
          lengthsb.append(sentb.length)
          lnoa += 1
          lnob += 1
        else:
          break
      couplea= self.plain_sents(sentsa)
      coupleb = self.plain_sents(sentsb)
      for i in range(len(couplea)):
        couplea[i],coupleb[i]=self.truncate_pair(couplea[i],coupleb[i],max_len=max_len)

      assert len(couplea)==len(coupleb) == 4
      assert len(couplea[0]+coupleb[0]) == len(couplea[1])+len(coupleb[1]) == len(couplea[2])+len(coupleb[2]) == len(couplea[3])+len(coupleb[3])
      return couplea,coupleb

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

  def plain(self,mat):  # (self.tokens, self.token_label,self.char_label, self.word_label)
    seq=[y  for x in mat  for y in x   ]
    return seq

  def plain_sents(self,sents):
    tokens,token_label, char_label, word_label=[],[],[],[]
    for sent in sents:
      t, l,  c, w=sent.get_features()
      tokens+=t
      token_label+=l
      char_label+=c
      word_label+=w
    assert len(tokens)==len(token_label)==len(char_label)==len(word_label)
    return [tokens,token_label, char_label, word_label]



