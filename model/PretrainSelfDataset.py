"""Tokenization classes."""

from __future__ import (absolute_import, division, print_function,                      unicode_literals)
import random
import logging
from torch.utils.data import Dataset
import torch
import numpy as np
from configs import Constants
from model.tokenization_shang import Sentence

logger = logging.getLogger(__name__)
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)

class PretrainSelfDataset(Dataset):
  def __init__(self, input_file, tokenizer,use_relation=False,task='self', max_tokens=256):
    task_dict = {'self': 0, 'answer': 1, 'question': 2}
    self.taskid=task_dict.get(task, Constants.TASK_SELF)
    self.tokenizer = tokenizer
    self.max_tokens = max_tokens
    self.use_relation=use_relation
    self.input_file=input_file
    self.folder=self.load_file(input_file)
    self.total_lines=sum( [ len(x) for x in self.folder  ] )
    self.doc_bounds=[(0,len(self.folder[0]))]*len(self.folder)
    for i in range(1,len(self.folder)):
      start=self.doc_bounds[i-1][1]
      self.doc_bounds[i]=( start,  start+len(self.folder[i]))
    logger.info(f"PretrainSelfDataset 装载{input_file}完毕{self.total_lines}")

  def load_file(self,path):
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
    folder = [x for x in folder if x]
    return folder

  def __len__(self):
    return self.total_lines

  def collate_fn(self,batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, type_ids, token_label,all_lens, relation_lable = zip(*batch)
    max_len = max(all_lens)
    # all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = batch
    all_input_ids = np.array(all_input_ids)[:, :max_len]
    all_attention_mask = np.array(all_attention_mask)[:, :max_len]
    type_ids = np.array(type_ids)[:, :max_len]
    token_label = np.array(token_label)[:, :max_len]

    return (torch.LongTensor(all_input_ids), torch.LongTensor(all_attention_mask),torch.LongTensor(type_ids),torch.LongTensor(token_label), torch.LongTensor(relation_lable))

  def __getitem__(self, idx):
    ( tokens,input_mask,type_ids,token_label,length,relation_lable ) =self.self_feature(idx)

    # return ( np.array(tokens),np.array(input_mask),np.array(type_ids),np.array(token_label),length,relation_lable  )
    return ( np.array(tokens),np.array(input_mask),np.array(type_ids),np.array(token_label),length,relation_lable  )

  def self_feature(self, idx):
    docid,lno=self.lineid2docid(idx)
    doc=self.folder[docid]
    rand=random.random()
    relation_lable=0
    couplea, coupleb=self.grab2(doc,lno,doc,lno,max_len=self.max_tokens-5)
    # 0.2-> 0.2<- 0.2b 0.2a 0.2o
    if not self.use_relation or rand<=0.2 or len(doc)==1 :  # ->
      couplea, coupleb=self.grab1(doc,lno,max_len=self.max_tokens-5)
      if not coupleb:
        tokens= [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS]
        token_label= [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS]
      else:
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]

    elif rand<0.4 and lno>=1 : #<-
      couplea, coupleb=self.grab1(doc,lno,max_len=self.max_tokens-5)
      if not coupleb:
        tokens= [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS]
        token_label= [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS]
      else:
        couplea, coupleb= coupleb, couplea
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]
        relation_lable=1
    elif rand<0.6 and lno>=1:  # before
        lnob=random.randint(0,lno-1)
        couplea, coupleb=self.grab2(doc,lno,doc,lnob,max_len=self.max_tokens-5)
        # tokens, char_label, word_label=(  (couplea[x]+coupleb[x])   for x in range(3)  )
        relation_lable=2
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]

    elif rand<0.8 and lno<=len(doc)-2: # after
        lnob=random.randint(lno+1,len(doc)-1)
        couplea, coupleb=self.grab2(doc,lno,doc,lnob,max_len=self.max_tokens-5)
        # tokens, char_label, word_label=(  (couplea[x]+coupleb[x])   for x in range(3)  )
        relation_lable=3
        tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
        token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]
    else: # 0.2 outer doc
      docidb=docid
      while docidb==docid:
        docidb=random.randint(0,len(self.folder)-1)
      docb=self.folder[docidb]
      lnob=random.randint(0,len(docb)-1)
      couplea, coupleb = self.grab2(doc, lno, docb, lnob,max_len=self.max_tokens-5)
      relation_lable = 4
      tokens = [Constants.TOKEN_BOS] + couplea[0] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[0] + [Constants.TOKEN_EOS]
      token_label = [Constants.TOKEN_BOS] + couplea[1] + [Constants.TOKEN_EOS] + [Constants.TOKEN_BOS] + coupleb[1] + [Constants.TOKEN_EOS]

    tokens= [Constants.TOKEN_CLS] + tokens
    length=len(tokens)
    token_label= [Constants.TOKEN_CLS] + token_label
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
    return (tokens,input_mask,type_ids,token_label,length,relation_lable)


  def lineid2docid(self,lineid):
    for i in range(len(self.doc_bounds)):
      if self.doc_bounds[i][0] <= lineid < self.doc_bounds[i][1]:
        return i, lineid-self.doc_bounds[i][0]
    return -1,-1

  def get_line(self,doc,offset):
    return doc[offset]

  def grab1(self,doc,lno,max_len): # grab continous lines
    sent = Sentence(doc[lno], tokenizer=self.tokenizer)
    lengths=[sent.length]
    sents=[sent]
    lno+=1
    while sum(lengths)<max_len and lno<len(doc):
      sent = Sentence(doc[lno], tokenizer=self.tokenizer)
      if sum(lengths)+sent.length<=max_len:
        sents.append(sent)
        lengths.append(sent.length)
        lno+=1
      else:
        break

    if len(lengths)==1:
      couple=self.plain_sents(sents)  #tokens, label
      for i in range(len(couple)):
        couple[i]=self.truncate_one(couple[i], max_len)
      assert len(couple)==2
      assert  len(couple[0])== len(couple[1])
      return couple,None

    seg=random.randint(0,len(lengths)-1)
    sentsa,sentsb=sents[:seg],sents[seg:]
    couplea= self.plain_sents(sentsa)
    coupleb = self.plain_sents(sentsb)
    for i in range(len(couplea)):
      couplea[i],coupleb[i]=self.truncate_pair(couplea[i],coupleb[i],max_len=max_len)

    assert len(couplea)==len(coupleb) == 2
    assert len(couplea[0]+coupleb[0]) == len(couplea[1])+len(coupleb[1])
    return couplea,coupleb

  def grab2(self,doca, lnoa,docb,lnob,max_len):
      senta = Sentence(doca[lnoa],tokenizer=self.tokenizer)
      sentb= Sentence(docb[lnob],tokenizer=self.tokenizer)
      lengthsa,lengthsb = [senta.length], [sentb.length]
      sentsa,sentsb = [senta],[sentb]

      lnoa+=1
      lnob+=1
      while sum(lengthsa)+sum(lengthsb)<max_len and lnoa<len(doca) and lnob<len(docb):
        senta = Sentence(doca[lnoa],tokenizer=self.tokenizer)
        sentb = Sentence(docb[lnob],tokenizer=self.tokenizer)
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
        if i>0:
          assert len(couplea[i-1])==len(couplea[i])
          assert len(coupleb[i - 1]) == len(coupleb[i])
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
        if len(tokensb)>max_len//2:
          tokensa = self.truncate_one(tokensa,  max_len//2)
          tokensb = self.truncate_one(tokensb,  max_len//2)
        else:
          tokensa = self.truncate_one(tokensa,  max_len - len(tokensb))
      else:
        if len(tokensa)>max_len//2:
          tokensa = self.truncate_one(tokensa,  max_len//2)
          tokensb = self.truncate_one(tokensb,  max_len//2)
        else:
          tokensb = self.truncate_one(tokensb,  max_len - len(tokensa))
    assert len(tokensa)+len(tokensb)<=max_len
    return tokensa,tokensb

  def plain(self,mat):   #  (self.masked, self.tokens)
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



