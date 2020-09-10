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
logger = logging.getLogger(__name__)
# FORMAT = '%(pathname)s %(filename)s  %(funcName)s %(lineno)d %(asctime)-15s  %(message)s'
FORMAT = ' %(filename)s %(lineno)d %(funcName)s %(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)

def load_vocab(vocab_path="configs/vocab_shang.txt"):
  with open(vocab_path) as f:
    doc = f.readlines()
  vocab = [x.rstrip('\n') for x in doc if len(x) > 0]
  specials = [Constants.TOKEN_PAD, Constants.TOKEN_UNKOWN, Constants.TOKEN_BOS, Constants.TOKEN_EOS, Constants.TOKEN_BOC, Constants.TOKEN_EOC, Constants.TOKEN_MASK,Constants.TOKEN_FAKE,Constants.TOKEN_HARMONY, Constants.TOKEN_SELF, Constants.TOKEN_QUESTION, Constants.TOKEN_ANSWER, Constants.TOKEN_CONTEXT, Constants.TOKEN_CLS, Constants.TOKEN_QA, Constants.TOKEN_TRANSLATE]
  ascii = ['\t'] + [chr(x) for x in range(ord(' '), ord('~') + 1)]
  tokens = specials +ascii + vocab
  vocab = collections.OrderedDict()
  for x in tokens:
    if x and x not in vocab:
      vocab[x] = len(vocab.keys())
  logger.info(f" {vocab_path} 原始词汇表 {len(tokens)} --> 加载词汇表{len(vocab.keys())} ")  # configs/vocab_shang.txt 原始词汇表 6717 --> 加载词汇表6717
  return vocab

def load_spliter(doc):
  spliter = {}
  jianjia = '⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻'
  # with open(split_path) as f:
  #   doc = f.readlines()
  for line in doc:
    words = line.strip().split("\t")
    if len(words) == 2:
      c, subs = words
      grams = []
      for s in subs:
        if s not in jianjia:
          grams.append(s)
      # spliter[c]=''.join( grams)
      if len(grams) > 4:
        grams = grams[:2] + grams[-2:]
      spliter[c] = grams
  return spliter

def shake(line, vocab,noise=0.5):
  # insert del replace swap
  rand=random.random()
  if rand>noise or not line :
    return line,False
  rand=rand/noise
  seq=list(line)
  length = len(line)
  random_words = random.sample(vocab, 1)
  place = np.random.randint(0, length,2)
  # in_place = np.random.randint(0, length, 4)
  if rand<0.25:## insert
    word = random_words[0]
    seq.insert(place[0],word)
  elif rand<0.5: # del
    pos = min(place[0],length-1)
    del seq[pos]
  elif rand<0.75:# replace
    pos = min(place[0],length-1)
    word = random_words[0]
    seq[pos]=word
  else: # swap
    i,j=place
    if i>j:
      i,j=j,i
    j = min(j,length-1)
    j = max(j,1)
    i=min(i,j-1)
    seq[i],seq[j]=seq[j],seq[i]
  return ''.join(seq),True

class ShangTokenizer(object):
  def __init__(self,vocab_path="configs/vocab.txt", bujian_path="configs/bujian.txt",use_bujian=True):
    self.vocab=load_vocab(vocab_path)
    self.idx2word={}
    for word,idx in self.vocab.items():
      self.idx2word[idx]=word
    with open(bujian_path) as f:
      self.bujian=f.readlines()
    self.use_bujian=use_bujian
    self.spliter=load_spliter(self.bujian)
    self.tokens=list(self.vocab.keys())[self.vocab['\t']:]

  def char2bujian(self,c):  # char->list
    if c in self.vocab:
      return [c]
    if not self.use_bujian:
      return [Constants.TOKEN_UNKOWN]
    if c in self.spliter:
      return self.spliter[c]
    cp=ord(c)
    bytes=[ cp>>16,cp>>8,cp ]
    lines=[ x&255 for x in bytes  ]
    bujians=[ 'byte'+str(x) for x in lines ]
    return bujians

  def shake(self,tokens, noise=0.0):
    # insert del replace swap
    rand = random.random()
    if rand > noise or not tokens:
      return tokens, False
    rand =random.random()
    # seq = list(tokens)
    seq = tokens
    length = len(tokens)
    # candidates=list(self.vocab.keys())[self.vocab['\t']:]
    random_words = random.sample(self.tokens, 1)
    place = np.random.randint(0, length, 2)
    # in_place = np.random.randint(0, length, 4)
    if rand < 0.25:  ## insert
      word = random_words[0]
      seq.insert(place[0], word)
    elif rand < 0.5:  # replace
      pos = min(place[0], length - 1)
      word = random_words[0]
      seq[pos] = word
    elif rand<0.75:  # swap
      i, j = place
      if i > j:
        i, j = j, i
      j = min(j, length - 1)
      # j = max(j, 1)
      # i = min(i, j - 1)
      seq[i], seq[j] = seq[j], seq[i]
    else:  # del
      pos = min(place[0], length - 1)
      del seq[pos]

    return seq, True

  def segment(self,line):
    pass

  def combine(self,toens):
    pass

  def token2idx(self,token):
    if not isinstance(token,str):
      c=0
    if token not in self.vocab:
      token=Constants.TOKEN_UNKOWN
    return self.vocab.get(token,1 )

  def idx2token(self,idx):
    if idx not in self.idx2word:
      idx=self.vocab[Constants.TOKEN_UNKOWN]
    return self.idx2word[idx]

  def convert_tokens_to_ids(self,tokens):
    return [self.token2idx(x) for x in tokens  ]

  def tokenize(self,line,noise=0): # seg tokens
    chars=self.tokenize2chars(line)
    if noise>0:
      chars,_=self.shake(chars,noise)
    bujians=[]
    for c in chars:
      bujians+=self.char2bujian(c)
    return bujians

  def tokenize2chars(self,line):
    return list(line)

  def chars2bujians(self,chars):
      return [ self.char2bujian(c) for c in chars    ]

  def save_vocabulary(self, vocab_path,bujian_path=''):
      """Save the tokenizer vocabulary to a directory or file."""
      index = 0
      if os.path.isdir(vocab_path):
        vocab_file = os.path.join(vocab_path, "vocab.txt")
      else:
          vocab_file = vocab_path
      with open(vocab_file, "w", encoding="utf-8") as writer:
          # for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):

          for token_index,token  in self.idx2word.items():
              if index != token_index:
                  logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                 " Please check that the vocabulary is not corrupted!".format(vocab_file))
                  index = token_index
              writer.write(token + u'\n')
              index += 1
      if not bujian_path:
        bujian_path=vocab_path
      if os.path.isdir(bujian_path):
        bujian_file = os.path.join(bujian_path, "bujian.txt")
      else:
         bujian_file = bujian_path
      with open(bujian_file,"w") as w:
        w.writelines('\n'.join(self.bujian)+'\n')
      logger.info(f"vocabulary saved -> {vocab_file} {bujian_file}" )
      return (vocab_file,bujian_file)


# HeadTail = {
#   'S': 0,
#   'B': 1,
#   'E': 2,
#   'M': 3
# }
# stage: warnup pretrain train eval

class SentenceWorded(object):
  def __init__(self, words, tokenizer, noise=0):
    self.tokenizer = tokenizer
    words=[ x.strip() for x in words.split(" ") if x  ]
    self.word_label = []
    self.char_label = []
    self.token_label=[]
    self.tokens = []
    for word in words:
      word_maksed = False
      if random.random() < len(word)/10:
        word_maksed = True
      chars = self.tokenizer.tokenize2chars(word)

      for i,char in enumerate(chars):
        char_maksed = False
        word_label = 0 if i==0 else 1
        bujians = self.tokenizer.char2bujian(char)
        self.token_label+=bujians
        for j ,bujian in enumerate(bujians):
          char_label = 0 if j==0 else 1
          if word_maksed or char_maksed:
            bujian = Constants.TOKEN_MASK
          # self.tokens.append(self.tokenizer.token2idx(bujian))
          self.tokens.append( bujian)
          self.char_label.append(char_label)
          self.word_label.append(word_label)
    self.length = len(self.tokens)

  def get_features(self):
    return (self.tokens, self.token_label,self.char_label, self.word_label)

class Sentence(object):
  def __init__(self, line, tokenizer, noise=0):
    self.tokenizer = tokenizer
    tokens =self.tokenizer.tokenize(line)
    self.tokens,self.modified=self.tokenizer.shake(tokens,noise=0)  # no arguement
    self.length = len(self.tokens)
    masked=self.tokens[:]
    probs=[3,5,6,7,8]
    i=0
    while i<len(masked):
      rand=random.randint(0,99)
      tomask=False
      for j,prob in enumerate(probs[:len(masked)-1-i]):
        if rand<prob:
          tomask=True
          for k in range(j+1):
            masked[i] = Constants.TOKEN_MASK
            i += 1
          break
      if not tomask:
        if rand<=9:
          masked[i] = random.sample(self.tokens, 1)[0]
        i += 1
    self.masked=masked

  def get_features(self):
    return (self.masked, self.tokens)

