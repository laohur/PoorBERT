
from ltp import LTP
from torch.utils.data import RandomSampler, DataLoader

from model.tokenization_shang import ShangTokenizer, SentenceWorded, PretrainDataset

# ltp = LTP(path='/media/u/t1/vender/ltp')  # 默认加载 Small 模型
# ltp = LTP(path="base")  # 默认加载 Small 模型
# ltp = LTP(path="tiny")  # 默认加载 Small 模型
# ltp = LTP()  # 默认加载 Small 模型
# ltp = LTP(path = "base|small|tiny")
# sent_list = ltp.sent_split(inputs, flag="all", limit=510)
line='LTP（Language Technology Platform） 提供了一系列中文自然语言处理工具，用户可以使用这些工具对于中文文本进行分词、词性标注、句法分析等等工作。'
# sent_list = ltp.sent_split([line], flag="all", limit=510)
# sent_list = ltp.sent_split(line,limit=510)
# segment, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
# segment, hidden = ltp.seg([line])
# return segment[0]

tokenizer=ShangTokenizer(vocab_path="configs/vocab_shang.txt",split_path="configs/spliter_cn.txt")

# sent=Sentence(line=line,tokenizer=tokenizer,ltp=ltp)
# tokens, char_label, word_label=sent.get_features()
# input_file="/media/u/t1/data/self/comment2019zh_corpus/comment_yf_dianping9_.txt"
input_file="/media/u/t1/data/self/clue_pretrain/clue_pretrain_3259.txt"
epoch_dataset = PretrainDataset(input_file=input_file, tokenizer=tokenizer,max_tokens=512)
# if args.local_rank == -1:
train_sampler = RandomSampler(epoch_dataset)
# else:
#     train_sampler = DistributedSampler(epoch_dataset)
# train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=16,num_workers=1)
train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=16)
# model.train()
nb_tr_examples, nb_tr_steps = 0, 0
for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to("cuda") for t in batch)
    ( tokens_tensor,input_mask_tensor,token_label_tensor,char_label_tensor,word_label_tensor, relarion_label )= batch
    print(f" {step} success!")
