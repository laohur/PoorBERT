
import numpy as np
from logzero import logger



class Bomber:
    def __init__(self, mask_ratio=0.1, lower=1, upper=10, peak=3, factor=0.66,scale=0, sent_lens=[], bank_size=128) -> None:
        self. mask_ratio = mask_ratio
        self. scale = scale
        self.lower = lower
        self.upper = upper
        self.peak = peak
        self.factor = factor
        self.sent_lens = sent_lens
        self.bank_size = bank_size
        lens, probs, avg_len = self.gen_glide_probs(lower, upper, peak, factor)
        expected_length = sum([x*probs[i] for i, x in enumerate(lens)])
        logger.info((lens, probs, expected_length))
        self.lens = lens
        self.probs = probs
        self.avg_len = avg_len
        store = {   }
        for sent_len in sent_lens:
            marks = [self.gen_mask(sent_len, mask_ratio, lens, probs) for _ in range(bank_size)]
            store[sent_len] = np.array(marks)
        self.bank = store

    def gen_glide_probs(self,lower: int, upper: int, peak: int, factor: float):
        lens = list(range(lower, upper + 1))
        weights = [factor**abs(x-peak) for x in lens]
        total = sum(weights)
        probs = [x/total for x in weights]
        avg_len = sum(lens[i]*x for i, x in enumerate(probs))
        return lens, probs, avg_len


    def gen_mask_spans(self,sent_len: int, mask_ratio: float, lens: list[int], probs: list[int]):
        convered=0
        spans=[]
        while convered<sent_len:
            mask_len=np.random.choice(lens, p=probs)
            cover=round(mask_len/mask_ratio)
            if cover+convered>=sent_len:
                cover=sent_len-convered
                mask_len=round(cover*mask_ratio)
            convered+=cover        
            if mask_len>=0:
                spans.append((mask_len,cover))
        return spans


    def paint(self,sent_len: int, spans: list,scale:float=0):
        marks = np.ones(sent_len)
        np.random.shuffle(spans)
        r=scale*np.random.rand()
        covered = 0
        for (mask_len, cover) in spans:
            offset = np.random.randint(0, cover-mask_len)
            start = covered+offset
            marks[start:start+mask_len] = r
            covered += cover
        return marks

    def gen_mask(self,sent_len: int, mask_ratio: float, lens: list[int], probs: list[int],scale:float=0):
        mask_spans=self.gen_mask_spans(sent_len,mask_ratio,lens,probs)
        marks=self.paint(sent_len,mask_spans,scale)     
        return marks

    def get_mask(self, sent_len):
        if sent_len not in self.bank:
            marks = [self.gen_mask(sent_len, mask_ratio, lens, probs,)
                     for _ in range(self.bank_size)]
            self.bank[sent_len] = np.array(marks)
        idx = np.random.randint(self.bank_size)
        if np.random.random() > 1/self.bank_size:
            mask_mark = self.bank[sent_len][idx]
        else:
            mask_mark = self.gen_mask(
                sent_len, self.mask_ratio, self.lens, self.probs)
            self.bank[sent_len][idx] = mask_mark
        return mask_mark

    def mask(self, x,dim=-1):
        if self.mask_ratio<=0:
            return x
        if dim==None:
            dim=np.random.randint(len(x.shape))-1
        sent_len = x.shape[dim]
        n_mask=round(sent_len*self.mask_ratio)
        if n_mask<1:
            return x
        start=np.random.randint(0,sent_len-n_mask)
        x = x.select(dim)[start,start+n_mask]=0
        return x/(1-n_mask/sent_len)

    def dropout(self, x,dim=-1):
        if self.mask_ratio<=0:
            return x
        sent_len = x.shape[dim]
        n_mask=round(sent_len*self.mask_ratio)
        if n_mask<1:
            return x
        start=np.random.randint(0,sent_len-n_mask)
        x = x.select(dim)[start,start+n_mask]=0
        return x/(1-n_mask/sent_len)/(1-self.scale)

if __name__ == "__main__":
    masker = Bomber()
    lens, probs, avg_len = masker.gen_glide_probs(1, 10, 3, 0.66)
    mask_ratio = 0.1
    for sent_len in range(0, 512):
        mask_mark = masker.gen_mask(sent_len, mask_ratio, lens, probs)
        print(sent_len, sum(mask_mark)/(sent_len+1e-6))
    import time
    t0 = time.time()
    sent_len = 512
    masker = Bomber()
    for i in range(10000):
        mark = masker.gen_mask(sent_len, mask_ratio, lens, probs)
    t1 = time.time()  # 10000  4.131736755371094
    logger.info(t1-t0)

"""
random crop
random cutoff
"""