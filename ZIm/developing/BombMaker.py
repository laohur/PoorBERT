import math
from turtle import position
import numpy as np
from logzero import logger


def gen_glide_probs(lower: int, upper: int, peak: int, factor: float):
    lens = list(range(lower, upper + 1))
    # weights = [(upper-peak) - abs(x-peak) for x in ]
    weights = [factor**abs(x-peak) for x in lens]
    total = sum(weights)
    probs = [x/total for x in weights]
    avg_len = sum(lens[i]*x for i, x in enumerate(probs))
    return lens, probs, avg_len


def gen_mask(sent_len: int, mask_ratio: float, lens: list[int], probs: list[int], avg_len: float):
    # n_mask = math.ceil(sent_len*mask_ratio)
    n_bomb = math.ceil(sent_len*mask_ratio/avg_len)
    centers = np.random.choice(range(0, sent_len), n_bomb)
    marks = np.zeros(sent_len)
    bombs = [(x, np.random.choice(lens, p=probs)) for x in centers]
    for center, diameter in bombs:
        diameter = min(diameter, round(sent_len*mask_ratio))
        start = center-diameter//2
        end = center+(diameter+1)//2
        start = max(0, start)
        end = min(end, sent_len)
        # print("paint",center, diameter, start, end)
        if end > start:
            marks[start:end] = 1
    return marks


def paint(sent_len: int, mask_pairs: list):
    marks = np.zeros(sent_len)
    np.random.shuffle(mask_pairs)
    covered = 0
    for (mask_len, cover) in mask_pairs:
        offset = 0
        if cover-mask_len > 0:
            offset = np.random.randint(0, cover-mask_len)
        start = covered+offset
        marks[start:start+mask_len] = 1
        covered += cover
    return marks


class Masker:
    def __init__(self, mask_ratio=0.1, lower=1, upper=10, peak=3, factor=1.5, sent_lens=[512], bank_size=1024) -> None:
        self. mask_ratio = mask_ratio
        self.lower = lower
        self.upper = upper
        self.peak = peak
        self.factor = factor
        self.sent_lens = sent_lens
        self.bank_size = bank_size

        lens, probs, avg_len = gen_glide_probs(lower, upper, peak, factor)
        expected_length = sum([x*probs[i] for i, x in enumerate(lens)])
        logger.info((lens, probs, expected_length))
        self.lens = lens
        self.probs = probs
        self.avg_len = avg_len
        store = {}
        for sent_len in sent_lens:
            marks = [gen_mask(sent_len, mask_ratio, lens, probs,
                              avg_len) for _ in range(bank_size)]
            store[sent_len] = np.array(marks)
        self.bank = store

    def get_mask(self, sent_len):
        if sent_len not in self.bank:
            marks = [gen_mask(sent_len, mask_ratio, lens, probs,
                              avg_len) for _ in range(self.bank_size)]
            self.bank[sent_len] = np.array(marks)
        idx = np.random.randint(self.bank_size)
        if np.random.random() > 2/self.bank_size:
            mask_mark = self.bank[sent_len][idx]
        else:
            mask_mark = gen_mask(sent_len, self.mask_ratio,
                                 self.lens, self.probs, self.avg_len)
            self.bank[sent_len][idx] = mask_mark
        return mask_mark

    def mask(self, x):
        sent_len = x.shape[1]
        mask_mark = self.get_mask(sent_len)
        x = x*mask_mark
        return x


if __name__ == "__main__":
    lens, probs, avg_len = gen_glide_probs(1, 10, 3, 0.66)
    mask_ratio = 0.15
    for sent_len in range(0, 512):
        mask_mark = gen_mask(sent_len, mask_ratio, lens, probs, avg_len)
        print(sent_len, sum(mask_mark)/(sent_len+1e-6))
    import time
    t0 = time.time()
    sent_len = 512
    masker = Masker()
    for i in range(10000):
        # mark = gen_mask(sent_len, mask_ratio, lens, probs, avg_len)
        mark = masker.get_mask(sent_len)
    t1 = time.time()  # 10000  4.131736755371094
    logger.info(t1-t0)
