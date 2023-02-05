import math
import numpy as np
from logzero import logger


def gen_glide_probs(lower: int, upper: int, peak: int, factor: float):
    lens = list(range(lower, upper + 1))
    weights = [(upper-peak) - abs(x-peak) for x in lens]
    weights = [factor**x for x in weights]
    total = sum(weights)
    probs = [x/total for x in weights]
    return lens, probs


def gen_mask(sent_len: int, mask_ratio: float, lens: list[int], probs: list[int], peak: int):
    n_mask = math.ceil(sent_len*mask_ratio)

    to_cover = sent_len
    masked = 0
    mask_pairs = []
    while masked < n_mask:
        if 1/mask_ratio >= to_cover:
            if 1/to_cover * np.random.random() < mask_ratio*2:
                mask_pairs.append((1, to_cover))
            break
        if peak/mask_ratio >= to_cover:
            mask_len = round(to_cover*mask_ratio)
            if mask_len >= 1:
                mask_pairs.append((mask_len, to_cover))
            break
        mask_len = np.random.choice(lens, p=probs)
        cover = round(mask_len/mask_ratio)
        mask_pairs.append((mask_len, cover))
        to_cover -= cover
        masked += mask_len
    marks = paint(sent_len, mask_pairs)
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

        lens, probs = gen_glide_probs(lower, upper, peak, factor)
        expected_length = sum([x*probs[i] for i, x in enumerate(lens)])
        logger.info((lens, probs, expected_length))
        self.lens = lens
        self.probs = probs
        store = {}
        for sent_len in sent_lens:
            marks = [gen_mask(sent_len, mask_ratio, lens, probs, peak)
                     for _ in range(bank_size)]
            store[sent_len] = np.array(marks)
        self.bank = store

    def get_mask(self, sent_len):
        idx = np.random.randint(self.bank_size)
        if np.random.random() > 2/self.bank_size:
            mask_mark = self.bank[sent_len][idx]
        else:
            mask_mark = gen_mask(sent_len, self.mask_ratio,
                                 self.lens, self.probs, self.peak)
            self.bank[sent_len][idx] = mask_mark
        return mask_mark

    def mask(self, x):
        sent_len = x.shape[1]
        mask_mark = self.get_mask(sent_len)
        x = x*mask_mark
        return x


if __name__ == "__main__":
    lens, probs = gen_glide_probs(1, 10, 3, 1.5)
    mask_ratio = 0.1
    for sent_len in range(0, 514):
        mask_mark = gen_mask(sent_len, mask_ratio, lens, probs, 3)
        # print(sent_len, sum(mask_mark)/(sent_len+1e-6))
    import time
    t0 = time.time()
    sent_len = 512
    masker = Masker()
    for i in range(1000000):
        # mark = gen_mask(sent_len, mask_ratio, lens, probs, 3)
        mark = masker.get_mask(sent_len)
    t1 = time.time()  # 10000  4.131736755371094
    logger.info(t1-t0)
