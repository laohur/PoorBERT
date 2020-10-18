import glob

import numpy as np

probs = [0.2, 0.2, 0.2, 0.2, 0.2]
lens = [1024, 512, 256, 128, 64]
sizes = [7, 19, 44, 97, 208]
steps = 0
for i, p in enumerate(probs):
    steps += probs[i] / sizes[i]
total_train_examples=33e7
num_train_optimization_steps = int(total_train_examples * steps)
batch_size=total_train_examples/num_train_optimization_steps
files = []
files += glob.glob(r"/media/u/t1/data/tokened/*/*.txt")
files+=glob.glob(r"/media/u/t1/data/self/*/*.txt")
files += glob.glob(r"/media/u/t1/data/qa/*/*.txt")
# files=np.random.choice(files,100,replace=False)
# files.sort()
steps=0
for fid, file_path in enumerate(files):
    idx = np.random.choice(a=len(probs), size=1, replace=False, p=probs)[0]
    examples=total_train_examples/len(files)
    max_len = lens[idx]
    train_batch_size = sizes[idx]
    steps+= examples/train_batch_size

print( f"num_train_optimization_steps:{num_train_optimization_steps}" )
print( f"batch_size:{batch_size}" )
print( f"steps:{steps}" )
print( f"avg batch size :{total_train_examples/steps}" )

# 15399975