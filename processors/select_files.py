import glob,os,subprocess,numpy,random
import numpy as np

files=[]
files+=glob.glob(r"/media/u/t1/data/tokened/*/*.txt")
# files+=glob.glob(r"/media/u/t1/data/self/*/*.txt")
files+=glob.glob(r"/media/u/t1/data/qa/*/*.txt")
random.shuffle(files)
selected=np.random.choice(files,100,replace=False)
print(  f" \n\n ==== found {len(files)} files ==== \n")  #13000
# files.sort()
for src in selected:
    tgt=src.replace("data","selected")
    tgt_dir=os.path.dirname(tgt) 
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    cmd=["bash","cp" ,src,tgt ]
    cmd=f" cp {src}  {tgt} "
    print(f" cmd {cmd}")
    re= subprocess.call(cmd,shell=True)
    print(f" cmd {cmd} re {re} ")

print(  f" \n\n ==== selected {len(selected)} files ==== \n")  #13000
