""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import subprocess


def run_all():
    tasks=["afqmc",'tnews','iflytek','cmnli','wsc','csl',"cmrc",'cmrc2018', 'c3','c30','chid','chid0']
    for i,task in enumerate(tasks):
        # cmd=f"  CUDA_VISIBLE_DEVICES={'0'} python3 tasks/{task}.py  --model_name none"
        cmd=f"  python3 tasks/{task}.py  "
        subprocess.call(cmd,shell=True)
        print(f"\n task{i} {task}  finetuned {cmd} \n")
if __name__ == "__main__":
    run_all()

'''
python3  finetune.py 

'''