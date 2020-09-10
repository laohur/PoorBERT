""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import subprocess


def run_all():
    tasks=["afqmc",'c3','chid','cmnli','tnews','cmrc','csl','iflytek','wsc']
    for i,task in enumerate(tasks):
        # cmd=f"  CUDA_VISIBLE_DEVICES={'0'} python3 tasks/{task}.py  --model_name none"
        cmd=f"  CUDA_VISIBLE_DEVICES={'0'} python3 tasks/{task}.py  "
        subprocess.call(cmd,shell=True)
        print(f"\n task{i} {task}  finetuned {cmd} \n")
if __name__ == "__main__":
    run_all()

'''
 CUDA_VISIBLE_DEVICES="0"  python3  run_tasks.py | tee tasks.log

'''
