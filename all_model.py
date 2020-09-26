""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import subprocess


def run_all():
    for model in ["none","gradation","bujian","relation","all"]:
        cmd=f" cd ~/work/Poor_{model} "
        subprocess.call(cmd,shell=True)
        print(f" cmd {cmd} ")
        cmd=f" python3 run_tasks.py "
        print(f" cmd {cmd} ")
        subprocess.call(cmd,shell=True)
        print(f" model {model} finetuned")

if __name__ == "__main__":
    run_all()

'''
 CUDA_VISIBLE_DEVICES="0"  python3  all_model.py 


'''
