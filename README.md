# PoorBert

 implements of "The Art of Pretrain PoorBERT"

## pretrain
 CUDA_VISIBLE_DEVICES="0"  python3  run_pretraining.py | tee pretrain.log

## finetune (CLUE Benchmark)

 CUDA_VISIBLE_DEVICES="0"  python3  run_tasks.py | tee tasks.log

## fork from

https://github.com/laohur/albert_pytorch


