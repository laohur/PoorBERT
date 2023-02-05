# PoorBert

 implements of "The Art of Pretraining PoorBERT"

## pretrain
  python3  pretraining.py 

## finetune (CLUE Benchmark)

  python3  finetune.py


## fork from

https://github.com/laohur/albert_pytorch

## dataset
### corpus
* "/media/u/t1/data/tokened/\*/\*.txt"
* "/media/u/t1/data/self/\*/\*.txt"
* "/media/u/t1/data/qa/\*/\*.txt"

### saved 
* "/media/u/t1/dataset/PoorBERT"

### clue
* "/media/u/t1/dataset/CLUEdatasets/" 

## mode
 [ PoorNone  PoorBujian PoorRelation PoorStair PoorBERT ]

 config in
* pretrain.py OUTPUTS_DIR -> PoorBERT
* tasks/utils.pt model_dir -> PoorBERT

  influes:
* models/tokenization_shang.py -> ShangTokenizer.use_bujian=False
* models/modeling_poor.py -> BertForPreTraining.forward() last layer

* models/PretrainTokenedDataset.py -> PretrainTokenedDataset.use_relations=False
* models/PretrainSelfDataset.py -> PretrainSelfDataset.use_relations=False
* models/PretrainQaDataset.py -> PretrainQaDataset.use_relations=False

## update
<!-- stair learning -->
smooth lr

## todo
DistributedDataParallel 