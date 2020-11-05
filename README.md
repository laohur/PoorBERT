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

models/tokenization_shang.py -> ShangTokenizer.use_bujian=False
models/modeling_poor.py -> BertForPreTraining.forward() last layer

models/PretrainTokenedDataset.py -> PretrainTokenedDataset.use_relations=False
models/PretrainSelfDataset.py -> PretrainSelfDataset.use_relations=False
models/PretrainQaDataset.py -> PretrainQaDataset.use_relations=False

### none
pretrain.py OUTPUTS_DIR -> PoorNone
tasks/utils.pt model_dir -> PoorNone

### bujian
pretrain.py OUTPUTS_DIR -> PoorBujian
tasks/utils.pt model_dir -> PoorBujian

### relation
pretrain.py OUTPUTS_DIR -> PoorRelation
tasks/utils.pt model_dir -> PoorRelation

### gradation
pretrain.py OUTPUTS_DIR -> PoorGradation
tasks/utils.pt model_dir -> PoorGradation