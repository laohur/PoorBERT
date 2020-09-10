CURRENT_DIR=`pwd`
#CURRENT_DIR=`..`
#CURRENT_DIR=`.`
#export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_large_zh
#export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_base
export BERT_BASE_DIR=$CURRENT_DIR/outputs/lm-checkpoint
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="lcqmc"

python  run_classifier.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --config_path=$BERT_BASE_DIR/config.json \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=14923 \
  --save_steps=14923 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir


#--model_type=albert
#--model_name_or_path=outputs/lm-checkpoint
#--task_name=lcqmc
#--do_train
#--do_eval
#--do_lower_case
#--data_dir=dataset/lcqmc
#--vocab_file=outputs/lm-checkpoint/vocab.txt
#--config_path=outputs/lm-checkpoint/config.json
#--max_seq_length=128
#--per_gpu_train_batch_size=16
#--per_gpu_eval_batch_size=16
#--learning_rate=1e-5
#--num_train_epochs=3.0
#--logging_steps=14923
#--save_steps=14923
#--output_dir=outputs/lcqmc_output/
#--overwrite_output_dir