#CURRENT_DIR='pwd'
CURRENT_DIR='..'
#export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_large_v2
#export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_base_zh
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_base

export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="sst-2"
python run_classifier.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --config_path='configs/albert_config_base.json' \
  --config_path=$BERT_BASE_DIR/config.json \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=8 \
  --spm_model_file=${BERT_BASE_DIR}/pytorch_model.bin \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=4210 \
  --save_steps=4210 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

