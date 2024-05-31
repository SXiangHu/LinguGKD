# !/bin/bash
now=$(date +"%Y-%m-%d-%H-%M-%S")
dataset_name="Cora"
llm_model_prefix="meta-llama" 
llm_model="Meta-Llama-3-8B" 

early_stopping_patience=1
epochs=1
has_id=true
is_old=true
is_dpo=false
remark=""
device=0

main_script="llm_main.py"
log_path="./log/llm/$dataset_name/$llm_model/$now"

echo $main_script
mkdir -p "$log_path"

nohup python -u ./$main_script \
  --is_complete_only \
  --device=$device \
  --seed=42 \
  --grad_steps=2 \
  --overwrite=True \
  --epochs=$epochs \
  --train_batch_size=2 \
  --learning_rate=0.0002 \
  --max_seq_length=520 \
  --early_stopping_patience=$early_stopping_patience \
  --llm_model="$llm_model_prefix/$llm_model" \
  --dataset_name=$dataset_name \
  --remark=$remark \
  --has_id=$has_id \
  --is_old=$is_old \
  --now=$now > "$log_path/bash_log.log" 2>&1 &