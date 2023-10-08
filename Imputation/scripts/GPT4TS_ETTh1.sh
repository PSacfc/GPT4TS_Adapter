export CUDA_VISIBLE_DEVICES=1

model_name=GPT4TS

for layer in 3
do
for patch in 1
do
for dff in 32
do
for lr in 0.001
do

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 768 \
  --patch_size $patch \
  --stride 1 \
  --des 'Exp' \
  --gpt_layer $layer \
  --adapter_dim $dff \
  --learning_rate $lr \
  --warmup_epochs 10 --train_epochs 40 \
  --T_type 1 --C_type 0 --scale 100 > logs_imputation/ETTh1_0.125_GPT4TS_$layer'_'$patch'_'$dff'_'$lr.logs

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 768 \
  --patch_size $patch \
  --stride $patch \
  --des 'Exp' \
  --itr 1 \
  --gpt_layer $layer \
  --adapter_dim $dff \
  --learning_rate $lr \
  --warmup_epochs 10 --train_epochs 40 \
  --T_type 1 --C_type 0 --scale 100 > logs_imputation/ETTh1_0.25_GPT4TS_$layer'_'$patch'_'$dff'_'$lr.logs

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 768 \
  --patch_size $patch \
  --stride $patch \
  --des 'Exp' \
  --itr 1 \
  --gpt_layer $layer \
  --adapter_dim $dff \
  --learning_rate $lr \
  --warmup_epochs 10 --train_epochs 40 \
  --T_type 1 --C_type 0 --scale 100 > logs_imputation/ETTh1_0.375_GPT4TS_$layer'_'$patch'_'$dff'_'$lr.logs

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 768 \
  --patch_size 1 \
  --stride 1 \
  --des 'Exp' \
  --itr 1 \
  --gpt_layer $layer \
  --adapter_dim $dff \
  --learning_rate $lr \
  --warmup_epochs 10 --train_epochs 40 \
  --T_type 1 --C_type 0 --scale 100 > logs_imputation/ETTh1_0.5_GPT4TS_$layer'_'$patch'_'$dff'_'$lr.logs

done
done
done
done