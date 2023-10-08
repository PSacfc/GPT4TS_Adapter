# export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=GPT4TS



root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1


random_seed=2021

for dim in 32
do
for seq_len in 720
do
for lr in 0.0002
do
for pred_len in 96
do
for scale in 10000
do
    python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 768 \
    --d_ff 32 \
    --head_dropout 0 \
    --adapter_dropout 0.1 \
    --patch_len 16 \
    --stride 8 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 5 \
    --itr 1 --batch_size 32 --learning_rate $lr \
    --warmup_epochs 10 \
    --scale $scale \
    --gpt_layers 6 \
    --spect_adapter_layer 6 \
    --adapter_layer 6 \
    --T_type 1 \
    --C_type 1 \
    --adapter_dim $dim \
    --use_multi_gpu
    #  > logs_final_new/etth1_$pred_len'_'$lr'_'$scale'_'$dim.log
done
done
done
done
done

