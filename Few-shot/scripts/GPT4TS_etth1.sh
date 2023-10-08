export CUDA_VISIBLE_DEVICES=1

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

for warmup in 0 10
do
for seq_len in 512
do
for pred_len in 720
do
for scale in 100
do
for lr in 0.001
do
for dff in 512
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
    --d_ff $dff \
    --head_dropout 0 \
    --adapter_dropout 0.1 \
    --patch_len 16 \
    --stride 16 \
    --des 'Exp' \
    --train_epochs 40 \
    --patience 3 \
    --itr 1 --batch_size 8 --learning_rate $lr \
    --warmup_epochs $warmup \
    --scale $scale \
    --gpt_layers 6 \
    --spect_adapter_layer 6 \
    --adapter_layer 6 \
    --T_type 1 \
    --C_type 1 \
    --adapter_dim 16 \
    --percent 10 > logs_few_shot/percent_10/etth1_new_$seq_len'_'$pred_len'_'$lr'_'$scale'_'$warmup'_'$lradj'_dff'$dff.log
done
done
done
done
done
done
done

