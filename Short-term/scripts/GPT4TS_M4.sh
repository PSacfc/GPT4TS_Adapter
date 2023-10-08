export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs/ShortForecasting/" ]; then
mkdir -p ./logs/ShortForecasting/
fi



model_name=GPT4TS

for lr in 0.0005 0.0001 0.001
do
for adapter_dim in 32 
do
for scale in 10000 1000
do
python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ../dataset/m4 \
    --seasonal_patterns 'Yearly' \
    --model_id m4_layer_2Yearly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 128 \
    --d_model 768 \
    --d_ff 32 \
    --patch_len 3 \
    --patience 100 \
    --train_epochs 200 \
    --stride 1 \
    --n_heads 16 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --lradj CARD \
    --learning_rate $lr \
    --loss 'SMAPE' --warmup_epochs 40 \
    --C_type 1 --T_type 1 --adapter_dim $adapter_dim --adapter_dropout 0.1 \
    --gpt_layers 6 --adapter_layer 6 \
    --scale $scale \
    --spect_adapter_layer 6 > logs_final/$model_name'_'m4_Yearly_$lr'_'$adapter_dim'_'$scale.log
done
done
done


for lr in 0.0001 0.001
do
for adapter_dim in 128 32 
do
for scale in 10000 1000
do
python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ../dataset/m4 \
    --seasonal_patterns 'Yearly' \
    --model_id m4_layer_2Yearly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 128 \
    --d_model 768 \
    --d_ff 32 \
    --patch_len 3 \
    --patience 100 \
    --train_epochs 200 \
    --stride 1 \
    --n_heads 16 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --lradj CARD \
    --learning_rate $lr \
    --loss 'SMAPE' --warmup_epochs 40 \
    --C_type 1 --T_type 1 --adapter_dim $adapter_dim --adapter_dropout 0.1 \
    --gpt_layers 6 --adapter_layer 6 \
    --scale $scale \
    --spect_adapter_layer 6 > logs_final/$model_name'_'m4_Yearly_$lr'_'$adapter_dim'_'$scale.log

python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ../dataset/m4 \
    --seasonal_patterns 'Quarterly' \
    --model_id m4_layer_2_Quarterly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 128 \
    --d_model 768 \
    --d_ff 32 \
    --patch_len 4 \
    --patience 100 \
    --train_epochs 200 \
    --stride 1 \
    --n_heads 16 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --lradj CARD \
    --learning_rate $lr \
    --loss 'SMAPE' --warmup_epochs 40 \
    --C_type 1 --T_type 1 --adapter_dim $adapter_dim --adapter_dropout 0.1 \
    --gpt_layers 6 --adapter_layer 6 \
    --scale $scale \
    --spect_adapter_layer 6 > logs_final/$model_name'_'m4_Quarterly_$lr'_'$adapter_dim'_'$scale.log
done
done
done
