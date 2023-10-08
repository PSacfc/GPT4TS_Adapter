export CUDA_VISIBLE_DEVICES=0

model=GPT4TS

for lr in 0.001
do
for d_ff in 768
do
for scale in 100
do

seq_len=16
pred_len=8
test_seq_len=16
test_pred_len=8
source_data=m3

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/m4/ \
    --data_path m3_quarterly_dataset.tsf \
    --test_data_path m4_quarterly_dataset.tsf \
    --model_id m3Quarterly_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 64 \
    --test_batch_size 128 \
    --decay_fac 1 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --dropout 0 \
    --patch_size 2 \
    --stride 1 \
    --proj_hid 64 \
    --print_int 1000 \
    --is_gpt 1 \
    --learning_rate $lr \
    --train_epochs 20 --warmup_epochs 5 \
    --T_type 1 --C_type 0 --scale $scale --d_ff $d_ff \
    > logs/new_m3_m4_Quarterly_'scale'$scale'_d_ff'$d_ff'_lr'$lr.logs


seq_len=48
pred_len=24
test_seq_len=48
test_pred_len=24

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/m4/ \
    --data_path m3_monthly_dataset.tsf \
    --test_data_path m4_monthly_dataset.tsf \
    --model_id m3Monthly_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $seq_len \
    --test_pred_len $test_pred_len \
    --label_len 10 \
    --batch_size 32 \
    --test_batch_size 128 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 8 \
    --loss_func smape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --dropout 0 \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --is_gpt 1 \
    --learning_rate $lr \
    --train_epochs 20 --warmup_epochs 5 \
    --adapter_layer 6 --spect_adapter_layer 6 \
    --T_type 1 --C_type 0 --scale $scale --d_ff $d_ff \
    > logs/new_m3_m4_Monthly_'scale'$scale'_d_ff'$d_ff'_lr'$lr.logs



seq_len=9
pred_len=6
test_seq_len=9
test_pred_len=6
source_data=m3

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/m4/ \
    --data_path m3_yearly_dataset.tsf \
    --test_data_path m4_yearly_dataset.tsf \
    --model_id m3Yearly_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 32 \
    --test_batch_size 128 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --dropout 0 \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --is_gpt 1 \
    --learning_rate $lr \
    --train_epochs 20 --warmup_epochs 5 \
    --T_type 1 --C_type 0 --scale $scale --d_ff $d_ff \
    > logs/new_m3_m4_Yearly_'scale'$scale'_d_ff'$d_ff'_lr'$lr.logs

seq_len=65
pred_len=13
test_seq_len=65
test_pred_len=13
source_data=m3

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/m4/ \
    --data_path m3_monthly_dataset.tsf \
    --test_data_path m4_weekly_dataset.tsf \
    --model_id m3Monthly_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 10 \
    --batch_size 128 \
    --test_batch_size 128 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --dropout 0 \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --is_gpt 1 \
    --learning_rate $lr \
    --train_epochs 20 --warmup_epochs 5 \
    --T_type 1 --C_type 0 --scale $scale --d_ff $d_ff \
    > logs/new_m3_m4_Weekly_'scale'$scale'_d_ff'$d_ff'_lr'$lr.logs


seq_len=9
pred_len=14
test_seq_len=9
test_pred_len=14

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/m4/ \
    --data_path m3_monthly_dataset.tsf \
    --test_data_path m4_daily_dataset.tsf \
    --model_id m3Monthly_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 10 \
    --batch_size 64 \
    --test_batch_size 128 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --dropout 0 \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --is_gpt 1 \
    --learning_rate $lr \
    --train_epochs 20 --warmup_epochs 5 \
    --T_type 1 --C_type 0 --scale $scale --d_ff $d_ff \
    > logs/new_m3_m4_Daily_'scale'$scale'_d_ff'$d_ff'_lr'$lr.logs


seq_len=2
pred_len=48
test_seq_len=2
test_pred_len=48

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/m4/ \
    --data_path m3_other_dataset.tsf \
    --test_data_path m4_hourly_dataset.tsf \
    --model_id m3Monthly_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 10 \
    --batch_size 16 \
    --test_batch_size 128 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --train_all 1 \
    --is_gpt 1 \
    --learning_rate $lr \
    --train_epochs 20 --warmup_epochs 5 \
    --T_type 1 --C_type 0 --scale $scale --d_ff $d_ff \
    > logs/new_m3_m4_Hourly_'scale'$scale'_d_ff'$d_ff'_lr'$lr.logs

done
done
done