export CUDA_VISIBLE_DEVICES=1

for k in 3
do
for lr in 0.0001
do
for patch in 3 5 7
do
for anormly_ratio in 0.5
do
python main.py \
    --anormly_ratio $anormly_ratio --num_epochs 1 --batch_size 16 --mode train \
    --dataset PSM --data_path dataset/PSM --input_c 25 --layer 3 \
    --anomaly_layer 3 --k $k --lr $lr --patch_len $patch > logs/PSM_new_train_$k'_'$lr'_'$patch'_'$anormly_ratio.logs
python main.py \
    --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test \
    --dataset SMD --data_path dataset/SMD --input_c 38 --pretrained_model 20 \
    --k $k --lr $lr # > logs/SMD_test_$k'_'$lr.logs
done
done
done
done
