#!/bin/bash
#source ~/easy-ocr/bin/activate
# case old
python train.py --train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--data_filtering_off --workers 0 --imgH 64 --imgW 200 --batch_size 32
# case 1 => 이옵션의 경우 지속적으로 batch_max_length 오류가 나서 case 2번으로 진행
CUDA_VISIBLE_DEVICES=0 python train.py --train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
    --valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation  \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --data_filtering_off --workers 0 --imgH 64 --imgW 200 \
    --batch_size 150 --batch_max_length 67 & > /dev/null

while true; do sleep 120; printf ".";done

# case 2 => 성공 케이스
CUDA_VISIBLE_DEVICES=0 python train.py --train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 200 --data_filtering_off --workers 0 \
--num_iter 100000 --valInterval 100 & > /dev/null
# --saved_model /home/ubuntu/deep-text-recognition-benchmark/models/TPS-ResNet-BiLSTM-CTC.pth \


while true; do sleep 120; printf ".";done

# old
python train.py \
--train_data data_lmdb_release/training \
--valid_data data_lmdb_release/validation \
--select_data MJ-ST \
--batch_ratio 0.5-0.5 \
--Transformation TPS \
--FeatureExtraction ResNet \
--SequenceModeling BiLSTM \
--Prediction Attn \
--saved_model models/TPS-ResNet-BiLSTM-Attn.pth \
--workers 0 \
--FT \
--num_iter 1000 \
--character "0123456789abcdefghijklmnopqrstuvwxyz"

## https://velog.io/@apphia39/python
# made1
CUDA_VISIBLE_DEVICES=0 python3 ./deep-text-recognition-benchmark/train.py \
    --train_data ./deep-text-recognition-benchmark/made1_data_lmdb/train \
    --valid_data ./deep-text-recognition-benchmark/made1_data_lmdb/validation \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --batch_size 512 --batch_max_length 200 --data_filtering_off --workers 0 \
    --saved_model ./pretrained_models/kocrnn.pth --num_iter 100000 --valInterval 100