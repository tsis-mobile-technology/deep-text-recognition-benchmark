#!/bin/bash
#source ~/easy-ocr/bin/activate
# TPS ResNet BiLSTM CTC
##CUDA_VISIBLE_DEVICES=0,1 python train.py --train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
##--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation  \
##--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
##--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
##--saved_model ./models/best_accuracy.pth --num_iter 5000 --valInterval 100
##--num_iter 50000 --valInterval 100 & > /dev/null
#--sensitive \

# TPS ResNet BiLSTM Attn(#1)
# Text_in_the_wild(Goods)
CUDA_VISIBLE_DEVICES=0,1 python train.py --train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--num_iter 100000 --valInterval 100 & > /dev/null
##--saved_model ./models/best_accuracy.pth --num_iter 1000 --valInterval 100

# TPS ResNet BiLSTM Attn(#2)
# Text_in_the_wild(Goods)->syllable
CUDA_VISIBLE_DEVICES=0,1 python train.py --train_data /home/ubuntu/syllable/train \
--valid_data /home/ubuntu/syllable/validation  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-0316-wild/best_accuracy.pth --num_iter 5000 --valInterval 100 & > /dev/null;
while true; do sleep 120; printf ".";done

# TPS ResNet BiLSTM Attn(#3) 0317
# Text_in_the_wild(Goods)->syllable->word
CUDA_VISIBLE_DEVICES=0 python train.py --train_data /home/ubuntu/word/train \
--valid_data /home/ubuntu/word/validation  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-wild-syllable-0317/best_accuracy.pth --num_iter 50000 --valInterval 100 & > /dev/null;
while true; do sleep 120; printf ".";done

## TPS ResNet BiLSTM Attn(#3)
#CUDA_VISIBLE_DEVICES=0,1 python train.py --train_data /home/ubuntu/word/train \
#--valid_data /home/ubuntu/word/validation  \
#--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
#--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
#--saved_model ./models/TPS-ResNet-BiLSTM-Attn-0316/best_accuracy.pth --num_iter 5000 --valInterval 100

## TPS ResNet BiLSTM CTC
# 0315 Text in the wild -> syllable 처리가 되지 않고 비어 보임
# Text_in_the_wild(Goods)->syllable
CUDA_VISIBLE_DEVICES=1 python train.py \
--train_data /home/ubuntu/syllable/train \
--valid_data /home/ubuntu/syllable/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-0311/best_accuracy.pth --num_iter 5000 --valInterval 100

# 0315 Text in the wild -> word
# Text_in_the_wild(Goods)->word
CUDA_VISIBLE_DEVICES=1 python train.py \
--train_data /home/ubuntu/word/train \
--valid_data /home/ubuntu/word/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-0311/best_accuracy.pth --num_iter 5000 --valInterval 100

# 0316 Text in the wild -> sentence
CUDA_VISIBLE_DEVICES=1 python train.py \
--train_data /home/ubuntu/sentence/train \
--valid_data /home/ubuntu/sentence/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-0315/best_accuracy.pth --num_iter 5000 --valInterval 100

## 처음부터 다시 해보자 3/16
# syllable(음절), output_channel 512 -> 256 
CUDA_VISIBLE_DEVICES=1,0 python train.py \
--train_data /home/ubuntu/syllable/train \
--valid_data /home/ubuntu/syllable/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--num_iter 5000 --valInterval 100 --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# syllable(음절) -> word(단어), output_channel 512 -> 256 
CUDA_VISIBLE_DEVICES=1,0 python train.py \
--train_data /home/ubuntu/word/train \
--valid_data /home/ubuntu/word/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed0318/best_accuracy.pth --num_iter 5000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# syllable(음절:1069) -> word(단어) -> Text_in_the_wild(Goods), output_channel 512 -> 256
CUDA_VISIBLE_DEVICES=1,0 python train.py \
--train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed0319/best_accuracy.pth --num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 0322 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth --num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
# syllable(음절:1069) -> word(단어) -> Text_in_the_wild(Goods), output_channel 512 -> 256
CUDA_VISIBLE_DEVICES=0 python train.py \
--train_data /home/ubuntu/zip1/train \
--valid_data /home/ubuntu/zip1/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 0323 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed322/best_accuracy.pth --num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# syllable(음절:1069) -> word(단어) -> Text_in_the_wild(Goods), output_channel 512 -> 256
# > self_1_data -> sentence
#TPS-ResNet-BiLSTM-Attn-wild-syllable-0317
CUDA_VISIBLE_DEVICES=0 python train.py \
--train_data /home/ubuntu/sentence/train \
--valid_data /home/ubuntu/sentence/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 0327 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-wild-syllable-0317/best_accuracy.pth --num_iter 10000 --valInterval 100  --output_channel 512 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=0 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 0328 \
--num_iter 5000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=0 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 0329 \
--FT \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed328/best_accuracy.pth \
--num_iter 5000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=0,1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--data_filtering_off \
--workers 0 \
--manualSeed 0331 \
--batch_max_length 256 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed328/best_accuracy.pth \
--output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
#--valInterval 2000
#--FT \
#--batch_size 192
#--num_iter 300000 
CUDA_VISIBLE_DEVICES=0,1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_1/train \
--valid_data /home/ubuntu/self_1_data_lmdb_1/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--data_filtering_off \
--workers 0 \
--manualSeed 0332 \
--batch_max_length 256 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed328/best_accuracy.pth \
--output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=0,1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_1/train \
--valid_data /home/ubuntu/self_1_data_lmdb_1/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--data_filtering_off \
--workers 0 \
--manualSeed 0333 \
--batch_max_length 256 \
--imgH 32 --imgW 32 \
--output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done


# printed up scale(음절:2350)
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/zip1/train \
--valid_data /home/ubuntu/zip1/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1323 \
--num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# printed up scale(음절:2350) -> self data
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb/train \
--valid_data /home/ubuntu/self_1_data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1324 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1323/best_accuracy.pth \
--num_iter 50000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# printed up scale(음절:2350) -> self data -> word data
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb/train \
--valid_data /home/ubuntu/self_1_data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1325 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1324/best_accuracy.pth \
--num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# printed up scale(음절:2350) -> self data -> word data -> self data(+en)
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb/train \
--valid_data /home/ubuntu/self_1_data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1326 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1325/best_accuracy.pth \
--num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# printed up scale(음절:2350) -> self data -> word data -> self data(+en) -> (+hand-write)
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb/train \
--valid_data /home/ubuntu/self_1_data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1327 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1326/best_accuracy.pth \
--num_iter 50000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# printed up scale(음절:2350) -> self data -> word data -> self data(+en) -> (+hand-write) -> word +
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 600 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1328 --FT \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1327/best_accuracy.pth \
--num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done

# self data(printed ko, en, hanwirte ko)
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1328 \
--num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1329 \
--FT \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1328/best_accuracy.pth \
--num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1330 \
--FT \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1329/best_accuracy.pth \
--num_iter 10000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1331 \
--FT \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1330/best_accuracy.pth \
--num_iter 15000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/word/train \
--valid_data /home/ubuntu/word/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--manualSeed 1332 \
--FT \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1331/best_accuracy.pth \
--num_iter 15000 --valInterval 100  --output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
# --FT \
#--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth \
#--imgH 256 --imgW 256 \
#--imgH 95 --imgW 256 \
# multi-GPU error case
#https://github.com/clovaai/deep-text-recognition-benchmark/issues/96
while true; do sleep 120; printf ".";done


CUDA_VISIBLE_DEVICES=1 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_2/train \
--valid_data /home/ubuntu/self_1_data_lmdb_2/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--data_filtering_off --workers 0 \
--manualSeed 1333 \
--batch_max_length 256 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1331/best_accuracy.pth \
--output_channel 256 & > /dev/null;
while true; do sleep 120; printf ".";done
#--workers 4 
#--batch_max_length 25
#--valInterval 2000
#--FT \
#--batch_size 192
#--num_iter 300000 

CUDA_VISIBLE_DEVICES=1,0 python train_2350.py \
--train_data /home/ubuntu/self_1_data_lmdb_1/train \
--valid_data /home/ubuntu/self_1_data_lmdb_1/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--data_filtering_off --workers 0 \
--manualSeed 1334 \
--batch_max_length 256 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1331/best_accuracy.pth \
--output_channel 256 & > /dev/null&
while true; do sleep 120; printf ".";done
