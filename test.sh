#!/bin/bash
python test.py \
--eval_data ./data_lmdb_sample/evaluation --benchmark_all_eval \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--saved_model ../trainning_model/TPS-ResNet-BiLSTM-CTC-syllable-word-wild-0316/best_accuracy.pth \
--batch_size 300 --batch_max_length 256 --data_filtering_off --workers 0 \
--imgH 32 --output_channel 512 --hidden_size 256
#--saved_model ../trainning_model/TPS-ResNet-BiLSTM-CTC-Seed1111/craft_mlt_25k.pth \
#--saved_model ../trainning_model/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth \


python demo.py \
--image_folder ./demo_image \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--saved_model ../trainning_model/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth \
--batch_size 300 --batch_max_length 256 --workers 0 \
--imgH 32 --output_channel 256 --hidden_size 256

python demo.py \
--image_folder ./demo_image \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--saved_model ../trainning_model/TPS-ResNet-BiLSTM-CTC-syllable-word-wild-0316/best_accuracy.pth \
--batch_size 300 --batch_max_length 256 --workers 0 \
--imgH 32 --output_channel 512 --hidden_size 256

#case ubuntu GPU
#CUDA_VISIBLE_DEVICES=0 python test.py \
#--eval_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation --benchmark_all_eval \
#--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
#--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth \
#--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 1

