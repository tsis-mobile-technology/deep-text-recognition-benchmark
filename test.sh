#!/bin/bash
. ~/easy-ocr/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn  \
--imgH 32 --imgW 100 --output_channel 256 \
--saved_model ./saved_models/None-VGG-BiLSTM-Attn-Seed334/best_accuracy.pth \
--batch_size 384 --batch_max_length 256 --data_filtering_off --workers 0 & > /dev/null 
deactivate

#CUDA_VISIBLE_DEVICES=0,1 python test.py \
#--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
#--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn  \
#--imgH 32 --imgW 100 --output_channel 256 \
#--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed332/best_accuracy.pth \
#--batch_size 384 --batch_max_length 256 --data_filtering_off --workers 0 & > /dev/null

#--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth --batch_size 150 \
#--saved_model ./models/best_accuracy.pth \
#--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed333/iter_100000.pth --batch_size 384 \

# 0307 15:48
#accuracy: IIIT5k_3000: 54.533	SVT: 49.459	IC03_860: 78.140	IC03_867: 76.932	IC13_857: 72.345	IC13_1015: 69.754	IC15_1811: 33.683	IC15_2077: 30.525	SVTP: 36.279	CUTE80: 39.931	total_accuracy: 51.512	averaged_infer_time: 1.681	# parameters: 49.123
# 0307 18:45
#accuracy: IIIT5k_3000: 57.967	SVT: 52.396	IC03_860: 80.349	IC03_867: 79.700	IC13_857: 73.979	IC13_1015: 72.020	IC15_1811: 37.604	IC15_2077: 33.606	SVTP: 41.240	CUTE80: 38.542	total_accuracy: 54.537	averaged_infer_time: 1.607	# parameters: 49.123
# 0308 09:18
#accuracy: IIIT5k_3000: 63.133	SVT: 57.032	IC03_860: 83.488	IC03_867: 82.699	IC13_857: 78.530	IC13_1015: 76.453	IC15_1811: 41.414	IC15_2077: 37.313	SVTP: 42.791	CUTE80: 45.139	total_accuracy: 58.656	averaged_infer_time: 1.641	# parameters: 49.123
# 0311 17:00
#accuracy: IIIT5k_3000: 62.500	SVT: 59.660	IC03_860: 83.837	IC03_867: 83.276	IC13_857: 78.180	IC13_1015: 76.847	IC15_1811: 42.076	IC15_2077: 38.036	SVTP: 43.101	CUTE80: 46.875	total_accuracy: 58.996	averaged_infer_time: 1.505	# parameters: 49.123

# 0332
#accuracy: IIIT5k_3000: 0.233	SVT: 0.000	IC03_860: 0.000	IC03_867: 0.000	IC13_857: 0.000	IC13_1015: 0.000	IC15_1811: 0.000	IC15_2077: 0.000	SVTP: 0.000	CUTE80: 0.000	total_accuracy: 0.058	averaged_infer_time: 4.283	# parameters: 18.925
# 0333
#accuracy: IIIT5k_3000: 0.067	SVT: 0.000	IC03_860: 0.000	IC03_867: 0.000	IC13_857: 0.000	IC13_1015: 0.197	IC15_1811: 0.000	IC15_2077: 0.000	SVTP: 0.000	CUTE80: 0.000	total_accuracy: 0.033	averaged_infer_time: 3.296	# parameters: 18.925
