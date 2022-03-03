#!/bin/bash

#sentence
python create_lmdb_dataset.py \
--inputPath /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/sentence/train \
--gtFile /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/sentence/gt_train.txt \
--outputPath /home/proidea/jupyter/aihub_study_data/data_lmdb/sentence/train

python create_lmdb_dataset.py \
--inputPath /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/sentence/validation \
--gtFile /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/sentence/gt_validation.txt \
--outputPath /home/proidea/jupyter/aihub_study_data/data_lmdb/sentence/train

#syllable
python create_lmdb_dataset.py \
--inputPath /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/syllable/train \
--gtFile /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/syllable/gt_train.txt \
--outputPath /home/proidea/jupyter/aihub_study_data/data_lmdb/syllable/train

python create_lmdb_dataset.py \
--inputPath /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/syllable/validation \
--gtFile /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/syllable/gt_validation.txt \
--outputPath /home/proidea/jupyter/aihub_study_data/data_lmdb/syllable/train

#word
python create_lmdb_dataset.py \
--inputPath /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/word/train \
--gtFile /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/word/gt_train.txt \
--outputPath /home/proidea/jupyter/aihub_study_data/data_lmdb/word/train

python create_lmdb_dataset.py \
--inputPath /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/word/validation \
--gtFile /home/proidea/jupyter/aihub_study_data/korean_spelling_save_image/word/gt_validation.txt \
--outputPath /home/proidea/jupyter/aihub_study_data/data_lmdb/word/train
