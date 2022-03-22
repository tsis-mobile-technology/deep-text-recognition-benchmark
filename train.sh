#!/bin/bash
#source ~/easy-ocr/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python train.py --train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
#--sensitive \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111 \
--num_iter 200000 --valInterval 100 & > /dev/null

#--imgH 256 --imgW 256 \
#--imgH 95 --imgW 256 \
# multi-GPU error case
#https://github.com/clovaai/deep-text-recognition-benchmark/issues/96
while true; do sleep 120; printf ".";done

#!/bin/bash
#source ~/easy-ocr/bin/activate
# case old
python train.py --train_data /home/ubuntu/Text_in_the_wild/data_lmdb/train \
--valid_data /home/ubuntu/Text_in_the_wild/data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--data_filtering_off --workers 0 --imgH 200 --imgW 200 --batch_size 32 \
--saved_model ./saved_model/TPS-ResNet-BiLSTM-CTC.pth

# self data
python train.py --train_data ../aihub_data/self_data_lmdb/train \
--valid_data ../aihub_data/self_data_lmdb/validation \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--batch_size 150 --batch_max_length 256 --data_filtering_off --workers 0 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth \
--manualSeed 1112 --num_iter 2000 --valInterval 100

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

# case 2 for mac
python train.py --train_data /Users/gotaejong/ExternHard/97_Workspace/jupyter/Text_in_the_wild/data_lmdb/train --valid_data /Users/gotaejong/ExternHard/97_Workspace/jupyter/Text_in_the_wild/data_lmdb/validation --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --batch_size 150 --batch_max_length 200 --data_filtering_off --workers 0 --num_iter 100000 --valInterval 100
# while true; do sleep 120; printf ".";done
# encoding 에러가 발생하여 (train.py) 0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㆍ가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘",
# --> 중간에 '"'을 제거하여 실행함.

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

# at nhn cloud
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

# syllable(음절) -> word(단어) -> Text_in_the_wild(Goods), output_channel 512 -> 256
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

