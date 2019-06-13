#CUDA_VISIBLE_DEVICES=0 ./train_Spa.py --a 0.1 & CUDA_VISIBLE_DEVICES=1 ./train_Spa.py --a 0.01 & CUDA_VISIBLE_DEVICES=2 ./train_Spa.py --a 0.001 &
#CUDA_VISIBLE_DEVICES=1 ./train_Spa.py --dataset HMDB51 --a 1 
#CUDA_VISIBLE_DEVICES=1 ./train_Spa.py --dataset HMDB51 --a 0.1 
#CUDA_VISIBLE_DEVICES=1 ./train_Spa.py --dataset HMDB51 --a 0.01   
CUDA_VISIBLE_DEVICES=1 ./train_Spa.py --dataset HMDB51 --a 100

#CUDA_VISIBLE_DEVICES=3 ./train_Spa.py --dataset HMDB51 --a 0.001   
#CUDA_VISIBLE_DEVICES=3 ./train_Spa.py --dataset HMDB51 --a 0.0001   
#CUDA_VISIBLE_DEVICES=3 ./train_Spa.py --dataset HMDB51 --a 0.00001
:<<EOF
./train_Spa.py --dataset UCF101 --split 1

./train_Time.py --dataset UCF101 --split 2
./train_Time.py --dataset UCF101 --split 3

./train_Time.py --dataset HMDB51 --split 1
./train_Time.py --dataset HMDB51 --split 2
./train_Time.py --dataset HMDB51 --split 3

./train_Spa.py --dataset UCF101 --split 1
./train_Spa.py --dataset UCF101 --split 2
./train_Spa.py --dataset UCF101 --split 3

./train_Spa.py --dataset HMDB51 --split 1
./train_Spa.py --dataset HMDB51 --split 2
./train_Spa.py --dataset HMDB51 --split 3
EOF
