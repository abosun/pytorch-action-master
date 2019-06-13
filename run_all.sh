#export CUDA_VISIBLE_DEVICES=2
./train_Time.py --dataset UCF101 --split 1
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
