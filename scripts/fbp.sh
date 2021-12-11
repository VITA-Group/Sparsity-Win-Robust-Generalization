# flying bird +
CUDA_VISIBLE_DEVICES=$1 python -u main_adv.py \
    --data $2 \
    --dataset cifar10 \
    --arch resnet18 \
    --save_dir result \
    --density 0.2 \
    --dynamic_fre \
    --fbp