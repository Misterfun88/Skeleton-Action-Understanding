cuda_device=$1
test_name=$2


if [ ! -d ./checkpoints/${test_name} ];then
    mkdir -p ./checkpoints/${test_name}_xsub

fi
 CUDA_VISIBLE_