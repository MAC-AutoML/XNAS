#### OFA超网训练示例：ofa_mbv3_cifar10

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./

# 1. 最大网络训练
python search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_normal_phase1.yaml
# 2. elastic kernel size
python search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_kernel_phase1.yaml
# 3. elastic depth
python search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_depth_phase1.yaml
python search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_depth_phase2.yaml
# 4. elastic width
python search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_expand_phase1.yaml
python search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_expand_phase2.yaml
