# 1. 最大网络训练
python scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_normal_phase1.yaml
# 2. elastic kernel size
python scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_kernel_phase1.yaml
# 3. elastic depth
python scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_depth_phase1.yaml
python scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_depth_phase2.yaml
# 4. elastic width
python scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_expand_phase1.yaml
python scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3_cifar10/OFA_expand_phase2.yaml
