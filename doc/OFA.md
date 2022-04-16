# OFA超网训练

OFA超网训练分为下述4个步骤

1. 最大网络训练: 对应`configs/search/OFA/mbv3_cifar10/OFA_normal_phase1.yaml`
2. elastic kernel size: 对应`configs/search/OFA/mbv3_cifar10/OFA_kernel_phase1.yaml`
3. elastic depth： 对应`configs/search/OFA/mbv3_cifar10/OFA_depth_phase1.yaml`和`configs/search/OFA/mbv3_cifar10/OFA_depth_phase2.yaml`
4. elastic width: 对应`configs/search/OFA/mbv3_cifar10/OFA_expand_phase1.yaml`和`configs/search/OFA/mbv3_cifar10/OFA_expand_phase2.yaml `

---

OFA超网训练示例：ofa_mbv3_cifar10
先创建文件夹`./experiment/OFA/mbv3_cifar10`，然后运行如下命令
```bash
nohup bash search/OFA/train_supernet.sh ./experiment/OFA/mbv3_cifar10/train_supernet.log 2>&1 &
```
