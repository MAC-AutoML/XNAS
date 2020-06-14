# Auto_NAS_V2

## Requirements
- Pytorch 1.0.1.post2
- Python 3.6+
- DALI

## Usuage

Step1: Go into your project path

```bash
cd /userhome/project/pytorch_image_classification; 
```

Step2: Move data to memory

```bash
./script/data_to_memory.sh cifar10
./script/data_to_memory.sh imagenet
```
Step3.1 : Run

```bash
# run on CIFAR-10 with ofa search space
python one_shot_search.py --search_space ofa --width_mult 1.2

# run on ImageNet with ofa search space
python one_shot_search.py --search_space ofa --width_mult 1.2 --dataset ImageNet --data_path /gdata/ImageNet2012/ --print_freq 1

```
