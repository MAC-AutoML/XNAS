# XNAS

**XNAS** is an effective neural architecture search codebase, written in [PyTorch](https://pytorch.org/).

## Installation

```bash
git clone https://github.com/MAC-AutoML/XNAS.git
cd XNAS
```

## Usage

```bash
# set root path
export PYTHONPATH=$PYTHONPATH:/Path/to/XNAS
# set gpu devices
export CUDA_VISIBLE_DEVICES=0
# unit test example
python tools/test_func/sng_function_optimization.py
# train example
python train/DARTS_train.py --cfg configs/search/darts.yaml
# replace config example
python train/DARTS_train.py --cfg configs/search/darts.yaml OUT_DIR /username/project/XNAS/experiment/darts/test1
```
