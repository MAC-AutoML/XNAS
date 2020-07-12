# 如何使用

## XNAS

XNAS 是一个轻量级高效的框架，囊括目前广泛使用的搜索空间（cell-based， chain-structure）以及搜索算法（darts），并且针对部件进行封装，让新手可以快速的上手开发自己的NAS搜索算法。

## Darts （可微分搜索方法）

入口文件为 `` `train_darts.py` ` `以及 ` ` `configs/search/darts.yaml` ` `。 ` ` `train_darts.py` ` ` 从 ` ` `configs/search/darts.yaml` ``当中读取相关超参数，来进行环境设置，数据加载以及优化器的选择以及构建。

具体运行方法为：

``` bash
cd xx(项目地址); PYTHONPATH=./ python tools/train_darts.py --cfg configs/search/darts.yaml
```

假设我们需要对darts.yaml文件中的超参数进行覆盖，那么利用key value来进行额外指定。比如在下面的命令中，会将OUT_DIR这个参数由原来的/userhome/project/XNAS/experiment/darts/test 覆盖为 /userhome/project/XNAS/experiment/darts/test1, 其他参数也同理，由于使用了yacs，相关的库会自动进行解析，例如 key True 也是一个正确的用法。

``` bash
cd /userhome/project/XNAS; PYTHONPATH=./ python tools/train_darts.py --cfg configs/search/darts.yaml OUT_DIR /userhome/project/XNAS/experiment/darts/test1
```
