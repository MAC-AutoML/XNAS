<div align="center">

<img src="https://cdn.thrase.cn/xnas%2Fheader.png" width="200">

<p>
	<a href="https://img.shields.io/badge/Python-%3E%3D3.7-blue"><img src="https://img.shields.io/badge/Python-%3E%3D3.7-blue"></a>
	<a href="https://img.shields.io/badge/PyTorch-1.9-informational"><img src="https://img.shields.io/badge/PyTorch-1.9-informational"></a>
	<a href="https://img.shields.io/badge/License-MIT-brightgreen"><img src="https://img.shields.io/badge/License-MIT-brightgreen"></a>
  <a href="https://img.shields.io/badge/Docs-latest-yellowgreen"><img src="https://img.shields.io/badge/Docs-latest-yellowgreen"></a>
</p>
</div>



<br>

**XNAS** is an effective, modular and flexible Neural Architecture Search (NAS) repository, which aims to provide a common framework and baselines for the NAS community. It is originally designed to decouple the search space, search algorithm and performance evaluation strategy to achieve freely combinable NAS.

This project is now supported by PengCheng Lab.

---

[**Overview**](#Overview) | [**Installation**](#Installation) | [**Contributing**](#Contributing) | [**Citation**](#Citation) | [**License**](#License)

**For more information and API usages, please refer to our** [**Documentation**](https://xnas.readthedocs.io).

<br>

## Overview

Based on a common division of NAS, the project is organized by **search space, search algorithm**, and **evaluation strategy**. The project currently supports the content shown below.

We also provide the interpretation of papers and experimental records for each algorithm. For more information, please refer to the links in the "**Docs**" column.

|       Search Spaces       |   Docs   |                        Official Links                        |
| :-----------------------: | :------: | :----------------------------------------------------------: |
|           DARTS           |          |         [`Github`](https://github.com/quark0/darts)          |
|      ~~MobileNetV3~~      |          |                              -                               |
|     ~~NAS-Bench-101~~     |          |   [`GitHub`](https://github.com/google-research/nasbench)    |
|       NAS-Bench-201       |          |      [`GitHub`](https://github.com/D-X-Y/NAS-Bench-201)      |
|     NAS-Bench-1Shot1      |          |    [`GitHub`](https://github.com/automl/nasbench-1shot1)     |
|           SPOS            |          | [`GitHub`](https://github.com/megvii-model/SinglePathOneShot) |
|   **Search Algorithms**   | **Docs** |                      **Official Links**                      |
|           DARTS           |          |         [`Github`](https://github.com/quark0/darts)          |
|          PDARTS           |          |       [`Github`](https://github.com/chenxin061/pdarts)       |
|          PCDARTS          |          |     [`Github`](https://github.com/yuhuixu1993/PC-DARTS)      |
|            SNG            |          |      [`Github`](https://github.com/shirakawas/ASNG-NAS)      |
|           ASNG            |          |      [`Github`](https://github.com/shirakawas/ASNG-NAS)      |
|          MDENAS           |          |       [`Github`](https://github.com/tanglang96/MDENAS)       |
|          DDPNAS           |          |       [`Github`](https://github.com/tanglang96/DDPNAS)       |
|          MIGONAS          |          |          [`Openi`](https://git.openi.org.cn/PCL_AutoML/XNAS/src/branch/dev)          |
|        GridSearch         |          |                              -                               |
|           DrNAS           |          |     [`Github`](https://github.com/xiangning-chen/DrNAS)      |
|          RMINAS           |          |          [`Openi`](https://git.openi.org.cn/PCL_AutoML/XNAS/src/branch/dev)          |
|          DropNAS          |          |      [`Github`](https://github.com/wiljohnhong/dropnas)      |
|           SPOS            |          | [`Github`](https://github.com/megvii-model/SinglePathOneShot) |
|            OFA            |          |   [`Github`](https://github.com/mit-han-lab/once-for-all)    |
| **Evaluation Strategies** | **Docs** |                      **Official Links**                      |
|     ~~NAS-Bench-101~~     |          |   [`GitHub`](https://github.com/google-research/nasbench)    |
|       NAS-Bench-201       |          |      [`GitHub`](https://github.com/D-X-Y/NAS-Bench-201)      |
|       NAS-Bench-301       |          |      [`GitHub`](https://github.com/automl/nasbench301)       |
|     NAS-Bench-1Shot1      |          |    [`GitHub`](https://github.com/automl/nasbench-1shot1)     |

We are gradually providing support for more settings.

## Installation

To run XNAS, `python>=3.7` and `pytorch=1.9` are required. Other versions of `PyTorch` may also work well, but there are potential API differences that can cause warnings to be generated.

For detailed instructions, please refer to [**Getting_started.md**](./docs/Getting_started.md) and [**Data_preparation.md**](./docs/Data_preparation.md) in our docs.

## Contributing

We welcome contributions to the library along with any potential issues or suggestions.

Please refer to [**Contributing.md**](./docs/Contributing.md) in our docs for more information.

## Citation

If you use this code in your own work, please use the following bibtex entries:

```bash
@inproceedings{zheng2022rminas,
  title={Neural Architecture Search with Representation Mutual Information},
  author={Xiawu Zheng, Xiang Fei, Lei Zhang, Chenglin Wu, Fei Chao, Jianzhuang Liu, Wei Zeng, Yonghong Tian, Rongrong Ji},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
@article{zheng2021migo,
  title={MIGO-NAS: Towards fast and generalizable neural architecture search},
  author={Zheng, Xiawu and Ji, Rongrong and Chen, Yuhang and Wang, Qiang and Zhang, Baochang and Chen, Jie and Ye, Qixiang and Huang, Feiyue and Tian, Yonghong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
@inproceedings{zheng2020rethinking,
  title={Rethinking performance estimation in neural architecture search},
  author={Zheng, Xiawu and Ji, Rongrong and Wang, Qiang and Ye, Qixiang and Li, Zhenguo and Tian, Yonghong and Tian, Qi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11356--11365},
  year={2020}
}
```

## License

This project is released under the [MIT license](https://mit-license.org).

## TODO

- 迁移OFA代码
- 补充101&201安装测试
  - 检查201搜索空间
  - 检查RMINAS
- 补充模块测试案例
- 多显卡支持
