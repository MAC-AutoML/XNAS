## Getting Started

XNAS does not provide installation via `pip` currently. To run XNAS, `python>=3.7` and `pytorch==1.9` are required. Other versions of `PyTorch` may also work well, but there are potential API differences that can cause warnings to be generated.

We have listed other requirements in `requirements.txt` file.

## Installation

1. Clone this repo.
2. (Optional) Create a virtualenv for this library.
```sh
virtualenv venv
```
3. Install dependencies.
```sh
pip install -r requirements.txt
```
4. Set the `$PYTHONPATH` environment variable.
```sh
export PYTHONPATH=$PYTHONPATH:/Path/to/XNAS
```
5. Set the visible GPU device for XNAS. Currently XNAS supports single GPU only, but we will provide support for multi-GPUs soon.
```sh
export CUDA_VISIBLE_DEVICES=0
```

Notably, environment variables are **valid only for the current terminal**. For ease of use, we recommend adding commands within your environment profile (like `~/.bashrc` for `bash`) to automatically configure environment variables after login:

```sh
echo "export PYTHONPATH=$PYTHONPATH:/Path/to/XNAS" >> ~/.bashrc
```

some search spaces or algorithms supported by XNAS require specific APIs provided by NAS benchmarks. Installation and properly setting are required to run these code.

Benchmarks supported by XNAS and their linkes are following.
- nasbench101: [GitHub](https://github.com/google-research/nasbench)
  - nasbench1shot1: [GitHub](https://github.com/automl/nasbench-1shot1)
- nasbench201: [GitHub](https://github.com/D-X-Y/NAS-Bench-201)
- nasbench301: [GitHub](https://github.com/automl/nasbench301)

For detailed instructions to install these benchmarks, please refer to the `$XNAS/docs/benchmarks` directory.


## Usage

Before running code in XNAS, please make sure you have followed instructions in [**Data_preparation.md**](./Data_preparation.md) in our docs to complete preparing the necessary data.

The main program entries for the search and training process are in the `$XNAS/scripts` folder. To modify and add NAS code, please place files in this folder.

### Configuration Files

XNAS uses the `.yaml` file format to organize the configuration files. All configuration files are placed under `$XNAS/configs` directory. To ensure the uniformity and clarity of files, we strongly recommend using the following naming convention:

```sh
Algorithm_Space_Dataset[_Evaluation][_ModifiedParams].yaml
```

For example, using `DARTS` algorithm, searching on `NASBench201` space and `CIFAR-10` dataset, evaluated by `NASBench301` while modifying `MAX_EPOCH` parameter to `75`, then the file should be named as this:

```sh
darts_nasbench201_cifar10_nasbench301_maxepoch75.yaml
```

### Running Examples

XNAS reads configuration files from the command line. A simple running example is following:

```sh
python scripts/search/DARTS.py --cfg configs/search/darts_darts_cifar10.yaml
```

The configuration file can be overridden by adding or modifying additional parameters on the command line. For example, run with the modified output directory:

```sh
python scripts/search/DARTS.py --cfg configs/search/darts_darts_cifar10.yaml OUT_DIR exp/another_folder
```

Using `.sh` files to save commands is very efficient for when you need to run and modify parameters repeatedly. We provide shell scripts under `$XNAS/examples` folder, together with other potential test code added in the future. It can be simply run with the following command:

```sh
./examples/darts_darts_cifar10.sh
```

A common mistake is forgetting to add run permissions to these files:

```sh
chmod +x examples/*/*.sh examples/*/*/*.sh
```

The script files follow the same naming convention as the configuration file above, and set the output directory to the same folder. You can achieve a continuous search/training process by adding multiple lines of commands to the script file at once.

