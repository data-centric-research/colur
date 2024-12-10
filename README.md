
# COLUR: Learning, Unlearning and Relearning from Data with Uncertain Labels via Model Confidence Refinement

<div align="center">

 [:sparkles: Overview](#sparkles-overview) | [:computer: Usage](#computer-usage) | [:link: License](#link-License)

<div align="left">


<! -- ADD WARNING: Only the core code is retained, and code such as baseline repetition, jupyter notebook for results analysis, and trained models are provided if needed. -->

## :sparkles: Overview

<!-- insert the pdf of framework from assets/framework.pdf -->
<img src="assets/framework.png" width="100%" style="border: none;"></img>

## :computer: Usage

### :rainbow: Create environment

```bash
conda create -n ${env_name} -r requirements.txt
```

### :rocket: Generate Dataset

#### :zap: Generate symmetric noise

Different ratios of symmetric noises are injected into Cifar-10 and Flower-102 dataset.

```bash
# CIFAR-10
PYTHONPATH=${code_base} python gen_dataset/gen_cifar10_exp_data_cvpr.py \
--dataset_name cifar-10 \
--data_dir ./data/cifar-10/normal \
--gen_dir ./data/cifar-10/gen \
--split_ratio 0.4 \
--noise_type symmetric \
--noise_ratio 0.1
```

```bash
# Flower-102
PYTHONPATH=${code_base} python gen_dataset/gen_flower102_exp_data_cvpr.py \
--dataset_name flower-102 \
--data_dir ./data/flower-102/normal/flowers-102/ \
--gen_dir ./data/flower-102/gen \
--split_ratio 0.4 \
--noise_type symmetric \
--noise_ratio 0.1
```

`split_ratio` for different dataset may vary.
`noise_ratio` may be 0.1/0.25/0.5/0.75/0.9.

#### :zap: Generate asymmetric noise

Different ratios of symmetric noises are injected into Cifar-100 and Oxford-IIIT Pet dataset.

```bash
# CIFAR-100
PYTHONPATH=${code_base} python gen_dataset/gen_cifar100_exp_data_cvpr.py \
--dataset_name cifar-100 \
--data_dir ./data/cifar-100/normal \
--gen_dir ./data/cifar-100/gen \
--split_ratio 0.6 \
--noise_type asymmetric \
--noise_ratio 0.1
```

```bash
# Oxford-IIIT Pet
PYTHONPATH=${code_base} python gen_dataset/gen_pet37_exp_data_cvpr.py \
--dataset_name pet-37 \
--data_dir ./data/pet-37/normal/oxford-pets \
--gen_dir ./data/pet-37/gen \
--split_ratio 0.3 \
--noise_type asymmetric \
--noise_ratio 0.1
```

`split_ratio` for different dataset may vary.
`noise_ratio` may be 0.1/0.25/0.5/0.75/0.9.

### :fire: Pre-Train

**Pre-train on CIFAR-10.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python ./run_experiment_cvpr.py \
--model efficientnet_s \
--dataset cifar-10 \
--num_epochs 30 \
--train_mode pretrain \
--learning_rate 1e-3 \
--optimizer adam \
--batch_size 256 \
```

**Pre-train on Flower-102.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python ./run_experiment_cvpr.py \
--model wideresnet50 \
--dataset flower-102 \
--num_epochs 20 \
--train_mode pretrain \
--learning_rate 1e-3 \
--optimizer adam \
--batch_size 256

```

**Pre-train on CIFAR-100.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python run_experiment_cvpr.py \
--model efficientnet_s \
--dataset cifar-100 \
--num_epochs 400 \
--train_mode pretrain \
--learning_rate 2e-4 \
--optimizer adam \
--batch_size 256 \
--data_aug
```

**Pre-train on Oxford-IIIT Pet.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python ./run_experiment_cvpr.py \
--model wideresnet50 \
--dataset pet-37 \
--num_epochs 15 \
--train_mode inc_train \
--learning_rate 2e-5 \
--optimizer adam \
--batch_size 16 \
--noise_type asymmetric \
--noise_ratio 0.1
```

### :fire: Inc-Train

#### :zap: Incremental training models on symmetric dataset

**Inc-train on CIFAR-10.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python ./run_experiment_cvpr.py \
--model efficientnet_s \
--dataset cifar-10 \
--num_epochs 30 \
--train_mode inc_train \
--learning_rate 1e-4 \
--optimizer adam \
--batch_size 256 \
--noise_type symmetric \
--noise_ratio 0.1
```

`noise_ratio` may be 0.1/0.25/0.5/0.75/0.9.

**Inc-train on Flower-102.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python ./run_experiment_cvpr.py \
--model wideresnet50 \
--dataset flower-102 \
--num_epochs 15 \
--train_mode inc_train \
--learning_rate 1e-3 \
--optimizer adam \
--batch_size 256 \
--noise_type symmetric \
--noise_ratio 0.1
```

`noise_ratio` may be 0.1/0.25/0.5/0.75/0.9.

#### :zap: Incremental training models on asymmetric dataset

**Inc-train on CIFAR-100.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python ./run_experiment_cvpr.py \
--model efficientnet_s \
--dataset cifar-100 \
--num_epochs 50 \
--train_mode inc_train \
--learning_rate 1e-4 \
--optimizer adam \
--batch_size 256 \
--noise_type asymmetric \
--noise_ratio 0.1
```

`noise_ratio` may be 0.1/0.25/0.5/0.75/0.9.

**Inc-train on Oxford-IIIT Pet.**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python ./run_experiment_cvpr.py \
--model wideresnet50 \
--dataset pet-37 \
--num_epochs 15 \
--train_mode inc_train \
--learning_rate 2e-5 \
--optimizer adam \
--batch_size 16 \
--noise_type asymmetric \
--noise_ratio 0.1
```

`noise_ratio` may be 0.1/0.25/0.5/0.75/0.9.

### :zap: Executing Baselines

**LNL methods**

1. <https://github.com/bhanML/Co-teaching>
2. <https://github.com/xingruiyu/coteaching_plus>
3. <https://github.com/emalach/UpdateByDisagreement>
4. <https://github.com/JackYFL/DISC>
5. <https://github.com/shengliu66/ELR>
6. <https://github.com/ErikEnglesson/GJS>
7. <https://github.com/hongxin001/JoCoR>
8. <https://github.com/ydkim1293/NLNL-Negative-Learning-for-Noisy-Labels>
9. <https://github.com/yikun2019/PENCIL>

```bib
@inproceedings{han2018coteaching,
  title = {Co-teaching: Robust training of deep neural networks with extremely noisy labels},
  author = {Bo Han and Quanming Yao and Xingrui Yu and Gang Niu and Miao Xu and Weihua Hu and Ivor W. Tsang and Masashi Sugiyama},
  booktitle = {NeurIPS},
  year = {2018}
}

@inproceedings{co-teaching+,
  title={How does disagreement help generalization against label corruption?},
  author={Yu, Xingrui and Han, Bo and Yao, Jiangchao and Niu, Gang and Tsang, Ivor and Sugiyama, Masashi},
  booktitle={International conference on machine learning},
  pages={7164--7173},
  year={2019},
  organization={PMLR}
}

@article{decoupling,
  title={Decoupling" when to update" from" how to update"},
  author={Malach, Eran and Shalev-Shwartz, Shai},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@inproceedings{DISC,
  title={Disc: Learning from noisy labels via dynamic instance-specific selection and correction},
  author={Li, Yifan and Han, Hu and Shan, Shiguang and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24070--24079},
  year={2023}
}

@article{ELR,
  title={Early-learning regularization prevents memorization of noisy labels},
  author={Liu, Sheng and Niles-Weed, Jonathan and Razavian, Narges and Fernandez-Granda, Carlos},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={20331--20342},
  year={2020}
}

@article{GJS,
  title={Generalized jensen-shannon divergence loss for learning with noisy labels},
  author={Englesson, Erik and Azizpour, Hossein},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={30284--30297},
  year={2021}
}

@inproceedings{JoCoR,
  title={Combating noisy labels by agreement: A joint training method with co-regularization},
  author={Wei, Hongxin and Feng, Lei and Chen, Xiangyu and An, Bo},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={13726--13735},
  year={2020}
}

@inproceedings{NLNL,
  title={Nlnl: Negative learning for noisy labels},
  author={Kim, Youngdong and Yim, Junho and Yun, Juseung and Kim, Junmo},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={101--110},
  year={2019}
}

@inproceedings{PENCIL,
  title={Probabilistic end-to-end noise correction for learning with noisy labels},
  author={Yi, Kun and Wu, Jianxin},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={7017--7025},
  year={2019}
}

```

**Machine Unlearning Methods:**

1. <https://github.com/lmgraves/AmnesiacML>
2. <https://github.com/OPTML-Group/Unlearn-Sparse>
3. <https://github.com/IST-DASLab/WoodFisher>

```bib

@inproceedings{graves2021amnesiac,
  title={Amnesiac machine learning},
  author={Graves, Laura and Nagisetty, Vineel and Ganesh, Vijay},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={13},
  pages={11516--11524},
  year={2021}
}

@article{jia2023model,
  title={Model sparsity can simplify machine unlearning},
  author={Liu, Jiancheng and Ram, Parikshit and Yao, Yuguang and Liu, Gaowen and Liu, Yang and SHARMA, PRANAY and Liu, Sijia and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@article{WF,
  title={Woodfisher: Efficient second-order approximation for neural network compression},
  author={Singh, Sidak Pal and Alistarh, Dan},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={18098--18109},
  year={2020}
}

```

### :hammer: Check results

After executing scripts of generating experimental dataset, you can check the generated data under the follow directory.

```bash
# cd your data directory
tree data
```

After executing scripts of training models, you can check the trained models under the follow directory.

```bash
# cd your model directory
tree ckpt
```

### :bar_chart: Evaluation

We wrote several jupyter notebooks for analysis, since they are not core code, we will not provide it as supplymentary material. However, we are very pleased to open them if anyone who need them.

## :scroll: License

This repository respects to MIT license.
