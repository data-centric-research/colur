#  python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.75 --noise_type asymmetric --dataset pet-37

import argparse
import sys
import pandas as pd
import os

sys.path.append(os.path.abspath(".."))

import torch
import numpy as np
from configs import settings
from core_model.custom_model import ClassifierWrapper, load_custom_model
from core_model.dataset import get_dataset_loader
from core_model.train_test import model_forward


def execute(set_noise_ratio, set_noise_type, set_dataset):

    # set_noise_ratio='0.75'
    # set_noise_type='asymmetric'
    # 0.5 对称 CRUL,ELR,GJS,wfisher的名字为efficientnet_s;发现fisher_new,FT_prune_bi没用

    cifar_model_name = "efficientnet_s"
    other_model_name = "wideresnet50"

    # raw疑似pretrain?
    set_basic_name = "pretrain,inctrain"
    set_LNL_name = "Coteaching,Coteachingplus,Decoupling,DISC,ELR,GJS,JoCoR,NegativeLearning,PENCIL"
    set_MU_name = "FT,GA,GA_l1,wfisher"
    set_OUR_name = "CRUL"
    set_uni_name = f"{set_basic_name},{set_LNL_name},{set_MU_name},{set_OUR_name}"

    # set_dataset='pet-37'
    set_model_suffix = "restore"
    set_batch_size = 64

    case = settings.get_case(set_noise_ratio, set_noise_type)
    uni_names = set_uni_name
    uni_names = [uni_names] if uni_names is None else uni_names.split(",")
    num_classes = settings.num_classes_dict[set_dataset]

    _, _, test_loader = get_dataset_loader(
        set_dataset,
        "test",
        None,
        batch_size=set_batch_size,
        shuffle=False,
    )
    _, _, noisy_loader = get_dataset_loader(
        set_dataset,
        "train_noisy",
        case,
        batch_size=set_batch_size,
        shuffle=False,
        label_name="train_noisy_true_label",
    )

    results_data = []

    for uni_name in uni_names:
        print(f"Evaluating {uni_name}:")
        dict_temp = {}
        model_name = cifar_model_name
        if set_dataset == "cifar-10" or set_dataset == "cifar-100":
            model_name = cifar_model_name
        else:
            model_name = other_model_name

        loaded_model = load_custom_model(model_name, num_classes, load_pretrained=False)
        model = ClassifierWrapper(loaded_model, num_classes)

        if uni_name == "pretrain":
            model_ckpt_path = settings.get_ckpt_path(
                set_dataset,
                "",
                model_name,
                model_suffix="pretrain",
                unique_name=uni_name,
            )
        elif uni_name == "inctrain":
            model_ckpt_path = settings.get_ckpt_path(
                set_dataset,
                case,
                model_name,
                model_suffix="inc_train",
                unique_name="",
            )
        else:
            model_ckpt_path = settings.get_ckpt_path(
                set_dataset,
                case,
                model_name,
                model_suffix=set_model_suffix,
                unique_name=uni_name,
            )
        print(f"Loading model from {model_ckpt_path}")
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint, strict=False)
        # print(f"Evaluating test_data:")
        # results, embedding = model_test(test_loader, model)
        # print("Results: %.4f" % results)
        torch.cuda.empty_cache()
        print(f"Evaluating train_noisy_data:")
        n_results, n_embedding = model_test(noisy_loader, model)
        # print("Results: %.4f" % results)
        # print("Results: ", n_results)
        dict_temp = {"uni_name": uni_name, **n_results}
        results_data.append(dict_temp)

    df = pd.DataFrame(results_data)
    return df


def model_test(data_loader, model, device="cuda"):
    eval_results = {}

    predicts, probs, embedding, labels = model_forward(
        data_loader, model, device, output_embedding=True, output_targets=True
    )

    # global acc
    global_acc = np.mean(predicts == labels)
    eval_results["global"] = global_acc.item()

    # error_rate
    eval_results["error_rate"] = 1 - eval_results["global"]
    # class acc
    label_list = sorted(list(set(labels)))
    for label in label_list:
        cls_index = labels == label
        class_acc = np.mean(predicts[cls_index] == labels[cls_index])
        eval_results["label_" + str(label.item())] = class_acc.item()

    return eval_results, embedding


def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Generate with noise and dataset settings."
    )

    # 添加命令行参数
    # parser.add_argument('--save_path', type=str, required=True, help="Path to save the results.")
    parser.add_argument(
        "--noise_rate", type=str, required=True, help="Noise rate (float value)."
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        required=True,
        help="Noise type (either 'sym' or 'asymmetric').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (either 'pet-37' or 'cifar10').",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    # save_path = args.save_path
    noise_rate = args.noise_rate
    noise_type = args.noise_type
    dataset = args.dataset

    set_noise_ratio, set_noise_type, set_dataset = noise_rate, noise_type, dataset

    # 根据参数动态生成 save_path
    save_path = f"result_analysis/visualize_results_cvpr/{dataset}_{noise_rate}_{noise_type}.csv"

    # 打印或者使用这些参数
    print(f"Save path: {save_path}")
    print(f"Noise rate: {set_noise_ratio}")
    print(f"Noise type: {set_noise_type}")
    print(f"Dataset: {set_dataset}")

    # 根据这些参数执行进一步的操作
    # 例如，生成结果并保存到 `save_path`
    # 这里你可以插入你的实际代码逻辑，比如加载数据集、添加噪声、训练模型等

    # 把所有模型的结果存到一个dataframe 里面
    results = pd.DataFrame()
    results = execute(set_noise_ratio, set_noise_type, set_dataset)

    results.to_csv(save_path)


if __name__ == "__main__":
    main()
