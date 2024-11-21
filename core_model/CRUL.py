import os
import logging
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import sys

from args_paser import parse_args, parse_kwargs
from lip_teacher import SimpleLipNet
from dataset import MixupDataset, get_dataset_loader, NormalizeDataset
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from train_test import model_train, model_test, model_forward
from configs import settings


def mixup_data(data_pair, dims, alpha):
    data_x, data_y = data_pair
    size = [1] * dims
    size[0] = len(data_x)
    a = np.random.beta(alpha, alpha, size)
    # a = np.maximum(a, 1-a)
    return a * data_x + (1 - a) * data_y


def iterate_repair_model(
    working_model,
    working_opt,
    working_lr_schedule,
    ul_worker_opt,
    ul_worker_lr_scheduler,
    working_criterion,
    teacher_model,
    teacher_opt,
    teacher_lr_scheduler,
    ul_teacher_opt,
    ul_teacher_lr_scheduler,
    teacher_criterion,
    inc_data,
    inc_labels,
    inc_dataloader,
    num_classes,
    mean,
    std,
    device,
    args
):
    working_inc_predicts, working_inc_probs = model_forward(
        inc_dataloader, working_model
    )
    teacher_inc_predicts, teacher_inc_probs = model_forward(
        inc_dataloader, teacher_model, output_embedding=False
    )

    """1. Unlearning confident disagreement data"""
    disagree_threshold = 0.75
    tradeoff_alpha = 2/3
    disagree_idx = working_inc_predicts != teacher_inc_predicts
    disagree_data = inc_data[disagree_idx]
    teacher_disagree_probs = teacher_inc_probs[disagree_idx]
    worker_disagree_probs = working_inc_probs[disagree_idx]
    worker_disagree_preds = working_inc_predicts[disagree_idx]
    # teacher_disagree_preds = teacher_inc_predicts[disagree_idx]
    teacher_disagree_model_conf = np.max(teacher_disagree_probs, axis=-1)
    worker_disagree_model_conf = np.max(worker_disagree_probs, axis=-1)
    joint_disagree_model_conf = np.sqrt(teacher_disagree_model_conf * worker_disagree_model_conf)
        # (tradeoff_alpha * worker_disagree_model_conf + (1 - tradeoff_alpha) * teacher_disagree_model_conf)
    disagree_conf_idx = joint_disagree_model_conf >= disagree_threshold

    mix_data = disagree_data[disagree_conf_idx]
    gamma = len(mix_data)/(len(disagree_data) + 1e-6)

    ga_loss_alpha = -1.0  # GA
    mix_worker_labels = worker_disagree_preds[disagree_conf_idx]
    mix_worker_labels = label_smooth(mix_worker_labels, num_classes, gamma=gamma)
    if args.ul_epochs > 0 and len(mix_worker_labels) > 0:
        print("Unlearning high-confidence for worker model...")

        forget_dataset = NormalizeDataset(
            mix_data, mix_worker_labels, mean=mean, std=std
        )
        forget_dataloader_shuffled = DataLoader(
            forget_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True
        )
        model_train(
            forget_dataloader_shuffled,
            working_model,
            ul_worker_opt,
            ul_worker_lr_scheduler,
            working_criterion,
            args.ul_epochs,
            args,
            device=device,
            loss_lambda=ga_loss_alpha,
        )

    # unlearn_teacher = False
    # mix_teacher_labels = teacher_disagree_preds[disagree_conf_idx]
    # mix_teacher_labels = label_smooth(mix_teacher_labels, num_classes, gamma=0.4)
    # if args.ul_epochs > 0 and len(mix_teacher_labels) > 0:
    #     print("Unlearning high-confidence for teacher model...")
    #
    #     forget_dataset = NormalizeDataset(
    #         mix_data, mix_teacher_labels, mean=mean, std=std
    #     )
    #     forget_dataloader_shuffled = DataLoader(
    #         forget_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True
    #     )
    #     model_train(
    #         forget_dataloader_shuffled,
    #         teacher_model,
    #         ul_teacher_opt,
    #         ul_teacher_lr_scheduler,
    #         teacher_criterion,
    #         args.ul_epochs,
    #         args,
    #         device=device,
    #         loss_lambda=ga_loss_alpha,
    #     )

    """2. Refine low-confidence agreement and disagreement data"""
    top_conf = 0.3
    agree_threshold = 0.75
    agree_idx = working_inc_predicts == teacher_inc_predicts
    agree_data = inc_data[agree_idx]
    teacher_agree_probs = teacher_inc_probs[agree_idx]
    worker_agree_probs = teacher_inc_probs[agree_idx]
    agree_predicts = teacher_inc_predicts[agree_idx]
    teacher_agree_model_conf = np.max(teacher_agree_probs, axis=-1)
    worker_agree_model_conf = np.max(worker_agree_probs, axis=-1)
    joint_agree_model_conf = np.sqrt(teacher_agree_model_conf * worker_agree_model_conf)
    sample_size = round(len(joint_agree_model_conf) * top_conf)
    sample_idx = np.argpartition(joint_agree_model_conf, -sample_size)[-sample_size:]
    conf_idx = joint_agree_model_conf >= agree_threshold
    conf_idx[sample_idx] = True

    conf_agree_data = agree_data[conf_idx]
    conf_agree_labels = agree_predicts[conf_idx]
    # conf_agree_mix_labels = np.eye(num_classes)[conf_agree_labels]
    conf_agree_mix_labels = (teacher_agree_probs[conf_idx] + worker_agree_probs[conf_idx])/2

    disagree_mix_data = disagree_data[~disagree_conf_idx]
    agree_mix_data = agree_data[~conf_idx]

    worker_disagree_lc_probs = worker_disagree_probs[~disagree_conf_idx]
    worker_agree_lc_probs = worker_agree_probs[~conf_idx]

    teacher_disagree_lc_probs = teacher_disagree_probs[~disagree_conf_idx]
    teacher_agree_lc_probs = teacher_agree_probs[~conf_idx]

    if args.num_epochs > 0:
        temperature = args.temperature
        conf_agree_mix_labels = sharpen(conf_agree_mix_labels, T=temperature)
        mix_lc_data = np.concatenate([disagree_mix_data, agree_mix_data], axis=0)

        print("Mix up lower confidence for worker model...")
        alpha = 1.2
        disagree_mix_labels = mixup_data((teacher_disagree_lc_probs, worker_disagree_lc_probs), 2, alpha)
        # tradeoff_alpha * teacher_disagree_lc_probs + (1 - tradeoff_alpha) * worker_disagree_lc_probs
        agree_mix_labels = mixup_data((teacher_agree_lc_probs, worker_agree_lc_probs), 2, alpha)
        # tradeoff_alpha * teacher_agree_lc_probs + (1 - tradeoff_alpha) * worker_agree_lc_probs
        mix_lc_label = np.concatenate([disagree_mix_labels, agree_mix_labels], axis=0)
        mix_lc_label = sharpen(mix_lc_label, T=temperature)
        if args.mixup_alpha == 0:
            mix_train = (mix_lc_data, mix_lc_label, mix_lc_data, mix_lc_label)
        else:
            mix_train = (mix_lc_data, mix_lc_label, conf_agree_data, conf_agree_mix_labels) \
                if len(mix_lc_label) > len(conf_agree_mix_labels) \
                else (conf_agree_data, conf_agree_mix_labels, mix_lc_data, mix_lc_label)

        mix_dataloader_shuffled = mix_up_dataloader(
            *mix_train,
            mean,
            std,
            batch_size=args.batch_size,
            alpha=args.mixup_alpha,
            transforms=None,
            first_max=False
        )

        model_train(
            mix_dataloader_shuffled,
            working_model,
            working_opt,
            working_lr_schedule,
            working_criterion,
            args.num_epochs,
            args,
            device=device,
        )

        print("Mix up lower confidence for teacher model...")

        disagree_mix_labels = mixup_data((teacher_disagree_lc_probs, worker_disagree_lc_probs), 2, alpha)
        # tradeoff_alpha * teacher_disagree_lc_probs + (1 - tradeoff_alpha) * worker_disagree_lc_probs
        agree_mix_labels = mixup_data((teacher_agree_lc_probs, worker_agree_lc_probs), 2, alpha)
        # tradeoff_alpha * teacher_agree_lc_probs + (1 - tradeoff_alpha) * worker_agree_lc_probs
        # agree_mix_labels = sharpen(agree_mix_labels, T=0.8)
        mix_lc_label = np.concatenate([disagree_mix_labels, agree_mix_labels], axis=0)
        mix_lc_label = sharpen(mix_lc_label, T=temperature)
        if args.mixup_alpha == 0:
            mix_train = (mix_lc_data, mix_lc_label, mix_lc_data, mix_lc_label)
        else:
            mix_train = (mix_lc_data, mix_lc_label, conf_agree_data, conf_agree_mix_labels) \
                if len(mix_lc_label) > len(conf_agree_mix_labels) \
                else (conf_agree_data, conf_agree_mix_labels, mix_lc_data, mix_lc_label)

        mix_dataloader_shuffled = mix_up_dataloader(
            *mix_train,
            mean,
            std,
            batch_size=args.batch_size,
            alpha=args.mixup_alpha,
            transforms=None,
            first_max=False
        )
        model_train(
            mix_dataloader_shuffled,
            teacher_model,
            teacher_opt,
            teacher_lr_scheduler,
            teacher_criterion,
            args.num_epochs,
            args,
            device=device,
        )

    """3. Refine high-confidence agreement data by label smoothing"""
    if args.agree_epochs > 0:
        print("Refine high-confidence for worker model...")
        conf_agree_probs = label_smooth(conf_agree_labels, num_classes, gamma=args.ls_gamma)
        conf_dataset = NormalizeDataset(
            conf_agree_data, conf_agree_probs, mean=mean, std=std
        )
        conf_data_loader = DataLoader(
            conf_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True
        )

        model_train(
            conf_data_loader,
            working_model,
            working_opt,
            working_lr_schedule,
            working_criterion,
            args.agree_epochs,
            args,
            device=device,
        )

        model_train(
            conf_data_loader,
            teacher_model,
            teacher_opt,
            teacher_lr_scheduler,
            teacher_criterion,
            args.agree_epochs,
            args,
            device=device,
        )


def mix_up_dataloader(
    inc_data,
    inc_probs,
    aug_data,
    aug_probs,
    mean,
    std,
    batch_size,
    alpha=1.0,
    transforms=None,
    shuffle=True,
    first_max=True
):
    mixed_dataset = MixupDataset(
        data_pair=(inc_data, aug_data),
        label_pair=(inc_probs, aug_probs),
        mixup_alpha=alpha,
        transforms=transforms,
        mean=mean,
        std=std,
        first_max=first_max
    )
    return DataLoader(mixed_dataset, batch_size, drop_last=False, shuffle=shuffle)


def sharpen(probs, T=1.0, axis=-1):
    probs = probs ** (1.0 / T)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def label_smooth(labels, num_classes, gamma=0.0):
    return np.diag(np.ones(num_classes) - gamma)[labels] + gamma / num_classes


def execute(args):
    # 1. 获取公共参数
    num_classes = settings.num_classes_dict[args.dataset]
    # kwargs = parse_kwargs(args.kwargs)
    case = settings.get_case(args.noise_ratio, args.noise_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(settings.root_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_path = os.path.join(log_path, "core_execution.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    learning_rate = getattr(args, "learning_rate", 0.001)
    lr_scale = getattr(args, "lr_scale", 1.0)
    working_lr = learning_rate
    teacher_lr = args.teacher_lr_scale * working_lr
    ul_lr = lr_scale * learning_rate
    weight_decay = getattr(args, "weight_decay", 5e-4)
    repair_iter_num = getattr(args, "repair_iter_num", 10)
    optimizer_type = getattr(args, "optimizer", "adam")
    num_epochs = getattr(args, "num_epochs", 50)
    step = getattr(args, "step", 1)
    uni_name = getattr(args, "uni_name", None)
    spec_norm = not args.no_spnorm

    working_model_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="inc_train"
    )  # model_paths["working_model_path"]
    working_model_repair_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="restore", unique_name=uni_name
    )
    working_history_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="history", unique_name=uni_name
    )
    teacher_model_path = settings.get_ckpt_path(
        args.dataset, "pretrain", args.model, model_suffix="pretrain"
    )
    teacher_model_repair_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="teacher_restore", unique_name=uni_name,
    )
    teacher_history_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="teacher_history", unique_name=uni_name,
    )

    mean, std = None, None

    # 2. load model
    # (1) load working model
    working_model = load_custom_model(args.model, num_classes, load_pretrained=False)

    working_model = ClassifierWrapper(working_model, num_classes)

    working_opt, working_lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        working_model.parameters(),
        num_epochs,
        working_lr,
        weight_decay,
    )

    working_criterion = nn.CrossEntropyLoss()

    # (2) load lip_teacher model, t0的情况重新训练
    backbone = load_custom_model(args.model, num_classes)
    teacher_model = ClassifierWrapper(backbone, num_classes)

    # 根据用户选择的优化器初始化
    teacher_opt, teacher_lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        teacher_model.parameters(),
        num_epochs,
        teacher_lr,
        weight_decay,
    )
    teacher_criterion = nn.CrossEntropyLoss()

    ul_worker_opt = optim.SGD(working_model.parameters(), lr=ul_lr)
    ul_worker_lr_scheduler = optim.lr_scheduler.StepLR(ul_worker_opt, step_size=1, gamma=0.9)
    ul_teacher_opt = optim.SGD(teacher_model.parameters(), lr=ul_lr)
    ul_teacher_lr_scheduler = optim.lr_scheduler.StepLR(ul_teacher_opt, step_size=1, gamma=0.9)


    # aux_data, aux_labels, aux_dataloader = get_dataset_loader(
    #     args.dataset, "aux", None, None, mean, std, args.batch_size, shuffle=False
    # )

    test_data, test_labels, test_dataloader = get_dataset_loader(
        args.dataset, "test", None, None, mean, std, args.batch_size, shuffle=False
    )

    checkpoint = torch.load(working_model_path)
    working_model.load_state_dict(checkpoint, strict=False)
    # working_model.to(device)

    checkpoint = torch.load(teacher_model_path)
    teacher_model.load_state_dict(checkpoint, strict=False)
    print("load teacher model from :", teacher_model_path)

    # 3. 迭代修复过程
    # (1) 测试修复前 Dts 在 Mp 的表现
    print(
        "---------------------working model test before------------------------------"
    )
    working_model_test_before = model_test(
        test_dataloader, working_model, device=device
    )
    print(
        "---------------------teacher model test before------------------------------"
    )
    teacher_model_test_before = model_test(
        test_dataloader, teacher_model, device=device
    )

    # (2) 构造修复过程数据集: Dtr、 Da、Dts
    inc_data, inc_labels, inc_dataloader = get_dataset_loader(
        args.dataset,
        ["train_clean", "train_noisy"],
        case,
        step,
        mean,
        std,
        args.batch_size,
        shuffle=False,
    )

    # (3) 迭代修复过程：根据 Dtr 迭代 Mp 、 Mt
    worker_history = []
    teacher_history = []
    best_worker = 0
    best_teacher = 0
    for i in range(repair_iter_num):
        print("-----------restore iterate %d ----------------------" % i)
        # ga_loss_alpha = -0.5 * (1 - i/repair_iter_num)
        iterate_repair_model(
            working_model,
            working_opt,
            working_lr_scheduler,
            ul_worker_opt,
            ul_worker_lr_scheduler,
            working_criterion,
            teacher_model,
            teacher_opt,
            teacher_lr_scheduler,
            ul_teacher_opt,
            ul_teacher_lr_scheduler,
            teacher_criterion,
            inc_data,
            inc_labels,
            inc_dataloader,
            num_classes,
            mean,
            std,
            device,
            args
        )

        working_model_evals = model_test(test_dataloader, working_model, device=device)
        worker_eval_result = working_model_evals["global"]
        worker_history.append(worker_eval_result)
        if best_worker < worker_eval_result:
            best_worker = worker_eval_result
            os.makedirs(os.path.dirname(working_model_repair_save_path), exist_ok=True)
            torch.save(working_model.state_dict(), working_model_repair_save_path)
            print(f"Worker model has saved to {working_model_repair_save_path}.")

        teacher_model_evals = model_test(test_dataloader, teacher_model, device=device)
        teacher_eval_result = teacher_model_evals["global"]
        teacher_history.append(teacher_eval_result)
        if best_teacher < teacher_eval_result:
            best_teacher = teacher_eval_result
            os.makedirs(os.path.dirname(teacher_model_repair_save_path), exist_ok=True)
            torch.save(teacher_model.state_dict(), teacher_model_repair_save_path)
            print(f"Teacher model has saved to {teacher_model_repair_save_path}.")

    torch.save(worker_history, working_history_save_path)
    torch.save(teacher_history, teacher_history_save_path)
    logging.info(
        f"Test results before restore: Worker: {working_model_test_before}, Teacher: {teacher_model_test_before}"
    )
    logging.info(
        f"Test results after restore: Worker: {working_model_evals}, Teacher: {teacher_model_evals}"
    )


if __name__ == "__main__":
    try:
        parse_args = parse_args()
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)
    execute(parse_args)
