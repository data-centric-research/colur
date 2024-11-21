import os
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys

from args_paser import parse_args, parse_kwargs
from lip_teacher import SimpleLipNet
from dataset import MixupDataset, get_dataset_loader
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from train_test import model_train, model_test, model_forward
from configs import settings


def train_teacher_model(
    args,
    step,
    num_classes,
    teacher_model,
    teacher_opt,
    teacher_lr_scheduler,
    teacher_criterion,
    save_path,
    mean=None,
    std=None,
    train_dataloader=None,
    test_dataloader=None,
    test_per_it=1,
):

    case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
    if train_dataloader is None:
        _, _, train_dataloader = get_dataset_loader(
            args.dataset,
            "train",
            case,
            step,
            mean,
            std,
            args.batch_size,
            num_classes=num_classes,
            drop_last=True,
            shuffle=True,
            onehot_enc=False,
        )

    if test_dataloader is None:
        _, _, test_dataloader = get_dataset_loader(
            args.dataset,
            "test",
            case,
            None,
            mean=None,
            std=None,
            batch_size=args.batch_size,
            num_classes=num_classes,
            drop_last=False,
            shuffle=False,
            onehot_enc=False,
        )

    model_train(
        train_dataloader,
        teacher_model,
        teacher_opt,
        teacher_lr_scheduler,
        teacher_criterion,
        args.num_epochs,
        args,
        save_path=save_path,
        mix_classes=num_classes,
        test_loader=test_dataloader,
        test_per_it=test_per_it,
    )


def iterate_repair_model(
    working_model,
    working_opt,
    working_lr_schedule,
    working_criterion,
    working_model_save_path,
    teacher_model,
    teacher_opt,
    teacher_lr_schedule,
    teacher_criterion,
    teacher_model_save_path,
    inc_data,
    inc_labels,
    inc_dataloader,
    aux_data,
    aux_labels,
    aux_dataloader,
    num_classes,
    mean,
    std,
    device,
    args,
):
    aux_labels_onehot = np.eye(num_classes)[aux_labels]

    # 1. 通过 Mt 获取 D_mix=Da+Ds+Dc,  Train Pp=Mp(Xp_mix), Loss=CrossEntropy(Pp, Yp_mix)
    # (1) 获取Ds: 通过 Yp=Mp(Xtr), Yt=Mt(Xtr) 两个模型预测分类标签，其中Yp=Yt=Ytr的数据为Ds, Ys为预测相同的标签
    working_inc_predicts, working_inc_probs = model_forward(
        inc_dataloader, working_model
    )
    teacher_inc_predicts, teacher_inc_probs, teacher_inc_embeddings = model_forward(
        inc_dataloader, teacher_model, output_embedding=True
    )

    agree_idx = working_inc_predicts == teacher_inc_predicts
    select_idx = agree_idx & (teacher_inc_predicts == inc_labels)
    selected_data = inc_data[select_idx]
    selected_labels = inc_labels[select_idx]
    selected_probs = teacher_inc_probs[select_idx]  # + working_inc_probs[select_idx])/2
    selected_embeddings = teacher_inc_embeddings[select_idx]

    # (2) 获取Dc: 通过 Mt(Xa+Xs) 计算class embedding centroids (i.e. Class mean): E_centroid
    # 通过 Mt(Das)获取 E_centroid(Das=Da+Ds)
    # 获取 Embedding_disa(D_disa=Dtr-D_agree) 距离离每个类c的中心 E_centroid[class=c]最近的Top 10% 的数据为 Dc (通过Lipschitz性质预测的伪标签)
    aux_predicts, aux_probs, aux_embeddings = model_forward(
        aux_dataloader, teacher_model, output_embedding=True
    )

    teacher_agree_embeddings = np.concatenate(
        [aux_embeddings, selected_embeddings], axis=0
    )
    agree_labels = np.concatenate([aux_labels, selected_labels], axis=0)

    disagree_idx = working_inc_predicts != teacher_inc_predicts
    disagree_data = inc_data[disagree_idx]
    teacher_disagree_predicts = teacher_inc_predicts[disagree_idx]
    teacher_disagree_probs = teacher_inc_probs[disagree_idx]
    teacher_disagree_embeddings = teacher_inc_embeddings[disagree_idx]

    centroid_data, centroid_probs = [], []

    for label in list(set(aux_labels)):
        agree_class_embedding = teacher_agree_embeddings[agree_labels == label]
        agree_class_embedding_centroid = np.mean(agree_class_embedding, axis=0)

        disagree_class_idx = teacher_disagree_predicts == label
        disagree_class_embeddings = teacher_disagree_embeddings[disagree_class_idx]
        disagree_class_data = disagree_data[disagree_class_idx]
        disagree_class_probs = teacher_disagree_probs[disagree_class_idx]

        distances = np.linalg.norm(
            disagree_class_embeddings - agree_class_embedding_centroid, axis=-1
        )
        selected_top_conf_num = len(disagree_class_probs) // 10
        top_idx = np.argpartition(distances, selected_top_conf_num)[
            selected_top_conf_num:
        ]

        centroid_class_data, centroid_class_label = (
            disagree_class_data[top_idx],
            disagree_class_probs[top_idx],
        )
        centroid_data.extend(centroid_class_data)
        centroid_probs.extend(centroid_class_label)

    centroid_data = np.array(centroid_data)
    centroid_probs = np.array(centroid_probs)
    centroid_probs_sharpen = sharpen(centroid_probs)

    # (3) train Mp: Train Pp=Mp(X_mix), Loss=CrossEntropy(Pp, Y_mix)
    mix_data = np.concatenate([aux_data, selected_data, centroid_data], axis=0)
    # selected_labels_onehot = np.eye(num_classes)[selected_labels]
    mix_labels_onehot = np.concatenate(
        [aux_labels_onehot, selected_probs, centroid_probs_sharpen], axis=0
    )
    mix_dataloader_shuffled = mix_up_dataloader(
        mix_data,
        mix_labels_onehot,
        mix_data,
        mix_labels_onehot,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        alpha=0.2,
        transforms=None,
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
        save_path=working_model_save_path,
    )

    # 2. 使用 D_mix=Da+Ds+Dc(重新mix_up), Train Pt=Mt(Xp_mix), Loss=CrossEntropy(Pt, Yt_mix)
    # teacher_mix_dataloader_shuffled = mix_up_dataloader(
    #     mix_data, mix_labels_onehot, mix_data, mix_labels_onehot,
    #     mean=mean, std=std, batch_size=args.batch_size, transform=True)

    model_train(
        mix_dataloader_shuffled,
        teacher_model,
        teacher_opt,
        teacher_lr_schedule,
        teacher_criterion,
        args.num_epochs,
        args,
        device=device,
        save_path=teacher_model_save_path,
    )

    # 3. 获取Dconf{Xs, Ys} 用与adapt: Dconf从Ds中top10% 数据(根据Ys_prob排序)
    select_probs_max = np.max(selected_probs, axis=-1)  # [N]
    sample_size = len(selected_probs) // 10
    sample_idx = np.argpartition(select_probs_max, -sample_size)[-sample_size:]
    conf_data = selected_data[sample_idx]
    conf_labels = selected_probs[sample_idx]

    return conf_data, conf_labels


def iterate_adapt_model(
    working_model,
    working_opt,
    working_lr_scheduler,
    working_criterion,
    working_model_save_path,
    teacher_model,
    teacher_opt,
    teacher_lr_scheduler,
    teacher_criterion,
    teacher_model_save_path,
    aug_data,
    aug_probs,
    test_data,
    test_dataloader,
    mean,
    std,
    device,
    args,
):
    # 1. 构造Dts融合数据集 Dt_mix: (Dts, D_aug), 进行mix up
    # (1) 构造 Dts: Dt={Xts, Pts}, Pt = Mt(Xts)
    test_predicts, test_probs = model_forward(test_dataloader, teacher_model)

    # (2) 构造 Dt_mix: Dt_mix = mix_up(Dts, D_aug), Xt_mix = {a*Xts+(1-a)*X_aug}, Yt_mix = {a*Pts+(1-a)*Y_aug}
    test_probs_sharpen = sharpen(test_probs)
    ts_mixed_dataloader_shuffled = mix_up_dataloader(
        test_data,
        test_probs_sharpen,
        aug_data,
        aug_probs,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        alpha=0.2,
        transforms=None,
    )

    # 2. train Mt: Pt=Mt(Xt_max), Update Mt: Loss=CrossEntropy(Pt, Yp_mix)
    model_train(
        ts_mixed_dataloader_shuffled,
        teacher_model,
        teacher_opt,
        teacher_lr_scheduler,
        teacher_criterion,
        args.adapt_epochs,
        args,
        device=device,
        save_path=teacher_model_save_path,
    )

    # 3. 重新构造 Dts融合数据集 Dp_mix
    # (1) 构造 Dts: Dt={Xts, Pts}, Pt = Mt(Xts)
    test_predicts_new, test_probs_new = model_forward(test_dataloader, teacher_model)

    # (2) 构造 Dp_mix: Dp_mix = mix_up(Dts, D_aug), Xp_mix = {a*Xts+(1-a)*X_aug}, Yt_mix = {a*Pts+(1-a)*Y_aug}
    test_probs_new_sharpen = sharpen(test_probs_new)
    ts_mixed_dataloader_shuffled_new = mix_up_dataloader(
        test_data,
        test_probs_new_sharpen,
        aug_data,
        aug_probs,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        alpha=0.2,
        transforms=None,
    )

    # 4. train Mp: Pp=Mp(Xp_max), Update Mp: Loss=CrossEntropy(Pp, Yp_mix)
    model_train(
        ts_mixed_dataloader_shuffled_new,
        working_model,
        working_opt,
        working_lr_scheduler,
        working_criterion,
        args.adapt_epochs,
        args,
        device=device,
        save_path=working_model_save_path,
    )


def mix_up_dataloader(
    inc_data,
    inc_probs,
    aug_data,
    aug_probs,
    mean,
    std,
    batch_size,
    alpha=1,
    transforms=None,
    shuffle=True,
):
    mixed_dataset = MixupDataset(
        data_pair=(inc_data, aug_data),
        label_pair=(inc_probs, aug_probs),
        mixup_alpha=alpha,
        transforms=transforms,
        mean=mean,
        std=std,
    )
    return DataLoader(mixed_dataset, batch_size, drop_last=True, shuffle=shuffle)


def sharpen(prob_max, T=1, axis=-1):
    prob_max = prob_max ** (1.0 / T)
    return prob_max / np.sum(prob_max, axis=-1, keepdims=True)


def execute(args):
    # 1. 获取公共参数
    num_classes = settings.num_classes_dict[args.dataset]
    kwargs = parse_kwargs(args.kwargs)
    case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(settings.root_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_path = os.path.join(log_path, "core_execution.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    learning_rate = getattr(args, "learning_rate", 0.001)
    lr_scale = getattr(args, "lr_scale", 2.0)
    working_lr = learning_rate * lr_scale
    weight_decay = getattr(args, "weight_decay", 5e-4)
    repair_iter_num = getattr(args, "repair_iter_num", 2)
    adapt_iter_num = getattr(args, "adapt_iter_num", 2)
    optimizer_type = getattr(args, "optimizer", "adam")
    num_epochs = getattr(args, "num_epochs", 50)
    step = getattr(args, "step", 1)
    tta_only = getattr(args, "tta_only", None)
    uni_name = getattr(args, "uni_name", None)
    spec_norm = not args.no_spnorm

    working_model_path = settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        model_suffix="worker_raw",
        step=step,
        unique_name=uni_name,
    )  # model_paths["working_model_path"]
    working_model_repair_save_path = settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        model_suffix="worker_restore",
        step=step,
        unique_name=uni_name,
    )
    working_model_adapt_save_path = settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        model_suffix="worker_tta",
        step=step,
        unique_name=uni_name,
    )
    lip_teacher_model_path = settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        model_suffix="teacher_restore",
        step=step - 1,
        unique_name=uni_name,
    )
    teacher_model_repair_save_path = settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        model_suffix="teacher_restore",
        step=step,
        unique_name=uni_name,
    )
    teacher_model_adapt_save_path = settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        model_suffix="teacher_tta",
        step=step,
        unique_name=uni_name,
    )

    mean, std = None, None
    # if args.dataset == "cifar-10":
    #     mean = cifar10_config["mean"]
    #     std = cifar10_config["std"]
    # elif args.dataset == "cifar-100":
    #     mean = cifar100_config["mean"]
    #     std = cifar100_config["std"]
    # elif args.dataset == "food-101":
    #     mean = food101_config["mean"]
    #     std = food101_config["std"]

    working_model, lip_teacher_model = None, None

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
    # features = backbone.fc.in_features
    # backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    # lip_teacher_model = SimpleLipNet(backbone, features, num_classes, spectral_norm=spec_norm)
    lip_teacher_model = ClassifierWrapper(
        backbone, num_classes, spectral_norm=spec_norm
    )
    # lip_teacher_model.to(device)

    # 根据用户选择的优化器初始化
    teacher_opt, teacher_lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        lip_teacher_model.parameters(),
        num_epochs,
        learning_rate,
        weight_decay,
    )
    teacher_criterion = nn.CrossEntropyLoss()

    aux_data, aux_labels, aux_dataloader = get_dataset_loader(
        args.dataset, "aux", case, None, mean, std, args.batch_size, shuffle=False
    )

    test_data, test_labels, test_dataloader = get_dataset_loader(
        args.dataset, "test", case, None, mean, std, args.batch_size, shuffle=False
    )

    conf_data_path = settings.get_dataset_path(args.dataset, case, "conf_data", step)
    conf_label_path = settings.get_dataset_path(args.dataset, case, "conf_label", step)
    subdir = os.path.dirname(conf_data_path)
    os.makedirs(subdir, exist_ok=True)

    if tta_only is None:
        checkpoint = torch.load(working_model_path)
        working_model.load_state_dict(checkpoint, strict=False)
        # working_model.to(device)

        if os.path.exists(lip_teacher_model_path):
            checkpoint = torch.load(lip_teacher_model_path)
            lip_teacher_model.load_state_dict(checkpoint, strict=False)
            print("load teacher model from :", lip_teacher_model_path)
        else:
            # t0 的情况下，使用D0数据重新训练 lip_teacher model
            print(
                "Teacher model pth: %s not exist, only train T0, if not T0 then stop!"
                % lip_teacher_model_path
            )

            train_teacher_model(
                args,
                0,
                num_classes,
                lip_teacher_model,
                teacher_opt,
                teacher_lr_scheduler,
                teacher_criterion,
                lip_teacher_model_path,
                test_dataloader=None,
                test_per_it=1,
            )

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
            test_dataloader, lip_teacher_model, device=device
        )

        # (2) 构造修复过程数据集: Dtr、 Da、Dts
        inc_data, inc_labels, inc_dataloader = get_dataset_loader(
            args.dataset, "train", case, step, mean, std, args.batch_size, shuffle=False
        )

        # (3) 迭代修复过程：根据 Dtr 迭代 Mp 、 Mt
        conf_data, conf_labels = None, None
        for i in range(repair_iter_num):
            print("-----------restore iterate %d ----------------------" % i)
            conf_data, conf_labels = iterate_repair_model(
                working_model,
                working_opt,
                working_lr_scheduler,
                working_criterion,
                working_model_repair_save_path,
                lip_teacher_model,
                teacher_opt,
                teacher_lr_scheduler,
                teacher_criterion,
                teacher_model_repair_save_path,
                inc_data,
                inc_labels,
                inc_dataloader,
                aux_data,
                aux_labels,
                aux_dataloader,
                num_classes,
                mean,
                std,
                device,
                args,
            )

        np.save(conf_data_path, conf_data)
        np.save(conf_label_path, conf_labels)

        # 4. 测试修复后 Dts 在 Mp 的表现
        working_model_after_repair = model_test(
            test_dataloader, working_model, device=device
        )
        teacher_model_after_repair = model_test(
            test_dataloader, lip_teacher_model, device=device
        )

        logging.info(f"Test results before restore: {working_model_test_before}")
        logging.info(f"Test results after restore: {working_model_after_repair}")

        print(
            "---------------------working model test before------------------------------"
        )
        print(working_model_test_before)
        print(
            "---------------------teacher model test before------------------------------"
        )
        print(teacher_model_test_before)

        print(
            "---------------------working model test after restore------------------------------"
        )
        print(working_model_after_repair)
        print(
            "---------------------teacher model test after restore------------------------------"
        )
        print(teacher_model_after_repair)

    # 5. 迭代测试数据适应过程
    # (1) 构造适应过程数据：Dts, D_aug:  = Da + Dconf
    aux_labels_onehot = np.eye(num_classes)[aux_labels]
    aug_data = aux_data
    aug_labels = aux_labels_onehot

    if tta_only is not None:
        if tta_only == 0:
            lip_teacher_model_path = settings.get_ckpt_path(
                args.dataset,
                case,
                args.model,
                model_suffix="teacher_restore",
                step=step,
                unique_name=uni_name,
            )
            checkpoint = torch.load(lip_teacher_model_path)
            lip_teacher_model.load_state_dict(checkpoint, strict=False)

            checkpoint = torch.load(working_model_path)
            working_model.load_state_dict(checkpoint, strict=False)
        else:
            conf_data = np.load(conf_data_path)
            conf_labels = np.load(conf_label_path)

            checkpoint = torch.load(teacher_model_repair_save_path)
            lip_teacher_model.load_state_dict(checkpoint, strict=False)
            checkpoint = torch.load(working_model_repair_save_path)
            working_model.load_state_dict(checkpoint, strict=False)

            aug_data = np.concatenate([aux_data, conf_data], axis=0)
            aug_labels = np.concatenate([aux_labels_onehot, conf_labels], axis=0)

            print(
                "---------------------working model test before------------------------------"
            )
            model_test(test_dataloader, working_model, device=device)
            print(
                "---------------------teacher model test before------------------------------"
            )
            model_test(test_dataloader, lip_teacher_model, device=device)

    # (2) 迭代测试数据适应过程：根据 混合的Dts 迭代 Mp 和 Mt
    for i in range(adapt_iter_num):
        print("-----------tta iterate %d ----------------------" % i)
        iterate_adapt_model(
            working_model,
            working_opt,
            working_lr_scheduler,
            working_criterion,
            working_model_adapt_save_path,
            lip_teacher_model,
            teacher_opt,
            teacher_lr_scheduler,
            teacher_criterion,
            teacher_model_adapt_save_path,
            aug_data,
            aug_labels,
            test_data,
            test_dataloader,
            mean,
            std,
            device,
            args,
        )

    # 6. 测试适应后 Dts 在 Mp 的表现
    working_model_after_adapt = model_test(
        test_dataloader, working_model, device=device
    )
    teacher_model_after_adapt = model_test(
        test_dataloader, lip_teacher_model, device=device
    )

    print(
        "---------------------working model test after tta-------------------------------"
    )
    print(working_model_after_adapt)
    print(
        "---------------------teacher model test after tta-------------------------------"
    )
    print(teacher_model_after_adapt)

    logging.info(f"Test results after adaptation: {working_model_after_adapt}")


if __name__ == "__main__":
    try:
        parse_args = parse_args()
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)
    execute(parse_args)
