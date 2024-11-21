import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import pruner
import utils
from pruner import extract_mask, prune_model_custom, remove_prune
from core_model.train_test import model_test

from configs import settings

sys.path.append(".")
from trainer import validate


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args, filename='eval_result.pth.tar'):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.save_dir,
        args.unlearn,
        filename=filename,
    )


def load_unlearn_checkpoint(model, device, args, filename="checkpoint.pth.tar"):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn, filename)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    # todo 屏蔽，只需要加载模型
    # current_mask = pruner.extract_mask(checkpoint["state_dict"])
    # pruner.prune_model_custom(model, current_mask)
    # pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # adding an extra forward process to enable the masks
    # x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    # model.eval()
    # with torch.no_grad():
    #     model(x_rand)W

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args):
        test_loader = data_loaders["test"]
        model_history = []
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.num_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.arch == "swin_t":
            optimizer = torch.optim.Adam(model.parameters(), args.unlearn_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.num_epochs
            )
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.num_epochs):
            start_time = time.time()
            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args
            )
            scheduler.step()

            if test_loader is not None:
                eval_results = model_test(test_loader, model)
                model_history.append(eval_results)

            print("one epoch duration:{}".format(time.time() - start_time))

        return model_history

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
