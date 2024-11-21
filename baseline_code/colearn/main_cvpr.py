import argparse
import importlib
from utils import (
    load_config,
    get_log_name,
    set_seed,
    save_results,
    plot_results,
    get_test_acc,
    print_config,
)
from datasets import cifar_dataloader
import algorithms
import numpy as np
# import nni
import torch
import os
import shutil

from core_model.dataset import get_dataset_loader
from core_model.train_test import model_test
from args_paser import parse_args
from configs import settings
from core_model.custom_model import load_custom_model, ClassifierWrapper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="./co_configs/standardCE.py",
    help="The path of config file.",
)
# args = parser.parse_args()


def main():
    # dora modify get config
    # tuner_params = nni.get_next_parameter()
    # config = load_config(args.config, _print=False)
    # config.update(tuner_params)
    # print_config(config)

    custom_args = parse_args()
    case = settings.get_case(
        custom_args.noise_ratio, custom_args.noise_type
    )
    step = custom_args.step
    uni_name = custom_args.uni_name
    num_classes = settings.num_classes_dict[custom_args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据 uni_name 动态加载对应的配置文件
    config_modules = {
        "Coteachingplus": "co_configs.coteachingplus",
        "Coteaching": "co_configs.coteaching",
        "JoCoR": "co_configs.jocor",
        "Colearning": "co_configs.colearning",
        "Decoupling": "co_configs.decoupling",
        "NegativeLearning": "co_configs.NL",
        "PENCIL": "co_configs.PENCIL",
    }

    if uni_name not in config_modules:
        raise ValueError(f"Unknown uni_name: {uni_name}")

    # 动态导入配置模块
    config_module = importlib.import_module(config_modules[uni_name])
    config = config_module.config
    config.update(custom_args.__dict__)
    config["lr"] = config["learning_rate"]
    config["epochs"] = config["num_epochs"]

    set_seed(config["seed"])

    # get corrected dataset and model path
    output_index = config["output_idx"] if "output_idx" in config else False
    _, _, trainloader = get_dataset_loader(
        custom_args.dataset,
        ["train_clean", "train_noisy"],
        case,
        batch_size=custom_args.batch_size,
        shuffle=True,
        output_index=output_index,
        device=device
    )

    _, _, testloader = get_dataset_loader(
        custom_args.dataset,
        "test",
        None,
        batch_size=custom_args.batch_size,
        shuffle=False
    )

    num_test_images = len(testloader.dataset)

    load_model_path1 = settings.get_ckpt_path(
        custom_args.dataset,
        case,
        custom_args.model,
        model_suffix="inc_train"
    )

    load_model_path2 = settings.get_ckpt_path(
        custom_args.dataset,
        "pretrain",
        custom_args.model,
        model_suffix="pretrain"
    )

    save_model_path = settings.get_ckpt_path(
        custom_args.dataset,
        case,
        custom_args.model,
        model_suffix="restore",
        unique_name=uni_name,
    )

    history_save_path = settings.get_ckpt_path(
        custom_args.dataset, case, custom_args.model, model_suffix="history", unique_name=uni_name
    )
    # checkpoint = torch.load(load_model_path)

    loaded_model1 = load_custom_model(
        custom_args.model, num_classes, load_pretrained=False
    )
    model1 = ClassifierWrapper(loaded_model1, num_classes)

    loaded_model2 = load_custom_model(
        custom_args.model, num_classes, load_pretrained=False
    )
    model2 = ClassifierWrapper(loaded_model2, num_classes)

    checkpoint = torch.load(load_model_path1)
    model1.load_state_dict(checkpoint, strict=False)
    checkpoint = torch.load(load_model_path2)
    model2.load_state_dict(checkpoint, strict=False)

    model1.to(device)
    model2.to(device)

    # 根据配置中的算法选择对应的模型
    model = algorithms.__dict__[config["algorithm"]](model1, model2)
    model.set_optimizer(trainloader.dataset, num_classes, config)

    epoch = 0
    # evaluate models with random weights
    test_acc = get_test_acc(model.evaluate(testloader))
    print(
        "Epoch [%d/%d] Test Accuracy on the %s test images: %.4f"
        % (epoch + 1, custom_args.num_epochs, num_test_images, test_acc)
    )

    best_acc, best_epoch = 0, 0
    model_history = []

    for epoch in range(1, custom_args.num_epochs):
        # train
        model.train(trainloader, epoch)
        # evaluate
        # test_acc, test_acc2 = model.evaluate(testloader)
        eval_result = model_test(
            testloader, model1, device=device
        )
        model_history.append(eval_result)
        test_acc = eval_result["global"]

        if epoch >= 5 and best_acc < test_acc:
            best_acc, best_epoch = test_acc, epoch
            # save model1
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(model.model1.state_dict(), save_model_path)
            print("model saved to:", save_model_path)

        print(
            "Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%"
            % (epoch + 1, custom_args.num_epochs, num_test_images, test_acc)
        )

    if best_acc == 0:
        # save model
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(model.model1.state_dict(), save_model_path)
        print("model saved to:", save_model_path)

    torch.save(model_history, history_save_path)
    # if config["save_result"]:
    #     acc_np = np.array(acc_list)
        # nni.report_final_result(acc_np.mean())
        # jsonfile = get_log_name(args.config, config)
        # np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
        # save_results(config=config, last_ten=acc_np, best_acc=best_acc, best_epoch=best_epoch, jsonfile=jsonfile)
        # plot_results(epochs=config['epochs'], test_acc=acc_all_list, plotfile=jsonfile.replace('.json', '.png'))


if __name__ == "__main__":
    main()
