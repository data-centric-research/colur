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
        custom_args.noise_ratio, custom_args.noise_type, custom_args.balanced
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
    }

    if uni_name not in config_modules:
        raise ValueError(f"Unknown uni_name: {uni_name}")

    # 动态导入配置模块
    config_module = importlib.import_module(config_modules[uni_name])
    config = config_module.config

    set_seed(config["seed"])

    # 根据配置中的算法选择对应的模型
    if config["algorithm"] == "Coteachingplus":
        model = algorithms.Coteachingplus(
            config,
            input_channel=config["input_channel"],
            num_classes=num_classes,
        )
        train_mode = "train"
    elif config["algorithm"] == "Coteaching":
        model = algorithms.Coteaching(
            config,
            input_channel=config["input_channel"],
            num_classes=num_classes,
        )
        train_mode = "train_single"
    elif config["algorithm"] == "JoCoR":
        model = algorithms.JoCoR(
            config,
            input_channel=config["input_channel"],
            num_classes=num_classes,
        )
        train_mode = "train_single"
    else:
        model = algorithms.__dict__[config["algorithm"]](
            config,
            input_channel=config["input_channel"],
            num_classes=num_classes,
        )
        train_mode = "train_single"

    # config = None

    # if uni_name == "Coteachingplus":
    #     from co_configs.coteachingplus import config
    # elif uni_name == "Coteaching":
    #     from co_configs.coteaching import config
    # elif uni_name == "JoCoR":
    #     from co_configs.jocor import config

    # set_seed(config["seed"])

    # if config["algorithm"] == "colearning":
    #     model = algorithms.Colearning(
    #         config,
    #         input_channel=config["input_channel"],
    #         num_classes=num_classes,
    #     )
    #     train_mode = "train"
    # else:
    #     model = algorithms.__dict__[config["algorithm"]](
    #         config,
    #         input_channel=config["input_channel"],
    #         num_classes=num_classes,
    #     )
    #     train_mode = "train_single"

    # dataloaders = cifar_dataloader(
    #     cifar_type=config["dataset"],
    #     root=config["root"],
    #     batch_size=config["batch_size"],
    #     num_workers=config["num_workers"],
    #     noise_type=config["noise_type"],
    #     percent=config["percent"],
    # )

    # dora modify dataloader and load model
    # trainloader, testloader = dataloaders.run(mode=train_mode), dataloaders.run(mode='test')

    # get corrected dataset and model path
    _, _, trainloader = get_dataset_loader(
        custom_args.dataset,
        "train",
        case,
        step,
        None,
        None,
        custom_args.batch_size,
        shuffle=True,
    )

    _, _, forgetloader = get_dataset_loader(
        custom_args.dataset,
        "train",
        case,
        step,
        None,
        None,
        custom_args.batch_size,
        shuffle=True,
    )

    _, _, testloader = get_dataset_loader(
        custom_args.dataset,
        "test",
        case,
        None,
        None,
        None,
        custom_args.batch_size,
        shuffle=False,
    )

    num_test_images = len(testloader.dataset)

    load_model_path = settings.get_ckpt_path(
        custom_args.dataset,
        case,
        custom_args.model,
        model_suffix="worker_raw",
        step=step,
        unique_name=uni_name,
    )
    # step=1, copy contra/step_0/ -> target/step_0
    # if step == 1 and not os.path.exists(load_model_path):
    #     contra_model_path = settings.get_ckpt_path(
    #         custom_args.dataset,
    #         case,
    #         custom_args.model,
    #         model_suffix="worker_restore",
    #         step=step - 1,
    #         unique_name="contra",
    #     )
    #     os.makedirs(os.path.dirname(load_model_path), exist_ok=True)
    #     shutil.copy(contra_model_path, load_model_path)
    #     print("copy contra model: %s to : %s" % (contra_model_path, load_model_path))

    save_model_path = settings.get_ckpt_path(
        custom_args.dataset,
        case,
        custom_args.model,
        model_suffix="worker_restore",
        step=step,
        unique_name=uni_name,
    )
    # checkpoint = torch.load(load_model_path)

    model.epochs = custom_args.num_epochs

    loaded_model1 = load_custom_model(
        custom_args.model, num_classes, load_pretrained=False
    )
    model.model1 = ClassifierWrapper(loaded_model1, num_classes)

    loaded_model2 = load_custom_model(
        custom_args.model, num_classes, load_pretrained=False
    )
    model.model2 = ClassifierWrapper(loaded_model2, num_classes)

    checkpoint = torch.load(load_model_path)
    model.model1.load_state_dict(checkpoint, strict=False)
    model.model2.load_state_dict(checkpoint, strict=False)

    model.model1.to(device)
    model.model2.to(device)
    # model.model1.load_state_dict(checkpoint, strict=False)
    # model.model2.load_state_dict(checkpoint, strict=False)

    epoch = 0
    # evaluate models with random weights
    test_acc = get_test_acc(model.evaluate(testloader))
    print(
        "Epoch [%d/%d] Test Accuracy on the %s test images: %.4f"
        % (epoch + 1, custom_args.num_epochs, num_test_images, test_acc)
    )

    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, custom_args.num_epochs):
        # train
        model.train(trainloader, epoch)
        # evaluate
        test_acc, test_acc2 = model.evaluate(testloader)
        # nni.report_intermediate_result(test_acc)
        if best_acc < test_acc:
            best_acc, best_epoch = test_acc, epoch

        print(
            "Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%"
            % (epoch + 1, custom_args.num_epochs, num_test_images, test_acc)
        )

        if epoch >= custom_args.num_epochs - 10:
            acc_list.extend([test_acc])
        acc_all_list.extend([test_acc])

        # save model1
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(model.model1.state_dict(), save_model_path)
        print("model saved to:", save_model_path)

    if config["save_result"]:
        acc_np = np.array(acc_list)
        # nni.report_final_result(acc_np.mean())
        # jsonfile = get_log_name(args.config, config)
        # np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
        # save_results(config=config, last_ten=acc_np, best_acc=best_acc, best_epoch=best_epoch, jsonfile=jsonfile)
        # plot_results(epochs=config['epochs'], test_acc=acc_all_list, plotfile=jsonfile.replace('.json', '.png'))


if __name__ == "__main__":
    main()
