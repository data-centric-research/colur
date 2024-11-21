import copy
import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np

import arg_parser
import evaluation
import pruner
import unlearn
import utils
from trainer import validate


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    # todo add shuffle_flg
    shuffle_flg = args.shuffle

    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, seed=seed, shuffle=shuffle_flg
        )
        print("len(forget_dataset)", len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]

        retain_loader = replace_loader_dataset(
            retain_dataset, seed=seed, shuffle=shuffle_flg
        )
        print("len(retain_dataset)", len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=shuffle_flg
            )
            print("len(forget_dataset)", len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=shuffle_flg
            )
            print("len(retain_dataset)", len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=shuffle_flg
            )
            print("len(forget_dataset)", len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=shuffle_flg
            )
            print("len(retain_dataset)", len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    # todo add save dataset
    if args.save_data:
        if not args.shuffle:
            path = args.save_data_path
            save_data_dir = os.path.split(os.path.split(path)[0])[0]
            os.makedirs(save_data_dir, exist_ok=True)
            test_data_path = os.path.join(save_data_dir, "test_data.npy")
            if not os.path.exists(test_data_path):
                if args.dataset == "TinyImagenet":
                    test_data = test_loader.dataset.imgs
                    forget_data = forget_dataset.imgs
                else:
                    test_data = test_loader.dataset.data
                    forget_data = forget_dataset.data

                np.save(os.path.join(save_data_dir, "test_data.npy"), test_data)
                np.save(
                    os.path.join(save_data_dir, "test_label.npy"),
                    test_loader.dataset.targets,
                )
                np.save(os.path.join(save_data_dir, "forget_data.npy"), forget_data)
                np.save(
                    os.path.join(save_data_dir, "forget_label.npy"),
                    forget_dataset.targets,
                )

                print("save test data label and forget data label done!")
        else:
            print("set --shuffle false to save data!!!")

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    def get_accuracy():
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            # loader.shuffle = False
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc, predicts = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

            if args.save_data:
                save_path = args.save_data_path
                os.makedirs(save_path, exist_ok=True)
                if name == "test":
                    np.save(os.path.join(save_path, "test_predicts.npy"), predicts)
                    print("save test predicts done!")
                if name == "forget":
                    np.save(os.path.join(save_path, "forget_predicts.npy"), predicts)
                    print("save forget predicts done!")

        class_replace_list = args.class_to_replace.split(",")
        f_dataset = copy.deepcopy(forget_dataset)
        for class_replace_str in class_replace_list:
            class_replace = int(class_replace_str)
            c_idx = forget_dataset.targets == class_replace
            if args.dataset == "TinyImagenet":
                f_dataset.imgs = forget_dataset.imgs[c_idx]
            else:
                f_dataset.data = forget_dataset.data[c_idx]
            f_dataset.targets = forget_dataset.targets[c_idx]
            loader = torch.utils.data.DataLoader(
                f_dataset,
                batch_size=args.batch_size,
                num_workers=0,
                shuffle=False,
            )
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc, predicts = validate(loader, model, criterion, args)
            accuracy["forget " + class_replace_str] = val_acc
            print(f"forget label {class_replace_str} acc: {val_acc}")
        return accuracy

    if args.resume:
        if args.eval_result_ft:
            ckpt_file = "checkpoint_ft.pth.tar"
            checkpoint = unlearn.load_unlearn_checkpoint(model, device, args, ckpt_file)
        else:
            checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint

        # todo load finetune_backbone and train unlearn
        if args.load_ff:
            accuracy = get_accuracy()

            unlearn_method = unlearn.get_unlearn_method(args.unlearn)
            unlearn_method(unlearn_data_loaders, model, criterion, args)
            unlearn.save_unlearn_checkpoint(model, None, args)
    else:
        # checkpoint = torch.load(args.mask, map_location=device)
        # if "state_dict" in checkpoint.keys():
        #     checkpoint = checkpoint["state_dict"]
        # current_mask = pruner.extract_mask(checkpoint)
        # pruner.prune_model_custom(model, current_mask)
        # pruner.check_sparsity(model)
        #
        # if (
        #     args.unlearn != "retrain"
        #     and args.unlearn != "retrain_sam"
        #     and args.unlearn != "retrain_ls"
        # ):
        #     model.load_state_dict(checkpoint, strict=False)

        accuracy = get_accuracy()
        # evaluation_result["accuracy"] = accuracy

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = get_accuracy()
        evaluation_result["accuracy"] = accuracy
        if args.eval_result_ft:
            ft_file = "eval_result_ft.pth.tar"
            unlearn.save_unlearn_checkpoint(model, evaluation_result, args, ft_file)
        else:
            unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    # todo temp
    # if "SVC_MIA_forget_efficacy" not in evaluation_result:
    if True:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)
        shadow_test_len = test_len // 8
        shadow_test = torch.utils.data.Subset(
            test_loader.dataset, list(range(shadow_test_len))
        )
        random_retain = np.random.choice(retain_len, shadow_test_len // 2)
        random_forget = np.random.choice(
            np.arange(retain_len, retain_len + forget_len), shadow_test_len // 2
        )
        random_idx = np.concatenate([random_retain, random_forget], axis=0)
        shadow_train = torch.utils.data.Subset(
            retain_dataset + forget_dataset, random_idx
        )
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=shadow_test_loader,
            target_train=forget_loader,
            target_test=None,
            model=model,
        )
        if args.eval_result_ft:
            ft_file = "eval_result_ft.pth.tar"
            unlearn.save_unlearn_checkpoint(model, evaluation_result, args, ft_file)
        else:
            unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    """training privacy MIA:
        in distribution: retain
        out of distribution: test
        target: (retain, test)"""
    # if "SVC_MIA_training_privacy" not in evaluation_result:
    #     test_len = len(test_loader.dataset)
    #     retain_len = len(retain_dataset)
    #     num = test_len // 2
    #
    #     utils.dataset_convert_to_test(retain_dataset, args)
    #     utils.dataset_convert_to_test(forget_loader, args)
    #     utils.dataset_convert_to_test(test_loader, args)
    #
    #     shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
    #     target_train = torch.utils.data.Subset(
    #         retain_dataset, list(range(num, retain_len))
    #     )
    #     shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
    #     target_test = torch.utils.data.Subset(
    #         test_loader.dataset, list(range(num, test_len))
    #     )
    #
    #     shadow_train_loader = torch.utils.data.DataLoader(
    #         shadow_train, batch_size=args.batch_size, shuffle=False
    #     )
    #     shadow_test_loader = torch.utils.data.DataLoader(
    #         shadow_test, batch_size=args.batch_size, shuffle=False
    #     )
    #
    #     target_train_loader = torch.utils.data.DataLoader(
    #         target_train, batch_size=args.batch_size, shuffle=False
    #     )
    #     target_test_loader = torch.utils.data.DataLoader(
    #         target_test, batch_size=args.batch_size, shuffle=False
    #     )
    #
    #     evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
    #         shadow_train=shadow_train_loader,
    #         shadow_test=shadow_test_loader,
    #         target_train=target_train_loader,
    #         target_test=target_test_loader,
    #         model=model,
    #     )
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    # unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
