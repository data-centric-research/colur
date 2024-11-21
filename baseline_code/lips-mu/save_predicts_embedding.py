import numpy as np
import os
import torch
from torchvision import models
from torch import nn

import arg_parser
from nets.train import get_loader, lip_test, SimpleLipNet
from nets.ft_unlearn import test
from models.VGG_LTH import vgg16_bn_lth


def embedding_test(test_loader, model):
    model.eval()
    outputs = []

    for i, (image, target) in enumerate(test_loader):
        image = image.cuda()

        embedding = model(image)
        embedding = embedding.data.cpu().numpy()
        embedding = np.squeeze(embedding)
        outputs.append(embedding)

    return np.concatenate(outputs, axis=0)


def save_predicts_embeddings(
    output_root,
    data_name,
    backbone_name,
    model_names,
    ckpt_name,
    num_classes,
    embed_file_name,
    predict_file_name,
    forget_loader,
    forget_label,
    test_loader,
    um_or_uram="um",
):

    data_path = backbone_name + "_" + data_name
    output_path = os.path.join(output_root, data_path)

    for model_name in model_names:
        if backbone_name == "vgg16" and model_name == "FF":
            continue

        save_dir = os.path.join(output_path, model_name)
        model_true_name = model_name
        if model_name == "FF":
            model_true_name = "fisher"
        elif model_name == "IU":
            model_true_name = "wfisher"

        last_layers_end = -1

        if um_or_uram == "um":
            # load um model
            if backbone_name == "resnet18":
                unlearn_model = models.resnet18(
                    pretrained=False, num_classes=num_classes
                )
                # if data_name == "fmnist":
                #     unlearn_model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

            elif backbone_name == "vgg16":
                unlearn_model = vgg16_bn_lth(num_classes=num_classes)
                last_layers_end = -2

            model_path = os.path.join(save_dir, model_true_name + ckpt_name)
            if not os.path.exists(model_path):
                print("um model path not found! Please check the path :", model_path)
                continue

            checkpoint = torch.load(model_path)
            unlearn_model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("loaded um model from", model_path)

            # um df predicts
            unlearn_model.cuda()
            forget_predicts = test(forget_loader, unlearn_model)
            forget_acc_unlearn = np.mean(forget_predicts == forget_label)
            print("*************************check**********************************")
            print("forget_acc: %.2f" % (forget_acc_unlearn * 100))
            # save df predicts
            predict_save_file = model_true_name + predict_file_name
            forget_predicts_save_path = os.path.join(save_dir, predict_save_file)
            np.save(forget_predicts_save_path, forget_predicts)
            print("saved um forget predicts to ", forget_predicts_save_path)

            # extractor embedding(df and dp)
            um_feat_extractor = nn.Sequential(
                *list(unlearn_model.children())[:last_layers_end]
            )
            um_feat_extractor.cuda()
            um_forget_embedding = embedding_test(forget_loader, um_feat_extractor)
            um_test_embedding = embedding_test(test_loader, um_feat_extractor)

            # save df embedding
            forget_save_file = model_true_name + "_forget" + embed_file_name
            forget_embedding_save_path = os.path.join(save_dir, forget_save_file)
            np.save(forget_embedding_save_path, um_forget_embedding)
            print("saved um forget embedding to ", forget_embedding_save_path)

            # save dp embedding
            test_save_file = model_true_name + "_test" + embed_file_name
            test_embedding_save_path = os.path.join(save_dir, test_save_file)
            np.save(test_embedding_save_path, um_test_embedding)
            print("saved um test embedding to ", test_embedding_save_path)

        elif um_or_uram == "uram":
            # uram after
            # load lipnet
            resnet = models.resnet18(pretrained=False, num_classes=512)
            resnet = nn.Sequential(*list(resnet.children())[:-1])
            lip_model = SimpleLipNet(resnet, 512, num_classes, [512])
            lip_model.cuda()

            lip_ckpt_name = "lipnet_" + ckpt_name
            model_path = os.path.join(save_dir, lip_ckpt_name)
            if not os.path.exists(model_path):
                print("uram model path not found! Please check the path :", model_path)
                continue

            checkpoint = torch.load(model_path)
            lip_model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("loaded uram model from", model_path)

            # lip forget and test predicts
            lip_test_embeddings, lip_test_pred = lip_test(test_loader, lip_model)
            lip_forget_embeddings, lip_forget_pred = lip_test(forget_loader, lip_model)

            forget_acc_lip = np.mean(lip_forget_pred == forget_label)
            print("*************************check**********************************")
            print("lip forget_acc: %.2f" % (forget_acc_lip * 100))

            # save df predicts
            predict_save_name = "lipnet" + predict_file_name
            forget_predicts_save_path = os.path.join(save_dir, predict_save_name)
            np.save(forget_predicts_save_path, lip_forget_pred)
            print("saved lip forget predicts to ", forget_predicts_save_path)

            # save df embedding
            forget_save_file = "lipnet_forget" + embed_file_name
            forget_embedding_save_path = os.path.join(save_dir, forget_save_file)
            np.save(forget_embedding_save_path, lip_forget_embeddings)
            print("saved lip forget embedding to ", forget_embedding_save_path)

            # save dp embedding
            test_save_file = "lipnet_test" + embed_file_name
            test_embedding_save_path = os.path.join(save_dir, test_save_file)
            np.save(test_embedding_save_path, lip_test_embeddings)
            print("saved uram test embedding to ", test_embedding_save_path)


def save_predicts_embeddings_uram_before(
    lip_output_path,
    data_name,
    ckpt_name,
    num_classes,
    embed_file_name,
    predict_file_name,
    forget_loader,
    forget_label,
    test_loader,
):
    output_path = os.path.join(lip_output_path, "lipnet", "resnet18", data_name)

    # load lipnet
    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    lip_model = SimpleLipNet(resnet, 512, num_classes, [512])
    lip_model.cuda()

    model_path = os.path.join(output_path, ckpt_name)
    if not os.path.exists(model_path):
        print("uram model path not found! Please check the path :", model_path)

    else:
        checkpoint = torch.load(model_path)
        lip_model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("loaded uram model from", model_path)

        # lip forget and test predicts
        lip_test_embeddings, lip_test_pred = lip_test(test_loader, lip_model)
        lip_forget_embeddings, lip_forget_pred = lip_test(forget_loader, lip_model)

        forget_acc_lip = np.mean(lip_forget_pred == forget_label)
        print("*************************check**********************************")
        print("lip forget_acc: %.2f" % (forget_acc_lip * 100))

        # save df predicts
        predict_save_name = "lipnet" + predict_file_name
        forget_predicts_save_path = os.path.join(output_path, predict_save_name)
        np.save(forget_predicts_save_path, lip_forget_pred)
        print("saved lip forget predicts to ", forget_predicts_save_path)

        # save df embedding
        forget_save_file = "lipnet_forget" + embed_file_name
        forget_embedding_save_path = os.path.join(output_path, forget_save_file)
        np.save(forget_embedding_save_path, lip_forget_embeddings)
        print("saved lip forget embedding to ", forget_embedding_save_path)

        # save dp embedding
        test_save_file = "lipnet_test" + embed_file_name
        test_embedding_save_path = os.path.join(output_path, test_save_file)
        np.save(test_embedding_save_path, lip_test_embeddings)
        print("saved uram test embedding to ", test_embedding_save_path)


if __name__ == "__main__":
    args = arg_parser.parse_args()

    ouput_root = args.save_dir
    backbones = ["resnet18", "vgg16"]
    # data_list = ['cifar10', 'cifar100', 'tinyimg', 'flowers102']
    # num_classes_list = [10, 100, 200, 102]

    data_list = ["cifar10"]
    num_classes_list = [10]
    unlearn_model_names = ["retrain", "FT", "FF", "GA", "IU", "FT_prune"]

    ckpt_name = "checkpoint_ft.pth.tar"
    embedding_file_name = "_embedding_ft.npy"
    pred_file_name = "_forget_predicts_ft.npy"
    if args.load_before_or_after == "before":
        ckpt_name = "checkpoint.pth.tar"
        embedding_file_name = "_embedding.npy"
        pred_file_name = "_forget_predicts.npy"

    forget_embed_results, test_embed_results = {}, {}

    for i, data in enumerate(data_list):
        num_classes = num_classes_list[i]

        # load data
        forget_dataset = data  # for forget data path
        dataset = data  # for create CustomDataset
        if data == "flowers102":
            forget_dataset = "flowers102_data"
        elif data == "tinyimg":
            forget_dataset = "tinyimgnet"
            dataset = "TinyImagenet"

        forget_test_data_path = os.path.join(args.save_data_path, forget_dataset)
        forget_label_path = os.path.join(forget_test_data_path, "forget_label.npy")
        forget_label = np.load(forget_label_path)

        forget_loader = get_loader(
            "forget", forget_test_data_path, args.batch_size, dataset
        )
        test_loader = get_loader(
            "test", forget_test_data_path, args.batch_size, dataset
        )

        # uram before
        if args.load_um_or_uram == "uram" and args.load_before_or_after == "before":
            save_predicts_embeddings_uram_before(
                ouput_root,
                data,
                ckpt_name,
                num_classes,
                embedding_file_name,
                pred_file_name,
                forget_loader,
                forget_label,
                test_loader,
            )
        else:
            for backbone in backbones:
                save_predicts_embeddings(
                    ouput_root,
                    data,
                    backbone,
                    unlearn_model_names,
                    ckpt_name,
                    num_classes,
                    embedding_file_name,
                    pred_file_name,
                    forget_loader,
                    forget_label,
                    test_loader,
                    args.load_um_or_uram,
                )

    print("save embedding and predict files done!")
