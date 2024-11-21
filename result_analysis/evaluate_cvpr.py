import argparse

import torch
import numpy as np
from args_paser import parse_args
from configs import settings
from core_model.custom_model import ClassifierWrapper, load_custom_model
from core_model.dataset import get_dataset_loader
from core_model.train_test import model_forward


def execute(args):
    case = settings.get_case(args.noise_ratio, args.noise_type)
    uni_names = args.uni_name
    uni_names = [uni_names] if uni_names is None else uni_names.split(",")
    num_classes = settings.num_classes_dict[args.dataset]

    loaded_model = load_custom_model(args.model, num_classes, load_pretrained=False)
    model = ClassifierWrapper(loaded_model, num_classes)

    _, _, test_loader = get_dataset_loader(
        args.dataset,
        "test",
        None,
        batch_size=args.batch_size,
        shuffle=False,
    )
    _, _, noisy_loader = get_dataset_loader(
        args.dataset,
        "train_noisy",
        case,
        batch_size=args.batch_size,
        shuffle=False,
        label_name="train_noisy_true_label"
    )

    for uni_name in uni_names:
        print(f"Evaluating {uni_name}:")
        model_ckpt_path = settings.get_ckpt_path(
            args.dataset,
            case,
            args.model,
            model_suffix=args.model_suffix,
            unique_name=uni_name,
        )
        print(f"Loading model from {model_ckpt_path}")
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Evaluating test_data:")
        results, embedding = model_test(test_loader, model)
        # print("Results: %.4f" % results)
        print("Results: ", results)
        print(f"Evaluating train_noisy_data:")
        n_results, n_embedding = model_test(noisy_loader, model)
        # print("Results: %.4f" % results)
        print("Results: ", n_results)



def model_test(data_loader, model, device="cuda"):
    eval_results = {}

    predicts, probs, embedding, labels = model_forward(
        data_loader, model, device, output_embedding=True, output_targets=True
    )

    # global acc
    global_acc = np.mean(predicts == labels)
    eval_results["global"] = global_acc.item()

    # class acc
    label_list = sorted(list(set(labels)))
    for label in label_list:
        cls_index = labels == label
        class_acc = np.mean(predicts[cls_index] == labels[cls_index])
        eval_results["label_" + str(label.item())] = class_acc.item()

    return eval_results, embedding


if __name__ == "__main__":
    try:
        pargs = parse_args()
        execute(pargs)
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
