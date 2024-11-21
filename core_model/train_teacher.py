import os
import shutil

from core import train_teacher_model
from torch import nn
from configs import settings

from args_paser import parse_args, parse_kwargs
from lip_teacher import SimpleLipNet
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from core_model.dataset import get_dataset_loader
import torch
from train_test_utils import train_model

if __name__ == "__main__":
    args = parse_args()

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    model_suffix = "teacher_restore" if args.model_suffix is None else args.model_suffix
    step = getattr(args, "step", 0)

    model_name = args.model
    noise_ratio = args.noise_ratio
    noise_type = args.noise_type
    balanced = args.balanced
    dataset = args.dataset
    spec_norm = not args.no_spnorm
    uni_name = args.uni_name

    case = "pretrain"
    model_p0_path = settings.get_ckpt_path(
        dataset, case, model_name, model_suffix, step=0
    )

    case = settings.get_case(noise_ratio, noise_type, balanced)
    save_model_path = settings.get_ckpt_path(
        dataset,
        case,
        model_name,
        model_suffix,
        step=step,
        unique_name=uni_name,
    )

    if uni_name is not None and step == 0:
        if os.path.exists(model_p0_path):
            subdir = os.path.dirname(save_model_path)
            os.makedirs(subdir, exist_ok=True)
            shutil.copy(model_p0_path, save_model_path)
            print(f"Copy {model_p0_path} to {save_model_path}")
        else:
            raise FileNotFoundError(model_p0_path)
    else:
        num_classes = settings.num_classes_dict[dataset]
        backbone = load_custom_model(model_name, num_classes, load_pretrained=True)
        lip_teacher_model = ClassifierWrapper(
            backbone, num_classes, spectral_norm=spec_norm
        )
        if step == 0:
            save_model_path = model_p0_path
        # else:
        #     if os.path.exists(save_model_path):
        #         checkpoint = torch.load(model_p0_path)
        #         lip_teacher_model.load_state_dict(checkpoint, strict=False)

        # 根据用户选择的优化器初始化
        teacher_opt, teacher_lr_scheduler = create_optimizer_scheduler(
            optimizer_type=optimizer_type,
            parameters=lip_teacher_model.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=num_epochs,
        )
        teacher_criterion = nn.CrossEntropyLoss()

        train_teacher_model(
            args,
            step,
            num_classes,
            lip_teacher_model,
            teacher_opt,
            teacher_lr_scheduler,
            teacher_criterion,
            save_model_path,
            test_per_it=1,
        )
