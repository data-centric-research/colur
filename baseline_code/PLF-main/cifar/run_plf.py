import logging
import os
import shutil

import torch
import torch.optim as optim
from torchvision import models

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import norm
import plf

from conf import cfg, load_cfg_fom_args
from core_model.dataset import get_dataset_loader
from args_paser import parse_args
from configs import settings
from core_model.custom_model import load_custom_model, ClassifierWrapper


logger = logging.getLogger(__name__)


def evaluate(description):
    # load_cfg_fom_args(description)

    # dora modify model resnet18
    # configure model
    # base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
    #                    cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

    custom_args = parse_args()
    case = settings.get_case(custom_args.noise_ratio, custom_args.noise_type, custom_args.balanced)
    step = getattr(custom_args, "step", 1)
    uni_name = getattr(custom_args, "uni_name", None)
    num_classes = settings.num_classes_dict[custom_args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get corrected dataset and model path
    test_data, test_labels, testloader = get_dataset_loader(
        custom_args.dataset, "test", case, None, None, None, custom_args.batch_size, shuffle=False
    )

    load_model_path = settings.get_ckpt_path(custom_args.dataset, case, custom_args.model,
                                             model_suffix="worker_restore",
                                             step=0, unique_name=uni_name)

    if not os.path.exists(load_model_path):
        contra_model_path = settings.get_ckpt_path(custom_args.dataset, case, custom_args.model,
                                                   model_suffix="worker_restore",
                                                   step=0, unique_name="contra")
        os.makedirs(os.path.dirname(load_model_path), exist_ok=True)
        shutil.copy(contra_model_path, load_model_path)
        print('copy contra model: %s to : %s' % (contra_model_path, load_model_path))

    save_model_path = settings.get_ckpt_path(custom_args.dataset, case, custom_args.model,
                                             model_suffix="worker_tta",
                                             step=step, unique_name=uni_name)

    loaded_model = load_custom_model(custom_args.model, num_classes, load_pretrained=False)
    base_model = ClassifierWrapper(loaded_model, num_classes)
    checkpoint = torch.load(load_model_path)
    base_model.load_state_dict(checkpoint, strict=False)
    base_model.to(device)

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "PLF":
        logger.info("test-time adaptation: PLF")
        model = setup_plf(base_model, custom_args, num_classes)
    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    # for severity in cfg.CORRUPTION.SEVERITY:
    #     for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
    # continual adaptation for all corruption
    # if i_c == 0:
    try:
        model.reset()
        logger.info("resetting model")
    except:
        logger.warning("not resetting model")
        # else:
        #     logger.warning("not resetting model")
        # x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
        #                                severity, cfg.DATA_DIR, False,
        #                                [corruption_type])

    # dora modify load Dts
    # test_data, test_labels, test_dataloader = get_dataset_loader(
    #     dataset, "test", data_dir, mean, std, cfg.TEST.BATCH_SIZE, shuffle=False
    # )
    x_test = torch.from_numpy(test_data)
    y_test = torch.from_numpy(test_labels)

    x_test, y_test = x_test.to(device), y_test.to(device)
    acc = accuracy(model, x_test, y_test, custom_args.batch_size, save_path=save_model_path)
    err = 1.0 - acc
    logger.info(f"error % ]: {err:.2%}")


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(
        model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC
    )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_plf(model, custom_args, num_classes):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = plf.configure_model(model)
    params, param_names = plf.collect_params(model)
    optimizer = setup_optimizer(params)
    plf_model = plf.PLF(
        model,
        optimizer,
        custom_args,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        mt_alpha=cfg.OPTIM.MT,
        rst_m=cfg.OPTIM.RST,
        ap=cfg.OPTIM.AP,
        num_classes=num_classes,
    )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return plf_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == "Adam":
        return optim.Adam(
            params,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD,
        )
    elif cfg.OPTIM.METHOD == "SGD":
        return optim.SGD(
            params,
            lr=cfg.OPTIM.LR,
            momentum=cfg.OPTIM.MOMENTUM,
            dampening=cfg.OPTIM.DAMPENING,
            weight_decay=cfg.OPTIM.WD,
            nesterov=cfg.OPTIM.NESTEROV,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    evaluate('"CIFAR-10-C evaluation.')
