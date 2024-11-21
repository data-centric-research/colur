from .cifar_10.vgg import *
from .cifar_10.dpn import *
from .cifar_10.lenet import *
from .cifar_10.senet import *
from .cifar_10.pnasnet import *
from .cifar_10.densenet import *
from .cifar_10.googlenet import *
from .cifar_10.shufflenet import *
from .cifar_10.shufflenetv2 import *
from .cifar_10.resnet import *
from .cifar_10.resnext import *
from .cifar_10.preact_resnet import *
from .cifar_10.mobilenet import *
from .cifar_10.mobilenetv2 import *
from .cifar_10.efficientnet import *
from .cifar_10.regnet import *
from .cifar_10.dla_simple import *
from .cifar_10.dla import *

from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
}
