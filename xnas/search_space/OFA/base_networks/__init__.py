from .proxyless_nets import *
from .mobilenet_v3 import *
from .resnets import *

def get_net_by_name(name):
    if name == ProxylessNASNet.__name__:
        return ProxylessNASNet
    elif name == MobileNetV3.__name__:
        return MobileNetV3
    elif name == ResNet.__name__:
        return ResNet
    else:
        raise ValueError("unrecognized type of network: %s" % name)
