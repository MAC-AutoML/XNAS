"""Evaluate model by NAS-Bench-201"""

from xnas.core.config import cfg
import xnas.logger.logging as logging

logger = logging.get_logger(__name__)


try:
    from nas_201_api import NASBench201API as API
    api = API(cfg.BENCHMARK.NB201PATH)
except ImportError:
    print('Could not import NASBench201.')
    exit(1)


def index_to_genotype(index):
    return api.arch(index)

def evaluate(genotype, epoch=12, **kwargs):
    """Require info from NAS-Bench-201 API.
    
    Implemented following the source code of DrNAS.
    """
    result = api.query_by_arch(genotype, str(epoch))
    (
        cifar10_train,
        cifar10_test,
        cifar100_train,
        cifar100_valid,
        cifar100_test,
        imagenet16_train,
        imagenet16_valid,
        imagenet16_test,
    ) = distill(result)
    
    logger.info("Evaluate with NAS-Bench-201 (Bench epoch:{})".format(epoch))
    logger.info("cifar10 train %f test %f", cifar10_train, cifar10_test)
    logger.info("cifar100 train %f valid %f test %f", cifar100_train, cifar100_valid, cifar100_test)
    logger.info("imagenet16 train %f valid %f test %f", imagenet16_train, imagenet16_valid, imagenet16_test)
    
    if "writer" in kwargs.keys():
        writer = kwargs["writer"]
        cur_epoch = kwargs["cur_epoch"]
        writer.add_scalars("nasbench201/cifar10", {"train": cifar10_train, "test": cifar10_test}, cur_epoch)
        writer.add_scalars("nasbench201/cifar100", {"train": cifar100_train, "valid": cifar100_valid, "test": cifar100_test}, cur_epoch)
        writer.add_scalars("nasbench201/imagenet16", {"train": imagenet16_train, "valid": imagenet16_valid, "test": imagenet16_test}, cur_epoch)
    return result

# distill 201api's results
def distill(result):
    result = result.split("\n")
    cifar10 = result[5].replace(" ", "").split(":")
    cifar100 = result[7].replace(" ", "").split(":")
    imagenet16 = result[9].replace(" ", "").split(":")

    cifar10_train = float(cifar10[1].strip(",test")[-7:-2].strip("="))
    cifar10_test = float(cifar10[2][-7:-2].strip("="))
    cifar100_train = float(cifar100[1].strip(",valid")[-7:-2].strip("="))
    cifar100_valid = float(cifar100[2].strip(",test")[-7:-2].strip("="))
    cifar100_test = float(cifar100[3][-7:-2].strip("="))
    imagenet16_train = float(imagenet16[1].strip(",valid")[-7:-2].strip("="))
    imagenet16_valid = float(imagenet16[2].strip(",test")[-7:-2].strip("="))
    imagenet16_test = float(imagenet16[3][-7:-2].strip("="))

    return (
        cifar10_train,
        cifar10_test,
        cifar100_train,
        cifar100_valid,
        cifar100_test,
        imagenet16_train,
        imagenet16_valid,
        imagenet16_test,
    )