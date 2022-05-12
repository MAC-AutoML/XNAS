import os

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter
from xbbo.search_algorithm.regularizedEA_optimizer import RegularizedEA
from xbbo.utils.constants import MAXINT

import xnas.search_space.DropNAS.utils as utils
from xnas.core.builders import build_space, build_loss_fun
from xnas.core.config import cfg
from xnas.datasets.loader import construct_loader


def evaluate_single_path(val_loader, model, criterion, config_dict):
    choice = []
    for _key in range(20):
        key = 'x{}'.format(_key)
        # print(key)
        choice.append(config_dict[key])
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    return val_loss.avg


def _build_space(rng):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    for i in range(20):
        # exec(f"x{i}=UniformIntegerHyperparameter('x{i}', 0, 3, default_value=1)")
        x = (UniformIntegerHyperparameter('x{}'.format(i), 0, 3, default_value=1))
        # for i in range(20, 40):
        #         # exec(f"x{i}=UniformIntegerHyperparameter('x{i}', 0, 3, default_value=1)")
        #     x = (UniformIntegerHyperparameter('x{}'.format(i), 0, 5, default_value=1))
        cs.add_hyperparameter(x)
    # con = LessThanCondition(x1, x0, 1.)
    # cs.add_condition(con)
    return cs


if __name__ == "__main__":

    net = build_space()
    best_supernet_weights = os.path.join(cfg.OUT_DIR, '%s' % 'best.pth')
    checkpoint = torch.load(best_supernet_weights)
    net.load_state_dict(checkpoint, strict=True)
    loss_fun = build_loss_fun().cuda()
    # Dataset Definition
    [train_, val_] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE, cfg.SEARCH.DATAPATH)

    MAX_CALL = 200
    rng = np.random.RandomState(42)
    # define black box function
    # blackbox_func = lambda x : evaluate_single_path(val_, net, loss_fun,x)
    blackbox_func = lambda x: evaluate_single_path(val_, net, loss_fun, x)

    # define search space
    cs = _build_space(rng)
    # define black box optimizer
    hpopt = RegularizedEA(space=cs, seed=rng.randint(MAXINT), llambda=50, sample_size=20)
    # Example call of the black-box function
    # def_value = blackbox_func()
    # def_value = cs.get_default_configuration()['x1']
    # print("Default Value: %.2f" % def_value)
    # ---- Begin BO-loop ----
    for i in range(cfg.RE.MAXCALL):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate 
        value = blackbox_func(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(observe_value=value)
        hpopt.observe(trial_list=trial_list)

        print(value)

    # plt.plot(hpopt.trials.get_history()[0])
    # plt.savefig('./out/rosenbrock_bo_gp.png')
    # plt.show()
    print('find best value:{}'.format(hpopt.trials.get_best()))
