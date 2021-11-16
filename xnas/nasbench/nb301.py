import os
from collections import namedtuple

import nasbench301 as nb
from torch import ge, normal

download_dir = './301model'
version = '0.9'


def init_model(version=0.9, download_dir="./301model"):

    # Note: Uses 0.9 as the default models, switch to 1.0 to use 1.0 models
    models_0_9_dir = os.path.join(download_dir, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name: os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(download_dir, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name: os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

    # If the models are not available at the paths, automatically download
    # the models
    # Note: If you would like to provide your own model locations, comment this out
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=version, delete_zip=True,
                           download_dir=download_dir)

    # Load the performance surrogate model
    # NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
    # NOTE: Defaults to using the default model download path
    ensemble_dir_performance = model_paths['xgb']
    performance_model = nb.load_ensemble(ensemble_dir_performance)

    # Load the runtime surrogate model
    # NOTE: Defaults to using the default model download path
    ensemble_dir_runtime = model_paths['lgb_runtime']
    runtime_model = nb.load_ensemble(ensemble_dir_runtime)

    return performance_model, runtime_model


def Eval_nasbench301(theta, search_space, logger):
    """
    Evaluate with nasbench301, space=DARTS/nasbench301

    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    genotype = Genotype(
            normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
            normal_concat=[2, 3, 4, 5],
            reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
            reduce_concat=[2, 3, 4, 5]
            )
    """
    performance_model, runtime_model = init_model(version, download_dir)
    genotype = search_space.genotype(theta)

    # reformat the output of DartsCNN.genotype()
    genotype = reformat_DARTS(genotype)

    prediction_genotype = performance_model.predict(
        config=genotype, representation="genotype", with_noise=True)
    runtime_genotype = runtime_model.predict(
        config=genotype, representation="genotype")

    logger.info("Genotype architecture performance: %f, runtime %f" %
                (prediction_genotype, runtime_genotype))

    """
    Codes below are used when sampling from a ConfigSpace.
    Rewrite it when using.
    """
    # configspace_path = os.path.join(download_dir, 'configspace.json')
    # with open(configspace_path, "r") as f:
    #     json_string = f.read()
    #     configspace = cs_json.read(json_string)
    # configspace_config = configspace.sample_configuration()
    # prediction_configspace = performance_model.predict(config=configspace_config, representation="configspace", with_noise=True)
    # runtime_configspace = runtime_model.predict(config=configspace_config, representation="configspace")
    # print("Configspace architecture performance: %f, runtime %f" %(prediction_configspace, runtime_configspace))

def reformat_DARTS(genotype):
    """
    format genotype for DARTS-like
    from:
        Genotype(normal=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 0)], [('sep_conv_3x3', 2), ('max_pool_3x3', 1)], [('sep_conv_3x3', 3), ('dil_conv_3x3', 2)], [('dil_conv_5x5', 4), ('dil_conv_5x5', 3)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1)], [('max_pool_3x3', 0), ('dil_conv_5x5', 2)], [('max_pool_3x3', 0), ('sep_conv_5x5', 1)], [('dil_conv_5x5', 4), ('max_pool_3x3', 0)]], reduce_concat=range(2, 6))
    to:
        Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
    """
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    _normal = []
    _reduce = []
    for i in genotype.normal:
        for j in i:
            _normal.append(j)
    for i in genotype.reduce:
        for j in i:
            _reduce.append(j)
    _normal_concat = [i for i in genotype.normal_concat]
    _reduce_concat = [i for i in genotype.reduce_concat]
    r_genotype = Genotype(
        normal=_normal,
        normal_concat=_normal_concat,
        reduce=_reduce,
        reduce_concat=_reduce_concat
    )
    return r_genotype
