from datetime import datetime
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import ParameterGrid


def generate_unique_name(config):
    # generate unique name based on the config
    run_name = config['model']+"_"+config['dataset'] + \
        "_"+"seed_"+str(config['random_seed'])+"_"
    run_name += datetime.now().strftime("%Y%m%d_%H_%M_%S")
    return run_name


def read_optional(config, arg, default):
    if arg in config:
        return config[arg]
    return default


def add_attribute(config: DictConfig, name, val):
    with open_dict(config):
        config[name] = val


def add_all_attributes(cfg, cfg2):
    # add all attributes from cfg2 to cfg
    for key in cfg2:
        add_attribute(cfg, key, cfg2[key])


def build_subdictionary(hyperparameters, loop_hyperparameters):
    """
    Given dictionary of hyperparameters (where some of the values may be lists) and a list of keys
    loop_hyperparameters, build a ParameterGrid

    """
    # build sub dictionary of hyperparameters
    subparameters = dict(
        (k, hyperparameters[k]) for k in loop_hyperparameters if k in hyperparameters)
    subparameters = dict((k, [subparameters[k]]) if not isinstance(
        subparameters[k], list) else (k, subparameters[k]) for k in subparameters)

    subparameters = ParameterGrid(subparameters)

    return subparameters
