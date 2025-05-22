from itertools import product
import train


logger = None  # default logger is None

def set_logger(ext_logger):
    global logger
    logger = ext_logger
    

def update_config(cfg, updates):
    """
    Update a nested dictionary using dotted keys.

    Args:
        cfg (dict): The original configuration dictionary.
        updates (dict): A dictionary of keys (using dotted notation) and their new values.
    """
    for key, value in updates.items():
        keys = key.split('.')
        d = cfg
        for k in keys[:-1]:  # Traverse to the nested dictionary
            d = d.setdefault(k, {})
        d[keys[-1]] = value  # Set the value for the final key
        
        
def main(cfg, DEVICE):
    """
    Perform a grid search over the hyperparameters specified in the config file.
    """
    # Create a list of all possible hyperparameter combinations
    keys, values = zip(*cfg['param_grid'].items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    
    # Run each experiment
    for i, exp in enumerate(experiments):
        updated_cfg = cfg.copy()
        
        for param_key, param_value in exp.items():
            update_config(updated_cfg, {param_key: param_value})        
            
        # Run the experiment
        logger.info("Running experiment {}/{}".format(i+1, len(experiments)))
        logger.info("Hyperparameters: {}".format(exp))
        train.main(updated_cfg, DEVICE)
        
    logger.info("Grid search complete.")
    