from pathlib import Path
import json
import yaml
from dataclasses import asdict 

from config.config_types import AppConfig
from utils.paths import VOL_EXPERIMENTS_DIR, PRICE_EXPERIMENTS_DIR, CONFIG_DIR
from utils.inference_utils import format_legend_name

EXP = VOL_EXPERIMENTS_DIR

NAMES = {
    "cnn": {
        "muon": "exp_169_cnn_100_muon_icml_3",
        "adam": "exp_037_cnn_100_adam_lr",
        "sgd" : "exp_041_cnn_100_sgd",
    },
    "lstm": {
        "muon": "exp_036_lstm_100_muon_lr",
        "adam": "exp_039_lstm_100_adam_lr",
        "sgd" : "exp_042_lstm_100_sgd",
    },
    "mlp": {
        "muon": "exp_035_mlp_100_muon_lr",
        "adam": "exp_038_mlp_100_adam_lr",
        "sgd" : "exp_043_mlp_100_sgd",
    },
}
TRIAL = 'trial_search_best'
FOLD = 'fold_000'


def create_yaml(base, intervention):
    '''
    Base will be in the form exp_....
    '''

    # -------- load config --------
    base_path = f"{EXP}/{base}/{TRIAL}/"
    conifg_path = f"{base_path}config_snapshot.json"
    
    with open(conifg_path, 'r') as f:
        cfg = json.load(f)

    cfg = cfg["cfg"]

    cfg = AppConfig.from_dict(cfg)

    # write intervention 
    cfg.trainer.hparams['initialization'] = f'{intervention}/{TRIAL}/{FOLD}'

    # Set search to false
    cfg.experiment.hyperparams_search = False

    # change name
    clean_names = [format_legend_name(n) for n in [base, intervention]]
    name = f'{clean_names[0]}_{clean_names[1]}'
    name = name.lower().replace(' ', '_')
    cfg.experiment.name = name


    # Save cfg
    cfg_dict = asdict(cfg)
    out_path = f'{CONFIG_DIR}/{name}.yaml' 
    print(name)

    with open(out_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

def main():
    for arch in NAMES.keys():
        for base in NAMES[arch].values():
            for int in NAMES[arch].values():
                if int != base:
                    create_yaml(base, int)


if __name__ == '__main__':
    main()
