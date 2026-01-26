from pathlib import Path
import json
import yaml
from dataclasses import asdict 

from config.config_types import AppConfig
from utils.paths import VOL_EXPERIMENTS_DIR, PRICE_EXPERIMENTS_DIR, CONFIG_DIR
from utils.inference_utils import format_legend_name

EXP = VOL_EXPERIMENTS_DIR

SEEDS = [31102003, 26021999, 31031963, 21061965]

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
    "transformer": {
        "muon": "exp_180_transformer_100_muon",
        "adam": "exp_179_transformer_100_adam_lr",
        "sgd" : "exp_181_transformer_100_sgd"
    }
}

TRIAL = 'trial_search_best'
FOLD = 'fold_000'


def create_yaml(base, seed, idx):
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
    cfg.trainer.hparams['initialization'] = None

    # Set search to false
    cfg.experiment.hyperparams_search = False

    # Set seed
    cfg.experiment.random_state = seed

    # change name
    name = format_legend_name(base).lower().replace(' lr', '').replace(' icml', '').replace(' ', '_')
    cfg.experiment.name = f'{name}_seeds'


    # Save cfg
    cfg_dict = asdict(cfg)
    second = f"seed_runs/{name}_seeds_{idx}.yaml"
    out_path = f'{CONFIG_DIR}/{second}' 
    print(f'{second}')

    with open(out_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

def main():
    for arch in NAMES.keys():
        for base in NAMES[arch].values():
                for i, s in enumerate(SEEDS):
                    create_yaml(base, s, i)


if __name__ == '__main__':
    main()
