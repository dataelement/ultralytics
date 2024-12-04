from pathlib import Path

import yaml

data_cfg = Path(__file__).parents[2] / 'ultralytics/cfg/datasets'
cfg = data_cfg / 'table_det.yaml'

with open(cfg, 'r') as f:
    cdf_info = yaml.load(f, Loader=yaml.FullLoader)
