import pandas as pd
import yaml
import argparse
import torch
from Cfg import Cfg
from HybridModel import HybridModel


def feature_scaling(data):
    max_value, min_value = data.max(), data.min()
    diff = max_value - min_value
    data = data.apply(lambda x: (x - min_value) / diff)
    return data


def read_data(path, feature_scale=False):
    print("Loading data file from: {}".format(str(path)))
    colnames = ["x", "y"]
    data = pd.read_csv(path, names=colnames, header=None)
    val_data = data.sample(frac=0.1, random_state=1)

    if feature_scale:
        data['y'] = feature_scaling(data['y'])
        val_data['y'] = feature_scaling(val_data['y'])

    x = data['x'].values.tolist()
    y = data['y'].values.tolist()
    return {'x': x, 'y': y}


def plot(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.plot(x, y, label="true pos")
    plt.plot(x, pred_y, label="predicted pos")
    plt.legend(loc="upper left")
    plt.show()


def main(path, config):
    cfg = Cfg(config)
    print("-" * 20 + "config" + "-" * 20)
    print(cfg)

    dataset = read_data(path, feature_scale=cfg.feature_scale)

    hm = HybridModel(cfg=cfg, data=dataset)
    hm.hybrid_training()

    hm = HybridModel()
    hm.load_index("checkpoint/stage[1_10]_32_500_32.pkl")
    print(hm.search(1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True, help="Data file path")
    parser.add_argument('--config',
                        '-c',
                        required=False,
                        help="Config file path")
    args = parser.parse_args()

    args.config = {
        'n_layer': [2, 1],
        'D_in': 1,
        'H': 32,
        'D_out': 1,
        'n_epochs': 500,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'stop_step': 10,
        'stages': [1, 10],
        'total_number': 1000,
        'block_size': 100,
        'feature_scale': True
    }
    # with open('config.yaml', 'w') as f:
    #     yaml.dump(config, f)
    main(args.path, args.config)