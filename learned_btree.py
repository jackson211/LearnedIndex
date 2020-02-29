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


def main(path, config):
    cfg = Cfg(config)
    print("-" * 20 + "config" + "-" * 20)
    print(cfg)

    dataset = read_data(path, feature_scale=cfg.feature_scale)

    # model, criterion, optimizer = build_model(cfg, level=0)
    # best_model = train(cfg, x, y, model, criterion, optimizer)
    # cfg, model, optimizer = load_checkpoint('checkpoint/16_500_32.tar')
    # train_x = list_to_tensor(x, requires_grad=true)
    # train_y = list_to_tensor(y, requires_grad=false)
    # val_x, val_y = create_test(train_x, train_y, val_ratio=0.1)
    # val_losses = val_epoch(val_x, val_y, cfg.batch_size, model, max_abs_err)
    # print(val_losses)
    # print(np.average(val_losses))
    # test(val_x, val_y, best_model)

    hm = hybridmodel(cfg, dataset)
    hm.hybrid_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True, help="Data file path")
    args = parser.parse_args()

    config = {
        'n_layer': [3, 1],
        'D_in': 1,
        'H': 16,
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
    main(args.path, config)