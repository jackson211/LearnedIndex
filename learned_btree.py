import os
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from NN import NN
from btree import BTree, Item

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)


class Cfg(object):

    def __init__(self, dict):
        self.__dict__.update(dict)

    def __str__(self):
        return "\n".join(f"{k}\t{v}" for k, v in self.__dict__.items())


def naive_models(path):
    print("Loading data file from: {}\n".format(str(path)))
    colnames = ["x", "y"]
    data = pd.read_csv(path, names=colnames, header=None)
    x, y = data['x'].values.reshape(-1, 1), data['y'].values.reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(x,
                                                      y,
                                                      test_size=0.1,
                                                      random_state=0)

    print("Linear Regression")
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_val, y_pred)))


def list_to_tensor(values, requires_grad=True):
    # convert a list of values to [n, 1] shape tensor
    return Variable(torch.FloatTensor(values).to(device).view(-1, 1),
                    requires_grad=requires_grad)


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


def train_epoch(train_x, train_y, epoch, batch_size, model, criterion,
                optimizer):
    model.train()
    losses = []
    for i in range(0, len(train_x) - 1, batch_size):
        optimizer.zero_grad()
        pred_y = model(train_x[i:i + batch_size])
        loss = criterion(pred_y, train_y[i:i + batch_size])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def val_epoch(val_x, val_y, epoch, batch_size, model, criterion, optimizer):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for j in range(0, len(val_x) - 1, batch_size):
            pred_val_y = model(val_x[j:j + batch_size])
            val_loss = criterion(pred_val_y, val_y[j:j + batch_size])
            val_losses.append(val_loss.item())
    return val_losses


def test(checkpoint_path):
    cfg, model, optimizer = load_checkpoint(checkpoint_path)
    model.eval()
    print(model(torch.FloatTensor([680.])))
    print(model(torch.FloatTensor([1000.])))
    print(model(torch.FloatTensor([1200.])))


def save_checkpoint(cfg, epoch, model, optimizer, loss, val_loss, model_name):
    checkpoint_stats = {
        'cfg': cfg,
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss
    }
    path = "./checkpoint"
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        path = os.path.join(path, model_name)
    torch.save(checkpoint_stats, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    cfg = checkpoint['cfg']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return cfg, model, optimizer


def max_abs_err(pred_y, true_y):
    return max(abs(pred_y - true_y))


def train(cfg, x, y, model, criterion, optimizer, val_ratio=0.1):
    train_x = list_to_tensor(x, requires_grad=True)
    train_y = list_to_tensor(y, requires_grad=False)
    train_size = train_x.size(0)
    val_size = int(train_size * val_ratio)
    idx = torch.randperm(train_size)
    val_x = train_x[idx[:val_size]]
    val_y = train_y[idx[:val_size]]

    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    best_loss = None
    best_model = None
    counter = 0

    for epoch in range(1, cfg.n_epochs + 1):
        losses = train_epoch(train_x, train_y, epoch, cfg.batch_size, model,
                             criterion, optimizer)
        train_losses.extend(losses)

        val_loss = val_epoch(val_x, val_y, epoch, cfg.batch_size, model,
                             criterion, optimizer)

        val_losses.extend(val_loss)

        train_loss = np.average(train_losses)
        val_loss = np.average(val_losses)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)

        if best_loss is None:
            best_loss = val_loss
        elif val_loss < best_loss:
            best_loss, best_model = val_loss, model
            model_name = f"{cfg.H}_{cfg.n_epochs}_{cfg.batch_size}.tar"
            counter = 0
        else:
            counter += 1
        if counter == cfg.stop_step:
            print(f'Early stop at epoch: {epoch}\tBest loss:{best_loss:3f}')
            break

        if (epoch % 10 == 0):
            print(
                f'Epoch:{epoch}/{cfg.n_epochs}\tAvg Train loss:{train_loss:3f}\tAvg Val loss:{val_loss:.3f}'
            )
        # clear losses data
        train_losses = []
        valid_losses = []
    # save_checkpoint(cfg, epoch, best_model, optimizer, train_loss, val_loss,
    #                 model_name)
    return best_model


def hybrid_training(cfg, x, y, threshold=100):
    M = len(cfg.stages)
    col_num = cfg.stages[1]
    tmp_x = [[[] for i in range(j)] for j in cfg.stages]
    tmp_y = [[[] for i in range(j)] for j in cfg.stages]
    tmp_x[0][0] = x
    tmp_y[0][0] = y
    index = [[None for i in range(j)] for j in cfg.stages]
    for i in range(M):
        for j in range(cfg.stages[i]):
            model, criterion, optimizer = build_model(cfg, level=i)
            if tmp_x[i][j]:
                index[i][j] = train(cfg, tmp_x[i][j], tmp_y[i][j], model,
                                    criterion, optimizer)
            if i < M - 1:
                for r in range(len(tmp_x[i][j])):
                    p = int(index[i][j](torch.FloatTensor([tmp_x[i][j][r]])))
                    if p > cfg.stages[i + 1] - 1:
                        p = cfg.stages[i + 1] - 1

                    tmp_x[i + 1][p].append(tmp_x[i][j][r])
                    tmp_y[i + 1][p].append(tmp_y[i][j][r])
    for j in range(len(index[M - 1])):
        if index[M - 1][j]:
            input_x = torch.FloatTensor([tmp_x[M - 1][j]]).view(-1, 1)
            true_y = torch.FloatTensor(tmp_y[M - 1][j]).view(-1, 1)
            pred_y = index[M - 1][j](input_x)
            mae = max_abs_err(pred_y, true_y).item()
            if mae > threshold:
                btree = BTree(32)
                for key, pos in zip(tmp_x[M - 1][j], tmp_y[M - 1][j]):
                    btree.insert(Item(key, pos))
                index[M - 1][j] = btree
    print(index)
    return index


def build_model(cfg, level=0):
    model = NN(cfg.n_layer[level], cfg.D_in, cfg.H, cfg.D_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    print("-" * 20 + "Model" + "-" * 20)
    print(f"Model: {model}\nCriterion: {criterion}\nOptimizer: {optimizer}\n")

    print("-" * 20 + "Training" + "-" * 20)
    return model, criterion, optimizer


def main(path, config):
    dataset = read_data(path, feature_scale=False)
    x = dataset['x']
    y = dataset['y']

    cfg = Cfg(config)
    print("-" * 20 + "Config" + "-" * 20)
    print(cfg)

    # model, criterion, optimizer = build_model(cfg, level=0)
    # train(cfg, x, y, model, criterion, optimizer)

    hybrid_training(cfg, x, y)
    # test('checkpoint/16_500_32.tar')


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
        'block_size': 100
    }
    # with open('config.yaml', 'w') as f:
    #     yaml.dump(config, f)
    main(args.path, config)
    # naive_models(args.path)