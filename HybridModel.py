import os
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from NN import NN
from BTree import BTree, Item
from Cfg import Cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)


class HybridModel(object):

    def __init__(self, cfg=None, data=None):
        super(HybridModel, self).__init__()
        self.cfg = cfg
        self.data = data
        self.index = None
        self.stage_num = None
        self.tmp_x = None
        self.tmp_y = None
        if self.cfg:
            self.set_cfg(self.cfg)
        if self.data:
            self.set_data(self.data)

    def get_cfg(self):
        return self.cfg

    def set_cfg(self, cfg):
        self.cfg = cfg
        self.index = [[None for i in range(j)] for j in self.cfg.stages]
        self.stage_num = len(self.cfg.stages)
        self.tmp_x = [[[] for i in range(j)] for j in self.cfg.stages]
        self.tmp_y = [[[] for i in range(j)] for j in self.cfg.stages]

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data
        assert "x" in self.data
        assert "y" in self.data
        self.tmp_x[0][0] = self.data['x']
        self.tmp_y[0][0] = self.data['y']

    def build_model(self, level=0):
        model = NN(self.cfg.n_layer[level], self.cfg.D_in, self.cfg.H,
                   self.cfg.D_out)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.cfg.learning_rate,
                                     weight_decay=self.cfg.weight_decay)
        print("-" * 20 + "Model" + "-" * 20)
        print(
            f"Model: {model}\nCriterion: {criterion}\nOptimizer: {optimizer}\n")

        print("-" * 20 + "Training" + "-" * 20)
        return model, criterion, optimizer

    def list_to_tensor(self, nums, requires_grad=True):
        # convert a list of values to [n, 1] shape tensor
        return Variable(torch.FloatTensor(nums).to(device).view(-1, 1),
                        requires_grad=requires_grad)

    def max_abs_err(self, pred_y, true_y):
        return max(abs(pred_y - true_y))

    def create_test(self, train_x, train_y, val_ratio=0.1):
        train_size = train_x.size(0)
        val_size = int(train_size * val_ratio)
        idx = torch.randperm(train_size)
        val_x = train_x[idx[:val_size]]
        val_y = train_y[idx[:val_size]]
        return val_x, val_y

    def train_epoch(self, train_x, train_y, batch_size, model, criterion,
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

    def val_epoch(self, val_x, val_y, batch_size, model, criterion):
        model.eval()
        val_losses = []
        with torch.no_grad():
            for j in range(0, len(val_x) - 1, batch_size):
                input_x = val_x[j:j + batch_size]
                true_y = val_y[j:j + batch_size]
                pred_val_y = model(input_x)
                val_loss = criterion(pred_val_y, true_y)
                # print(torch.squeeze(torch.stack([pred_val_y, true_y], dim=1)))
                val_losses.append(val_loss.item())
        return val_losses

    def train(self, x, y, model, criterion, optimizer, val_ratio=0.1):
        train_x = self.list_to_tensor(x, requires_grad=True)
        train_y = self.list_to_tensor(y, requires_grad=False)
        val_x, val_y = self.create_test(train_x, train_y, val_ratio=val_ratio)

        train_losses = []
        val_losses = []
        avg_train_losses = []
        avg_val_losses = []
        best_loss = None
        best_model = model
        counter = 0

        for epoch in range(1, self.cfg.n_epochs + 1):
            losses = self.train_epoch(train_x, train_y, self.cfg.batch_size,
                                      model, criterion, optimizer)

            val_loss = self.val_epoch(val_x, val_y, self.cfg.batch_size, model,
                                      criterion)

            train_losses.extend(losses)
            val_losses.extend(val_loss)
            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)
            avg_train_losses.append(train_loss)
            avg_val_losses.append(val_loss)

            if best_loss is None:
                best_loss = val_loss
            elif val_loss < best_loss:
                best_loss, best_model = val_loss, model
                counter = 0
            else:
                counter += 1
            if counter == self.cfg.stop_step:
                print(f'Early stop at epoch: {epoch}\tBest loss:{best_loss:3f}')
                return best_model

            if (epoch % 10 == 0):
                print(
                    f'Epoch:{epoch}/{self.cfg.n_epochs}\tAvg Train loss:{train_loss:3f}\tAvg Val loss:{val_loss:.3f}'
                )
            # clear losses data
            train_losses = []
            valid_losses = []
        return best_model

    def hybrid_training(self, threshold=100):
        for i in range(self.stage_num):
            for j in range(self.cfg.stages[i]):
                # Build model according to different stages
                model, criterion, optimizer = self.build_model(level=i)

                # In case of empty tmp_x train data
                if self.tmp_x[i][j]:
                    self.index[i][j] = self.train(self.tmp_x[i][j],
                                                  self.tmp_y[i][j],
                                                  model,
                                                  criterion,
                                                  optimizer,
                                                  val_ratio=0.1)

                # Update training example for next level model
                if i < self.stage_num - 1:
                    for r in range(len(self.tmp_x[i][j])):
                        p = self.index[i][j](torch.FloatTensor(
                            [self.tmp_x[i][j][r]]))
                        if self.cfg.feature_scale == True:
                            p = int(p * self.cfg.stages[i + 1])
                        else:
                            p = int(p)
                        if p > self.cfg.stages[i + 1] - 1:
                            p = self.cfg.stages[i + 1] - 1

                        self.tmp_x[i + 1][p].append(self.tmp_x[i][j][r])
                        self.tmp_y[i + 1][p].append(self.tmp_y[i][j][r])

        # If MAE(Maximum Absolute Error) of a ML model is greater than threshold, then we replace it with a BTree structure
        last = self.stage_num - 1
        for j in range(len(self.index[last])):
            if self.index[last][j]:
                input_x = torch.FloatTensor([self.tmp_x[last][j]]).view(-1, 1)
                true_y = torch.FloatTensor(self.tmp_y[last][j]).view(-1, 1)
                pred_y = self.index[last][j](input_x)
                mae = self.max_abs_err(pred_y, true_y).item()
                if mae > threshold:
                    btree = BTree(32)
                    for key, pos in zip(self.tmp_x[last][j],
                                        self.tmp_y[last][j]):
                        btree.insert(Item(key, pos))
                    self.index[last][j] = btree
        self.save_index()

    def search(self, key):
        if not torch.is_tensor(key):
            key = torch.tensor([key], dtype=torch.float32)
        search_result = None

        p = self.index[0][0](key)
        for i in range(1, self.stage_num):
            if i < self.stage_num:
                model_index = int(p * self.cfg.stages[i])
                next_model = self.index[i][model_index]
                if isinstance(next_model, NN):
                    p = int(next_model(key) * self.cfg.stages[i])
                elif isinstance(next_model, BTree):
                    p = int(next_model.search(key) * self.cfg.stages[i])
            search_result = p

        return search_result

    def save_index(self):
        index_stats = {'cfg': self.cfg, 'index': self.index, 'weights': []}

        # Save weights for each model
        for stage in self.index:
            for model in stage:
                if isinstance(model, NN):
                    index_stats['weights'].append(model.state_dict())

        model_name = f"stage[{self.cfg.stages[0]}_{self.cfg.stages[1]}]_{self.cfg.H}_{self.cfg.n_epochs}_{self.cfg.batch_size}.pkl"
        path = "./checkpoint"
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            path = os.path.join(path, model_name)
        with open(path, 'wb') as out_file:
            pickle.dump(index_stats, out_file)
            print(f'{path} saved')

    def load_index(self, path):
        with open(path, 'rb') as in_file:
            print(f'Loading {path}')
            index_stats = pickle.load(in_file)
        self.cfg = index_stats['cfg']
        self.set_cfg(self.cfg)
        self.index = index_stats['index']

        # Load weights for each model
        for stage in self.index:
            for model in stage:
                if isinstance(model, NN):
                    model.load_state_dict(index_stats['weights'].pop(0))