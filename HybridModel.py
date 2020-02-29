import os
import numpy as np
import torch
from torch.autograd import Variable
from NN import NN
from BTree import BTree, Item
from Cfg import Cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)


class HybridModel(object):

    def __init__(self, cfg, data):
        super(HybridModel, self).__init__()
        self.cfg = cfg
        self.data = data
        assert "x" in self.data
        assert "y" in self.data
        self.total_x = self.data['x']
        self.total_y = self.data['y']

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
        best_model = None
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
                model_name = f"{self.cfg.H}_{self.cfg.n_epochs}_{self.cfg.batch_size}.tar"
                counter = 0
            else:
                counter += 1
            if counter == self.cfg.stop_step:
                print(f'Early stop at epoch: {epoch}\tBest loss:{best_loss:3f}')
                break

            if (epoch % 10 == 0):
                print(
                    f'Epoch:{epoch}/{self.cfg.n_epochs}\tAvg Train loss:{train_loss:3f}\tAvg Val loss:{val_loss:.3f}'
                )
            # clear losses data
            train_losses = []
            valid_losses = []
        # self.save_checkpoint(epoch, best_model, optimizer, train_loss,
        #  val_loss, model_name)
        return best_model

    def hybrid_training(self, threshold=100):
        M = len(self.cfg.stages)
        tmp_x = [[[] for i in range(j)] for j in self.cfg.stages]
        tmp_y = [[[] for i in range(j)] for j in self.cfg.stages]
        tmp_x[0][0] = self.total_x
        tmp_y[0][0] = self.total_y
        index = [[None for i in range(j)] for j in self.cfg.stages]

        for i in range(M):
            for j in range(self.cfg.stages[i]):
                # Build model according to different stages
                model, criterion, optimizer = self.build_model(level=i)

                # In case of empty tmp_x train data
                if tmp_x[i][j]:
                    index[i][j] = self.train(tmp_x[i][j],
                                             tmp_y[i][j],
                                             model,
                                             criterion,
                                             optimizer,
                                             val_ratio=0.1)

                # Update training example for next level model
                if i < M - 1:
                    for r in range(len(tmp_x[i][j])):
                        p = int(index[i][j](torch.FloatTensor([tmp_x[i][j][r]
                                                              ])))
                        if self.cfg.feature_scale == True:
                            p *= self.cfg.stages[i + 1]
                        if p > self.cfg.stages[i + 1] - 1:
                            p = self.cfg.stages[i + 1] - 1

                        tmp_x[i + 1][p].append(tmp_x[i][j][r])
                        tmp_y[i + 1][p].append(tmp_y[i][j][r])

        # If MAE(Maximum Absolute Error) of a ML model is greater than threshold, then we replace it with a BTree structure
        for j in range(len(index[M - 1])):
            if index[M - 1][j]:
                input_x = torch.FloatTensor([tmp_x[M - 1][j]]).view(-1, 1)
                true_y = torch.FloatTensor(tmp_y[M - 1][j]).view(-1, 1)
                pred_y = index[M - 1][j](input_x)
                mae = self.max_abs_err(pred_y, true_y).item()
                if mae > threshold:
                    btree = BTree(32)
                    for key, pos in zip(tmp_x[M - 1][j], tmp_y[M - 1][j]):
                        btree.insert(Item(key, pos))
                    index[M - 1][j] = btree
        print(index)
        return index

    def save_checkpoint(self, epoch, model, optimizer, loss, val_loss,
                        model_name):
        checkpoint_stats = {
            'cfg': self.cfg,
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

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        cfg = checkpoint['cfg']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return cfg, model, optimizer
