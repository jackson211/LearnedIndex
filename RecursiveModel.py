import torch
from NN import NN


class RecursiveModel():

    def __init__(self, cfg, train_x, train_y, val_x, val_y, threshold=1):
        self.cfg = cfg
        self.stages = cfg.stages
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.threshold = threshold
        self.index = [[None for i in range(j)] for j in cfg.stages]

    def first_stage(self):
        model = NN(cfg.n_layer, cfg.D_in, cfg.H, cfg.D_out)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)