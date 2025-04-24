import torch
import numpy as np
import logging
from tqdm import tqdm

class ModelTrainer(object):
    def __init__(self, 
                 model: torch.Module,
                 name: str=None, 
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 clip_gradients: bool=False, 
                 early_stop: bool=True, 
                 warm_up: int=10,
                 patience: int=5, 
                 min_delta: float=1e-4, 
                 device: torch.device=None,
                 logger: logging.Logger= None
    ):
        self.model = model
        self.name = name
        self.early_stop = early_stop
        self.warm_up = warm_up
        self.patience = patience
        self.min_delta = min_delta
        self.early_stop_cnt = 0
        self.min_val_loss = float('inf')
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_gradients = clip_gradients # control whether to clip the gradients during the training process

        self.best_model = None
        self.best_model_epoch = None

        self.train_losses = []
        self.batch_loss = []
        self.val_losses = []

        self.device = device
        self.logger = logger

    def _reset_early_stopping(self):
        self.early_stop_cnt = 0
        self.min_val_loss = float('inf')

    def _early_stopping(self, val_loss, epoch):
        if val_loss < self.min_val_loss:
            self.early_stop_cnt = 0
            self.min_val_loss = val_loss
            self.best_model = self.model.state_dict()
            self.best_model_epoch = epoch
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.early_stop_cnt += 1
            if self.early_stop_cnt >= self.patience:
                return True
        return False

    def _train_epoch(self, loader):
        self.model.train()
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()

            # TODO: forward pass
            outputs = self.model(xb)
            loss = self.criterion(outputs, yb)
            loss.backward()
            self.optimizer.step()

            if self.clip_gradients:
                # torch.utils.clip_grad_norm_ 函数的作用是对所有参数的梯度进行归一化，使其 L2 范数（即欧几里得范数）不超过 max_norm 指定的值（这里是 1）。
                # norm_type=2 表示使用 L2 范数进行裁剪。
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
            
            # tracking metrics
            epoch_loss += loss.item() * xb.size(0)
        
        return epoch_loss / len(loader.dataset)
    
    def _eval_epoch(self, loader):
        self.model.eval()
        val_loss = 0
        predictions = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                # TODO: forward pass
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                # TODO: format predictions
                predictions.append(outputs)

                val_loss += loss.item() * xb.size(0)
                
        return val_loss / len(loader.dataset), predictions
    
    def train(self, train_dl, val_dl, n_epochs, chpt_path='checkpoints', print_every=10):
        if self.early_stop:
            self._reset_early_stopping()
        self.train_losses = []
        self.val_losses = []
        
        for e in range(n_epochs):
            tqdm_bar = tqdm(train_dl)
            tqdm_bar.set_description(f"[Train | {e+1:03d}/{n_epochs:03d}]")
            train_loss = self._train_epoch(tqdm_bar)
            if self.scheduler is not None:
                self.scheduler.step()
            val_loss, _ = self._eval_epoch(val_dl)

            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

            if (e == 0) or ((e+1) % print_every == 0):
                if self.logger:
                    self.logger.info(f'Epoch {e+1:03d}/{n_epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            if self.early_stop and (e > self.warm_up) and self._early_stopping(val_loss, e):
                if self.logger:
                    self.logger.info(f'Early stopping at epoch {e}')
                    self.logger.info(f'Epoch {e+1:03d}/{n_epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
                break
            
        # save the best model even if early stopping is not triggered
        if self.early_stop:
            torch.save(self.best_model, f'{chpt_path}/{self.name}_{self.best_model_epoch}_{self.min_val_loss:.4f}.pth')
            if self.logger:
                self.logger.info(f'Best model saved at epoch {self.best_model_epoch} with val_loss: {self.min_val_loss:.4f}')

        # TODO: when training is done, load the best model_state_dict to the current model?

    def sanity_check(self, batch, n_epochs=10):
        print('Sanity check using one batch of data')
        self.model.train()
        losses = []
        for e in range(n_epochs):
            self.optimizer.zero_grad()
            xb, yb = batch
            xb, yb = xb.to(self.device), yb.to(self.device)
            # TODO: forward pass
            outputs = self.model(xb)
            loss = self.criterion(outputs, yb)
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {e+1:03d}/{n_epochs:03d} | Loss: {loss.item():.4f}')
            losses.append(loss.item())
        return losses

    def validate(self, val_dl):
        val_loss, preds = self._eval_epoch(val_dl)
        return val_loss, preds
    
    def restore_best_model(self):
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)