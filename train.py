import os
import random
import numpy as np
import torch
import argparse
import gc
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import MyModel
from utils.logger import get_logger
from utils.trainer import ModelTrainer


def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
       
    return args, device


def prepare_data(args):
    # TODO:
    train_dl = DataLoader()
    val_dl = DataLoader()
    test_dl = DataLoader()

    return train_dl, val_dl, test_dl


def prepare_model(args, device):
    # TODO:
    model = MyModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    args, device = get_parameters()

    logfile = f''
    logger = get_logger(logfile)

    train_dl, val_dl, test_dl = prepare_data()

    model, criterion, optimizer, scheduler = prepare_model()

    trainer = ModelTrainer(
        model=model, 
        name='Project',
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_gradients=True,
        early_stop=True,
        warm_up=10,
        patience=args.patience,
        min_delta=0.0001,
        device=device,
        logger=logger 
    )

    # train
    trainer.train(
        train_dl=train_dl,
        val_dl=val_dl,
        n_epochs=args.epochs,
        chpt_path='checkpoints',
        print_every=10
    )

    # test
    trainer.validate(
        val_dl=test_dl
    )