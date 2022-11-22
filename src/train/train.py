"""
@PROJECT: StoreSales - train.py
@IDE: PyCharm
@DATE: 2022/11/11 下午3:43
@AUTHOR: lxx
"""

import logging
import math, random

import evaluate
import torch
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from src.model.LSTM import LstmRNN
from src.model.Transformer import TransAm
from src.model.model import Transformer
from src.pretreatment.datamachine import scale
from src.utils.plot import *
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return True if random.random() < p else False


def train_epoch(config: EasyDict,
                model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device,
                loss_function,
                step: int):
    train_loss = 0
    model.train()
    train_bar = tqdm(dataloader)
    train_bar.set_description(f'Epoch [{step + 1}/{config.train.epoch}] Training')
    # train_bar.set_description(f'Epoch [{step + 1}/{length}] Training')
    for _input, target in train_bar:
        sampled_src = _input.permute(1, 0, 2).float().to(device)  # [window_size, batch_size, feature]
        sampled_target = target.permute(1, 0, 2).float().to(device)  # [window_size, batch_size, feature]

        prediction = model(sampled_src)[-1:, :, :]  # model->[window_size, batch_size, 1]

        # sampled_target = sampled_target.to(torch.device('cpu'))
        # prediction = prediction.to(torch.device('cpu'))
        # for s in range(len(sampled_target)):
        #     sampled_target[s] = scale(sampled_target[s].detach().numpy())
        #     prediction[s] = scale(prediction[s].detach().numpy())
        loss = loss_function(sampled_target, prediction)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.detach().item()
        train_bar.set_postfix({'loss': f'{loss:1.5f}'})
    return train_loss / len(train_bar)


def val_epoch(config: EasyDict,
              model: nn.Module,
              dataloader: DataLoader,
              device,
              step: int):
    model.eval()
    metrics = evaluate.load("mse")
    val_bar = tqdm(dataloader)
    acc = float(0)
    val_bar.set_description(f'Epoch [{step + 1}/{config.train.epoch}] Validation')
    for _input, target in val_bar:
        with torch.no_grad():
            src_input = _input.permute(1, 0, 2).double().to(device)
            prediction = model(src_input)[-1, :, :]  # model->[window_size, batch_size, 1]
            # for i in range(len(prediction)):
            #     length = len(prediction[i])
            #     p = prediction[i].reshape([length])
            #     t = target.permute(1, 0, 2).double().to(device)[i].reshape([length])
            p = prediction.reshape([len(prediction)])
            t = target.target.permute(1, 0, 2).double().to(device)[-1, :, :].reshape([len(target)])
            metrics.add_batch(predictions=p, references=t)
    acc = metrics.compute()['mse']
    return acc


def train(config: EasyDict,
          model: nn.Module,
          dataloader,
          val_dataloader,
          path_to_save_model,
          path_to_save_loss,
          path_to_save_predictions,
          device):
    device = torch.device(device)
    epoch = config.train.epoch

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    best_model = ""
    optimizer.param_groups[0]['lr'] = 0.001
    min_train_loss = float('inf')
    min_acc = float('inf')
    for step in range(epoch + 1):
        train_loss = train_epoch(config=config,
                                 model=model,
                                 dataloader=dataloader,
                                 optimizer=optimizer,
                                 device=device,
                                 loss_function=criterion,
                                 step=step)
        print(f"Mean loss: {train_loss:.5f}")
        metrics = val_epoch(config=config,
                            model=model,
                            dataloader=val_dataloader,
                            device=device,
                            step=step)
        print(f"MSE: {100 * metrics:.5f}%")
        # train_loss < min_train_loss
        if metrics < min_acc:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{step}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{step}.pth")
            min_acc = metrics
            best_model = f"best_train_{epoch}.pth"

        if epoch % 5 == 0:
            for p in optimizer.param_groups:  # 更新每个group里的参数lr
                p['lr'] *= 0.9
        # if epoch % 10 == 0:  # Plot 1-Step Predictions
        #     print(f"Epoch: {epoch}, Training loss: {train_loss}")
        #     logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
        #     scaler = load('scalar_item.joblib')
        #     sampled_src_humidity = scaler.inverse_transform(sampled_src[:, :, 0].cpu())  # torch.Size([35, 1, 7])
        #     src_humidity = scaler.inverse_transform(src[:, :, 0].cpu())  # torch.Size([35, 1, 7])
        #     target_humidity = scaler.inverse_transform(target[:, :, 0].cpu())  # torch.Size([35, 1, 7])
        #     prediction_humidity = scaler.inverse_transform(
        #         prediction[:, :, 0].detach().cpu().numpy())  # torch.Size([35, 1, 7])
        #     plot_training_3(epoch, path_to_save_predictions, src_humidity, sampled_src_humidity, prediction_humidity,
        #                     store_number)
        #
        # train_loss /= len(dataloader)
        # log_loss(train_loss, path_to_save_loss, train=True)
    plot_loss(path_to_save_loss, train=True)
    return best_model




"""
def train(config: EasyDict,
          model: nn.Module,
          dataloader,
          val_dataloader,
          path_to_save_model,
          path_to_save_loss,
          path_to_save_predictions,
          device):
    device = torch.device(device)
    epoch = config.train.epoch

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    best_model = ""
    optimizer.param_groups[0]['lr'] = 0.001
    min_train_loss = float('inf')
    min_acc = float('inf')
    for step in range(epoch + 1):
        train_loss = train_epoch(config=config,
                                 model=model,
                                 dataloader=dataloader,
                                 optimizer=optimizer,
                                 device=device,
                                 loss_function=criterion,
                                 step=step)
        print(f"Mean loss: {train_loss:.5f}")
        metrics = val_epoch(config=config,
                            model=model,
                            dataloader=val_dataloader,
                            device=device,
                            step=step)
        print(f"MSE: {100 * metrics:.5f}%")
        # train_loss < min_train_loss
        if metrics < min_acc:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{step}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{step}.pth")
            min_acc = metrics
            best_model = f"best_train_{epoch}.pth"

        if epoch % 5 == 0:
            for p in optimizer.param_groups:  # 更新每个group里的参数lr
                p['lr'] *= 0.9
        # if epoch % 10 == 0:  # Plot 1-Step Predictions
        #     print(f"Epoch: {epoch}, Training loss: {train_loss}")
        #     logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
        #     scaler = load('scalar_item.joblib')
        #     sampled_src_humidity = scaler.inverse_transform(sampled_src[:, :, 0].cpu())  # torch.Size([35, 1, 7])
        #     src_humidity = scaler.inverse_transform(src[:, :, 0].cpu())  # torch.Size([35, 1, 7])
        #     target_humidity = scaler.inverse_transform(target[:, :, 0].cpu())  # torch.Size([35, 1, 7])
        #     prediction_humidity = scaler.inverse_transform(
        #         prediction[:, :, 0].detach().cpu().numpy())  # torch.Size([35, 1, 7])
        #     plot_training_3(epoch, path_to_save_predictions, src_humidity, sampled_src_humidity, prediction_humidity,
        #                     store_number)
        #
        # train_loss /= len(dataloader)
        # log_loss(train_loss, path_to_save_loss, train=True)
    plot_loss(path_to_save_loss, train=True)
    return best_model

"""


