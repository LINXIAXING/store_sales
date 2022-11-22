import easydict
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from src.pretreatment.dataload import StoreDataset, DataloaderType, get_dataset
from src.train.train import train
from src.train.prediction import Prediction
from src.model.LSTM import LstmRNN
from src.model.MLP import MLP
from src.model.Transformer import TransAm

if __name__ == '__main__':
    config_path = "./src/config.yml"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loading config file
    config = EasyDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))

    # model = LstmRNN(input_size=5, hidden_size=20, output_size=1, num_layers=10).double().to(device)
    # model = Transformer(feature_size=tl * 5)
    model = TransAm(feature_size=5).to(device)
    # model = MLP(5, 10, 1)

    # loading datasets

    train_dataset = StoreDataset(config=config, data_type=DataloaderType.train)
    train_dataset_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=False)

    val_dataset = StoreDataset(config=config, data_type=DataloaderType.validate)
    val_dataset_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

    model = train(config=config,
                  model=model,
                  dataloader=train_dataset_loader,
                  val_dataloader=val_dataset_loader,
                  path_to_save_model=config.save.model_path,
                  path_to_save_loss=config.save.loss_path,
                  path_to_save_predictions=config.save.predictions_path,
                  device=device)

    # test_dataset = StoreDataset(config=config, data_type=DataloaderType.prediction)
    # test_dataset_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)
    #
    # pred = Prediction(config=config,
    #                   model_name='best_train_1.pth',
    #                   model=model,
    #                   dataloader=test_dataset_loader)
    # data = pred.get_infer()
    # print(data.shape)
    # data.to_csv(config.save.predictions_path + 'prediction.csv', index=False)

"""
datasets = get_dataset(config=config, data_type=DataloaderType.train)
    model = train(config=config,
                  model=model,
                  datasets=datasets,
                  path_to_save_model=config.save.model_path,
                  path_to_save_loss=config.save.loss_path,
                  path_to_save_predictions=config.save.predictions_path,
                  device=device)
"""