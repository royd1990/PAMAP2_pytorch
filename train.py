# from syft.frameworks.torch.dp import pate
import torch
random_seed = 1 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import scipy
# np.random.seed(random_seed)
from torch.utils.tensorboard import SummaryWriter
# from syft.core.frameworks.torch.dp import pate
import argparse
import pickle
from utils.helper import get_device_id, parse_configuration
from utils.datasets import *
from utils.losses import *
from src.agent import *
from torch.utils.data import Subset

train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")



if __name__ == "__main__":

    print("Training Student Model")
    acc_scores = []
    # Configuration Loading
    parser = argparse.ArgumentParser(description='Perform student training.')
    parser.add_argument('--configfile',type=str,default='configuration.json', help='path to the configfile')
    args = parser.parse_args()
    configuration = parse_configuration(args.configfile)

    lstm_config_params = configuration['models']['lstm_model_opportunity']
    experiment_name = configuration['experiment']
    network_config_params = lstm_config_params['network_params']
    train_config_params = lstm_config_params['training_params']
    test_config_params = lstm_config_params['testing_params']
    
    # Parsing config params and setting up respective variables

    training_config_params = configuration['training_params']
    dataset_config_params = configuration['dataset_params']

    data_folder = training_config_params["data_folder"]
    data_file = training_config_params["data_file"]
    data_path = os.path.join(data_folder, data_file)

    seq_length = dataset_config_params["seq_len"]
    overlap = dataset_config_params["overlap"]
    train_batch_size = train_config_params["batch_size"]
    test_batch_size = test_config_params["batch_size"]

    # Parsing general configuration params
    general_config_params = configuration['general_params']
    

    # Teacher Parameters
    model_directory = training_config_params["model_directory"]

    # Data Loading and pytorch dataloader creation
    data = scipy.io.loadmat(data_path)

    # train_data = {"X":data["X_train"],"y":data["y_train"]}
    # test_data = {"X":data["X_test"],"y":data["y_test"]}
    # valid_data = {"X":data["X_valid"],"y":data["y_valid"]}

    load_obj = LoadDatasets(data,seq_length,overlap,experiment_name,LoadStrategyA())
    train_data_loader = load_obj.prepare_train_data_loader(train_batch_size)

    # load_obj_test = LoadDatasets(test_data,seq_length,overlap,LoadStrategyD())
    test_data_loader = load_obj.prepare_test_data_loader(test_batch_size)
    
    # load_obj_valid = LoadDatasets(valid_data,seq_length,overlap,LoadStrategyD())
    valid_data_loader = load_obj.prepare_valid_data_loader(test_batch_size)

    if os.path.exists(model_directory):
        print("Model directory already exists")
    else:
        os.mkdir(model_directory)
    pamap2_student_agent = agent(train_config_params, network_config_params, verbose=True, model_dir=model_directory)
    acc_scores.append(pamap2_student_agent.train(train_data_loader, valid_data_loader))
    print(acc_scores[0])