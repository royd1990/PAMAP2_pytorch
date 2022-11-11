
import torch
from torch.utils.tensorboard import SummaryWriter
import utils.ts_processor as tsp
import argparse
import pickle
from utils.helper import get_device_id, parse_configuration
from utils.datasets import *
from utils.losses import *
from src.agent import *
import scipy

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

    experiment_name = configuration['experiment']
    lstm_config_params = configuration['models']['lstm_model_opportunity']

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
    model_name = general_config_params["inference_model"]

    # Teacher Parameters
    model_directory = training_config_params["model_directory"]
    model_path = os.path.join(model_directory, model_name)

    data = scipy.io.loadmat(data_path)

    tsp_obj = tsp.ts_processor(seq_length, overlap)

    X_train = data["trainingData"].T
    X_test = data["testingData"].T
    y_test = data["testingLabels"].reshape(-1)-1

    train_sd = np.std(X_train,axis=0)
    train_mean = np.mean(X_train,axis=0)
    X_test = normalize_standardize(X_test,train_mean,train_sd)

    X_test_processed,y_test= tsp_obj.process_standard_ts(X_test,y_test)


    # Training agent creation and training
    opportunity_inference_agent = agent(train_config_params, network_config_params, verbose=True, model_dir=model_directory)
    
    # Model Loading
    opportunity_inference_agent.load_model(model_path)
    predictions = opportunity_inference_agent.predict(torch.from_numpy(np.float32(X_test_processed)).cuda())


    acc = calc_accuracy(predictions.data,torch.from_numpy(y_test).cuda())
    f1_macro = f1_score(predictions.data,torch.from_numpy(y_test).cuda())

    print(acc,f1_macro)
