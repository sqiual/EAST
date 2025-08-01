import argparse
import torch
import csv
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from llms import get_registed_model
from openai import OpenAI
import time
import random
import numpy as np
import os
from torch.utils.data import Subset
from scipy.spatial import KDTree
from load_data_marine import MyDataset
from utils.tools_prof import adjust_learning_rate, vali, load_content

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#from models import TimeLLM_redo, lstm #, Autoformer, DLinear
from models import TimeLLM_prof#, Autoformer, DLinear
#from utils.tools_ceshi_local import adjust_learning_rate, vali, load_content


parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--data', type=str, default='all_nyctaxi', help='marine, nycbike')
parser.add_argument('--best_model', type=str, default='all_taxi', help='prefix when saving test results')
parser.add_argument('--seq_len', type=int, default=6, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=6, help='prediction sequence length')


parser.add_argument('--is_training', type=int, default=0, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default="LLM",
                    help='model name, options: [Autoformer, DLinear, LLM, LSTM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader

parser.add_argument('--flag', type=str, default='inflow', help='inflow, outflow')
parser.add_argument('--data_short', type=str, default='supervised', help='zeroshot, supervised')
parser.add_argument('--root_path', type=str, default='/root/Time-LLM_OURS/OURS/dataset/NYC/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--img_size', type=int, default=[8,8], help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument("--llm_model", type=str, default="GPT2", help="Model name, GPT2")
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--source_num', type=int, default='200', help='number of source')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=20, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.000005, help='optimizer learning rate, 0.00001')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--new_token', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
args.content = load_content(args)
print('flag: ', args.flag, 'data_short: ', args.data_short)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])#, deepspeed_plugin=deepspeed_plugin)

test_set = MyDataset(args, 'test', 1)
test_loader = DataLoader(test_set , batch_size=args.batch_size, shuffle=False)

if args.model == "LLM":
    model = TimeLLM_prof.Model(args).float().to(accelerator.device)
elif args.model == "LSTM":
    model = lstm.Model(args).float().to(accelerator.device)
    
model_path = os.path.join(args.checkpoints, '{}_'.format(args.model) + args.best_model+'.pt')
print('best model path: ', model_path)
model.load_state_dict(torch.load(model_path, map_location=accelerator.device))


test_mae_loss_in, test_mse_loss_in, test_rmse_loss_in = vali(args, accelerator, model, test_loader, 'test')

accelerator.print(
            "Dataset Tested: {0} {1} \nTest MAE_in: {2:.2f} \nTest MSE_in: {3:.2f} \nTest RMSE_in: {4:.2f}".format(args.data, args.flag, test_mae_loss_in, test_mse_loss_in, test_rmse_loss_in, ))