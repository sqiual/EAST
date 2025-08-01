import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from models import TimeLLM_prof#, lstm#, Autoformer, DLinear
from llms import get_registed_model
from openai import OpenAI
import time
import random
import numpy as np
import os
from torch.utils.data import Subset
from scipy.spatial import KDTree
from load_data_marine import MyDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from utils.tools_prof import adjust_learning_rate, vali, load_content


os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
print(f"cuda:{torch.cuda.current_device()}")


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Time-LLM')

# basic config
parser.add_argument('--best_model', type=str, default='marinecb', help='prefix when saving test results')
parser.add_argument('--data', type=str, default='marinecb', help='nyctaxi, marine, marinecb, nycbike, trafficnj')

parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')

parser.add_argument('--model', type=str, default='LLM',help='model name, options: [Autoformer, DLinear, LLM, LSTM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
# data loader
parser.add_argument('--data_short', type=str, default='supervised', help='zeroshot, supervised')
parser.add_argument('--flag', type=str, default='inflow', help='inflow, outflow')
parser.add_argument('--root_path', type=str, default='./dataset/NYC/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--upsample_size', type=int, default=224, help='224')

# forecasting task
parser.add_argument('--img_size', type=int, default=[8,8], help='prediction sequence length')
# model define
parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument("--llm_model", type=str, default="GPT2", help="Model name, GPT2")
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.000005, help='optimizer learning rate, 0.000001')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--llm_layers', type=int, default=6)
#parser.add_argument('--device', type=str, default=accelerator.device) 
args = parser.parse_args()

def get_scheduled_prob(step, total_steps, strategy="linear"):
    if strategy == "linear":
        return max(0.0, 1.0 - step / total_steps)
    elif strategy == "inverse_sigmoid":
        import math
        k = 5.0  # 控制衰减速率
        return k / (k + math.exp(step / k))
    else:
        return 1.0  # 默认全用 teacher forcing
        
for ii in range(args.itr):
    args.device = accelerator.device
    train_set = MyDataset(args, 'train', 1)
    #val_set = MyDataset(args, 'val', 1)
    test_set = MyDataset(args, 'test', 0.1)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    #vali_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set , batch_size=args.batch_size, shuffle=False)

        
    
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float().to(accelerator.device)
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float().to(accelerator.device)
    elif args.model == 'LSTM':
        model = lstm.Model(args).float().to(accelerator.device)
    else:
        model = TimeLLM_prof.Model(args).float().to(accelerator.device)

    # unique checkpoint saving path
    args.content = load_content(args)
    # if not os.path.exists(path) and accelerator.is_local_main_process:
    #     os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
            
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    # if args.model == 'LLM':
    #     model_optim = torch.optim.AdamW(
    #         [
    #             {"params": model.llm_model.parameters(), "lr": 1e-5},
    #             {"params": model.x.parameters(), "lr": 1e-4},
    #             {"params": model.x2.parameters(), "lr": 1e-4},  
    #         ],
    #         weight_decay=0.01 
    #     )
    # else:
    #     model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)


    criterion2 = nn.MSELoss()

    # train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
    #     train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    #print(model)
    if args.model == 'LLM':
        print("The number of trainable parameters: {}".format(model.param_num()))
    print("The learning rate: {}".format(args.learning_rate))
    val_best_loss = float('inf')
    val_loss_lst = []
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        train_loss_pred = []
        model.train()
        epoch_time = time.time()
        print('Model', args.model)
        print('len(train_loader)', len(train_loader))
        train_bar = tqdm(train_loader, desc="Training")
        scheduled_prob = get_scheduled_prob(epoch, args.train_epochs, strategy="linear")
        for i, batch_data in enumerate(train_bar):
            iter_count += 1
            model_optim.zero_grad()
            batch_x, batch_y = batch_data
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            #mean, std = mean.float().to(accelerator.device), std.float().to(accelerator.device)
            #mean, std = mean.permute(0,2,3,1,4), std.permute(0,2,3,1,4)
            if args.model == 'LLM':
                output, c_loss = model(batch_x, batch_y, scheduled_prob, training = True) 
                batch_y = batch_y[:,:,:,:,2].unsqueeze(-1) #[B,H,W,T]  
            else:
                batch_x = batch_x[:,:,:,:,2].permute(0,3,1,2) # (tot,224,224,24,12) -> (tot,1,224,224)
                batch_x = batch_x.reshape(-1, args.seq_len, 1) 
                batch_y = batch_y[:,:,:,:,2].permute(0,3,1,2) # (tot,224,224,24,12) -> (tot,1,224,224)
                batch_y = batch_y.reshape(-1, args.seq_len, 1) 
                output = model(batch_x)
            
            #output, batch_y = output*std+mean, batch_y*std+mean
            output = output.reshape(-1, args.seq_len, 1)
            batch_y = batch_y.reshape(-1, args.seq_len, 1)
            loss_pred = criterion2(output, batch_y)
            train_loss_pred.append(loss_pred.item())
            
            denominator = loss_pred + c_loss
            loss_pred_weight = loss_pred / denominator
            c_loss_weight = c_loss / denominator
            alpha = 0.95
            total_loss = alpha * loss_pred_weight * loss_pred + (1-alpha) * c_loss_weight * c_loss

            accelerator.backward(total_loss)
            model_optim.step()
            train_bar.set_description(desc='[%d/%d] Training pred loss: %.3f Cont loss: %.3f' % (epoch, args.train_epochs, loss_pred.item(), c_loss.item()))

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss_pred = np.average(train_loss_pred)
        print('train_loss_pred', train_loss_pred)
        print('len(test_loader)', len(test_loader))
        test_mae_loss, test_mse_loss, test_rmse_loss = vali(args, accelerator, model, test_loader, 'test')
        vali_mae_loss = test_mae_loss

        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.2f} Vali Loss: {2:.2f} \nTest MAE: {3:.2f} \nTest MSE: {4:.2f} \nTest RMSE: {5:.2f}".format(
                epoch + 1, train_loss_pred, vali_mae_loss, test_mae_loss, test_mse_loss, test_rmse_loss))
        
        val_loss_lst.append(round(vali_mae_loss, 2))
        print('vali loss progress: ', val_loss_lst)
        batch_val_loss = vali_mae_loss
        if batch_val_loss < val_best_loss:
            val_best_loss = batch_val_loss
            is_best = True
        else:
            is_best = False
        best_model_epoch = os.path.join(args.checkpoints, '{}_'.format(args.model) + args.best_model+ '_epoch_'+ str(epoch)+'.pt')
        best_model = os.path.join(args.checkpoints, '{}_'.format(args.model) + args.best_model+'.pt')
        if is_best:
            model = accelerator.unwrap_model(model)
            torch.save(model.state_dict(), best_model_epoch)
            torch.save(model.state_dict(), best_model)
            print("Best model saved to ", best_model)