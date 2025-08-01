import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
import csv
import torch.nn.functional as F

plt.switch_backend('agg')

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
    
def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def vali(args, accelerator, model, vali_loader, mode):
    total_mae_loss = []
    total_mse_loss = []
    total_rmse_loss = []
    
    model.eval()
    val_bar = tqdm(vali_loader, desc=mode)
    with torch.no_grad():
        for i, batch_data in enumerate(val_bar):
            batch_x, batch_y = batch_data
            batch_x = batch_x.float().to(accelerator.device)
            true = batch_y.float().to(accelerator.device)
            #mean, std = mean.float().to(accelerator.device), std.float().to(accelerator.device)
            #mean, std = mean.permute(0,2,3,1,4), std.permute(0,2,3,1,4)
            if args.model == 'LLM':
                pred,_ = model(batch_x, true, scheduled_prob = 0, training = False)
                true = true[:,:,:,:,2].unsqueeze(-1)
            else:
                batch_x = batch_x[:,:,:,:,2].permute(0,3,1,2) # (tot,224,224,24,12) -> (tot,1,224,224)
                batch_x = batch_x.reshape(-1, args.seq_len, 1) 
                batch_y = batch_y[:,:,:,:,2].permute(0,3,1,2) # (tot,224,224,24,12) -> (tot,1,224,224)
                true = batch_y.reshape(-1, args.seq_len, 1) 
                pred = model(batch_x) 
                
            #pred, true = pred*std+mean, true*std+mean
            pred = pred.reshape(-1, args.seq_len, 1)
            true = true.reshape(-1, args.seq_len, 1)
            pred, true = accelerator.gather_for_metrics((pred, true))

            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            
            mae_loss = MAE(pred, true)
            mse_loss = MSE(pred, true)
            rmse_loss = RMSE(pred, true)
  
            total_mae_loss.append(mae_loss.item())
            total_mse_loss.append(mse_loss.item())
            total_rmse_loss.append(rmse_loss.item())

            val_bar.set_description(desc="Validating mae: %.2f mse: %.2f rmse: %.2f" % (mae_loss.item(), mse_loss.item(), rmse_loss.item())) 
    
    total_mae_loss = np.average(total_mae_loss)
    total_mse_loss = np.average(total_mse_loss)
    total_rmse_loss = np.average(total_rmse_loss) 
    model.train()
    return total_mae_loss, total_mse_loss, total_rmse_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(args.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(args.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(args.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(args.device)
        batch_y_mark = torch.ones(true.shape).to(args.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content