# Contains functions for all metrics used in Table 1
import torch
from torcheval.metrics import PeakSignalNoiseRatio

# RMSE
def RMSE(y_true, y_pred):
    # y_true = torch.from_numpy(y_true)
    # y_pred = torch.from_numpy(y_pred)
    mse = torch.mean((y_true - y_pred)**2)
    rmse = torch.sqrt(mse)
    return rmse.item()

# PSNR
def PSNR(y_true, y_pred):
    # y_true = torch.from_numpy(y_true)
    # y_pred = torch.from_numpy(y_pred)
    metric = PeakSignalNoiseRatio()
    metric.update(y_pred, y_true)
    psnr = metric.compute()
    return psnr.item()