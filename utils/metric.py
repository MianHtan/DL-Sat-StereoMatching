import numpy as np
import torch
from utils.stereo_datasets import fetch_dataset

def epe(disp_pred:np.ndarray, gt:np.ndarray, valid:np.ndarray):
    disp_error = np.abs(gt - disp_pred)
    epe = disp_error[valid].sum()/valid.sum()
    return epe

def epe_tensor(disp_pred:torch.tensor, gt:torch.tensor, valid:torch.tensor):
    disp_error = torch.abs(gt - disp_pred)
    epe = disp_error[valid].sum()/valid.sum()
    return epe.cpu()

def px_error(disp_pred:np.ndarray, gt:np.ndarray, valid:np.ndarray, px:int):
    disp_error = np.abs(gt - disp_pred)
    disp_error[~valid] = np.nan
    px_error = np.zeros_like(disp_error)
    px_error[np.where(disp_error > px)] = 1
    px_error = px_error.sum()/valid.sum()
    return px_error

def px_error_tensor(disp_pred:torch.tensor, gt:torch.tensor, valid:torch.tensor, px:int):

    disp_error = torch.abs(gt - disp_pred)
    disp_error[~valid] = torch.nan
    px_error = torch.zeros_like(disp_error)
    px_error[disp_error > px] = 1
    px_error = px_error.sum()/valid.sum()
    return px_error.cpu()
