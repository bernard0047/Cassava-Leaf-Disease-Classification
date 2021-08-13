#!/usr/bin/env python
# coding: utf-8
package_path = '../input/pytorch-image-models/pytorch-image-models-master' #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
import sys; sys.path.append(package_path)
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Transpose, ShiftScaleRotate,
    Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip,
    OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from  torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset,DataLoader
from albumentations.pytorch import ToTensorV2
from scipy.ndimage.interpolation import zoom
from contextlib import contextmanager
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
from skimage import io
from tqdm import tqdm
from glob import glob
from torch import nn
import pandas as pd
import torchvision
import numpy as np
import warnings
import pydicom
import sklearn
import joblib
import random
import torch
import timm
import time
import cv2
import os


# # Efficientnet_B3 (Without SnapMix):

CFG = {
    'fold_num': 10,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b3_ns',
    'img_size': 512,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 16,
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 1, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [6,7,8,9],
    'weights': [1,1,1,1]
}

train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

class CassavaDataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
        img  = get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img

def get_train_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

seed_everything(CFG['seed'])
folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)

for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
    if fold > 0:
        break 

    print('Inference fold {} started'.format(fold))

    valid_ = train.loc[val_idx,:].reset_index(drop=True)
    valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(), output_label=False)
    
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
    test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)
    
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)
    
    val_preds = []
    eff_preds = []
    
    #for epoch in range(CFG['epochs']-3):
    for i, epoch in enumerate(CFG['used_epochs']):    
        model.load_state_dict(torch.load('../input/cassava-eff-results-1/eff_b3_without_snap_32_5_10/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
        
        with torch.no_grad():
            for _ in range(CFG['tta']):
                #val_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, val_loader, device)]
                eff_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

    #val_preds = np.mean(val_preds, axis=0) 
    eff_preds = np.sum(eff_preds, axis=0) 
    
    #print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
    #print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
    
    del model
    torch.cuda.empty_cache()

# # ViT Base_Patch16_384:

package_path = '../input/pytorch-image-models/pytorch-image-models-master' #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
import sys; sys.path.append(package_path)

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'vit_base_patch16_384',
    'img_size': 384,
    'epochs': 10,
    'train_bs': 32,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 1, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [5,6,7,8,9],
    'weights': [1,1,1,1,1]
}

submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        MODEL_PATH = "../input/vit-base-models-pretrained-pytorch/jx_vit_base_p16_384-83fb41ba.pth"
        self.model = timm.create_model("vit_base_patch16_384", pretrained=False)
        
        self.model.load_state_dict(torch.load(MODEL_PATH))

        self.model.head = nn.Linear(self.model.head.in_features, 5)
        
    def forward(self, x):
        x = self.model(x)
        return x

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

seed_everything(CFG['seed'])
folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)

for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
    if fold > 0:
        break 

    print('Inference fold {} started'.format(fold))

    valid_ = train.loc[val_idx,:].reset_index(drop=True)
    valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(), output_label=False)
    
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
    test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)
    
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)
    
    val_preds = []
    vit_preds = []
    
    #for epoch in range(CFG['epochs']-3):
    for i, epoch in enumerate(CFG['used_epochs']):    
        model.load_state_dict(torch.load('../input/cassava-eff-results-1/vision_trans_64_4_10/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
        
        with torch.no_grad():
            for _ in range(CFG['tta']):
                #val_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, val_loader, device)]
                vit_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

    #val_preds = np.mean(val_preds, axis=0) 
    vit_preds = np.sum(vit_preds, axis=0) 
    
    #print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
    #print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
    
    del model
    torch.cuda.empty_cache()

# # ResNeXt50_32x4d:

# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
MODEL_DIR = '../input/cassava-eff-results-1/resnext_snap_inc_32_5_10/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = '../input/cassava-leaf-disease-classification/train_images'
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'


# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    num_workers=4
    model_name='resnext50_32x4d'
    size=512
    batch_size=16
    seed=42
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=False
    inference=True
    tta=3
    used_epochs= [8,9,10,11,12]
    weights= [1,1,1,1,1]

# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'inference.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

#LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)


train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

# # ====================================================
# # Dataset
# # ====================================================
class CassavaDataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
        img  = get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img

# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            #Resize(CFG.size, CFG.size),
            RandomResizedCrop(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        n_features = backbone.fc.in_features
        self.model = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, CFG.target_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward_features(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats

# ====================================================
# Helper functions
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                #y_preds = model(images)
                y_preds,_ = model(images) #snapmix
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        #image_preds = model(imgs)
        image_preds, _ = model(imgs)   #for snapmix inference
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

seed_everything(CFG.seed)
folds = StratifiedKFold(n_splits=CFG.n_fold).split(np.arange(train.shape[0]), train.label.values)

for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
    if fold > 0:
        break 

    print('Inference fold {} started'.format(fold))

    valid_ = train.loc[val_idx,:].reset_index(drop=True)
    valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(), output_label=False)
    
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
    test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)
    
    
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device('cuda:0')
    model = CustomResNext(CFG.model_name, pretrained=False).to(device)
    
    val_preds = []
    resv1tta = []
    
    #for epoch in range(CFG['epochs']-3):
    for i, epoch in enumerate(CFG.used_epochs):    
        model.load_state_dict(torch.load(MODEL_DIR+f'{CFG.model_name}_fold{i}_best.pth')['model'])
        
        with torch.no_grad():
            for _ in range(CFG.tta):
                #val_preds += [CFG.weights[i]/sum(CFG.weights)/CFG.tta*inference_one_epoch(model, val_loader, device)]
                resv1tta += [CFG.weights[i]/sum(CFG.weights)/CFG.tta*inference_one_epoch(model, tst_loader, device)]

    #val_preds = np.mean(val_preds, axis=0) 
    resv1tta = np.sum(resv1tta, axis=0) 
    
    #print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
    #print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
    
    del model
    torch.cuda.empty_cache()

# ====================================================
# Dataset
# ====================================================
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image
    
# ====================================================
# Helper functions
# ====================================================
def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                #y_preds = model(images)
                y_preds, _ = model(images) #for snapmix
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs

# ====================================================
# inference
# ====================================================
model = CustomResNext(CFG.model_name, pretrained=False)
states = [torch.load(MODEL_DIR+f'{CFG.model_name}_fold{fold}_best.pth') for fold in range(5)]
test_dataset = TestDataset(test, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
resv1notta = inference(model, states, test_loader, device)

MODEL_DIR = '../input/cassava-resnext50-results-1/resnext50_master_512_32_5_10/'
# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    num_workers=4
    model_name='resnext50_32x4d'
    size=512
    batch_size=16
    seed=42
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=False
    inference=True
    tta=3
    #used_epochs= [0,1,2,3,4]
    used_folds = [0,2,3,4]
    weights= [1,1,1,1]

# ====================================================
# inference
# ====================================================
model = CustomResNext(CFG.model_name, pretrained=False)
states = [torch.load(MODEL_DIR+f'{CFG.model_name}_fold{fold}_best.pth') for fold in CFG.used_folds]
test_dataset = TestDataset(test, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
resv2notta = inference(model, states, test_loader, device)


res_preds = 0.5*resv1tta + 0.25*resv1notta + 0.25*resv2notta
res_preds


# # EfficientNet_B3 (With SnapMix):

CFG = {
    'fold_num': 10,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b3_ns',
    'img_size': 512,
    'epochs': 25,
    'train_bs': 32,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 1, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [6,7,8,9],
    'weights': [1,1,1,1]
}


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        backbone = timm.create_model(CFG['model_arch'], pretrained=pretrained)
        n_features = backbone.classifier.in_features  #backbone.classifier.in_features
        self.model = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, n_class)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats


def get_train_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds,_ = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

seed_everything(CFG['seed'])
folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)

for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
    if fold > 0:
        break 

    print('Inference fold {} started'.format(fold))

    valid_ = train.loc[val_idx,:].reset_index(drop=True)
    valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(), output_label=False)
    
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
    test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)
    
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)
    
    val_preds = []
    effsnapmix_preds = []
    
    #for epoch in range(CFG['epochs']-3):
    for i, epoch in enumerate(CFG['used_epochs']):    
        model.load_state_dict(torch.load('../input/cassava-eff-results-2/eff_b3_smapmix_512_32_5_10/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
        
        with torch.no_grad():
            for _ in range(CFG['tta']):
                #val_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, val_loader, device)]
                effsnapmix_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

    #val_preds = np.mean(val_preds, axis=0) 
    effsnapmix_preds = np.sum(effsnapmix_preds, axis=0) 
    
    #print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
    #print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
    
    del model
    torch.cuda.empty_cache()

# # ResNeSt_50d:

# CFG = {
#     'fold_num': 5,
#     'seed': 719,
#     'model_arch': 'resnest50d',
#     'img_size': 512,
#     'epochs': 10,
#     'train_bs': 32,
#     'valid_bs': 32,
#     'lr': 1e-4,
#     'num_workers': 4,
#     'accum_iter': 1, # suppoprt to do batch accumulation for backprop with effectively larger batch size
#     'verbose_step': 1,
#     'device': 'cuda:0',
#     'tta': 5,
#     'used_epochs': [6,7,8,9],
#     'weights': [1,1,1,1]
# }


# train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
# submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')


# class CassavaDataset(Dataset):
#     def __init__(
#         self, df, data_root, transforms=None, output_label=True
#     ):
        
#         super().__init__()
#         self.df = df.reset_index(drop=True).copy()
#         self.transforms = transforms
#         self.data_root = data_root
#         self.output_label = output_label
    
#     def __len__(self):
#         return self.df.shape[0]
    
#     def __getitem__(self, index: int):
        
#         # get labels
#         if self.output_label:
#             target = self.df.iloc[index]['label']
          
#         path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
#         img  = get_img(path)
        
#         if self.transforms:
#             img = self.transforms(image=img)['image']
            
#         # do label smoothing
#         if self.output_label == True:
#             return img, target
#         else:
#             return img


# def get_train_transforms():
#     return Compose([
#             RandomResizedCrop(CFG['img_size'], CFG['img_size']),
#             Transpose(p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             ShiftScaleRotate(p=0.5),
#             HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#             RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#             CoarseDropout(p=0.5),
#             Cutout(p=0.5),
#             ToTensorV2(p=1.0),
#         ], p=1.)
  
        
# def get_valid_transforms():
#     return Compose([
#             CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
#             Resize(CFG['img_size'], CFG['img_size']),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#             ToTensorV2(p=1.0),
#         ], p=1.)

# def get_inference_transforms():
#     return Compose([
#             RandomResizedCrop(CFG['img_size'], CFG['img_size']),
#             Transpose(p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#             RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#             ToTensorV2(p=1.0),
#         ], p=1.)

# class CassvaImgClassifier(nn.Module):
#     def __init__(self, model_arch, n_class, pretrained=False):
#         super().__init__()
#         backbone = timm.create_model('resnest50d', pretrained=pretrained)
#         n_features = backbone.fc.in_features  #backbone.classifier.in_features
#         self.model = nn.Sequential(*backbone.children())[:-2]
#         self.classifier = nn.Linear(n_features, n_class)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward_features(self, x):
#         x = self.model(x)
#         return x

#     def forward(self, x):
#         feats = self.forward_features(x)
#         x = self.pool(feats).view(x.size(0), -1)
#         x = self.classifier(x)
#         return x, feats

# def inference_one_epoch(model, data_loader, device):
#     model.eval()

#     image_preds_all = []
    
#     pbar = tqdm(enumerate(data_loader), total=len(data_loader))
#     for step, (imgs) in pbar:
#         imgs = imgs.to(device).float()
        
#         image_preds,_ = model(imgs)   #output = model(input)
#         image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
#     image_preds_all = np.concatenate(image_preds_all, axis=0)
#     return image_preds_all

# seed_everything(CFG['seed'])
# folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)

# for fold, (trn_idx, val_idx) in enumerate(folds):
#     # we'll train fold 0 first
#     if fold == 1: 

#         print('Inference fold {} started'.format(fold))

#         valid_ = train.loc[val_idx,:].reset_index(drop=True)
#         valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(), output_label=False)

#         test = pd.DataFrame()
#         test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
#         test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)

#         val_loader = torch.utils.data.DataLoader(
#             valid_ds, 
#             batch_size=CFG['valid_bs'],
#             num_workers=CFG['num_workers'],
#             shuffle=False,
#             pin_memory=False,
#         )

#         tst_loader = torch.utils.data.DataLoader(
#             test_ds, 
#             batch_size=CFG['valid_bs'],
#             num_workers=CFG['num_workers'],
#             shuffle=False,
#             pin_memory=False,
#         )

#         device = torch.device(CFG['device'])
#         model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)

#         #val_preds = []
#         resnest_preds = []

#         #for epoch in range(CFG['epochs']-3):
#         for i, epoch in enumerate(CFG['used_epochs']):    
#             model.load_state_dict(torch.load('../input/resnest-firstfold-898/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))

#             with torch.no_grad():
#                 for _ in range(CFG['tta']):
#                     #val_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, val_loader, device)]
#                     resnest_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

#         #val_preds = np.mean(val_preds, axis=0) 
# #             resnest_preds = np.sum(resnest_preds, axis=0) 

#         #print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
#         #print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))

#         del model
#         torch.cuda.empty_cache()

# # SEResNeXt:
# 1. SEResNeXt101

# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
MODEL_DIR = '../input/cassava-seresnext-results-1/seresnext101/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = '../input/cassavapreprocessed/train_images/train_images'
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'

# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    num_workers=4
    model_name='seresnext101_32x4d'
    size=512
    batch_size=16
    seed=42
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=False
    inference=True
    tta=3
    used_folds= [0,1,2,3,4]
    weights= [1,1,1,1,1]

train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo

pretrained_settings = {
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'],         'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

def se_resnext101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            #Resize(CFG.size, CFG.size),
            RandomResizedCrop(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

class CustomSEResNext101(nn.Module):
    def __init__(self,pretrained='imagenet'):
        super().__init__()
        backbone = se_resnext101(pretrained=pretrained)
        n_features = backbone.last_linear.in_features
        self.model = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, CFG.target_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward_features(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        #image_preds = model(imgs)
        image_preds, _ = model(imgs)   #for snapmix inference
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

seed_everything(CFG.seed)
folds = StratifiedKFold(n_splits=CFG.n_fold).split(np.arange(train.shape[0]), train.label.values)

for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
    if fold > 0:
        break 

    print('Inference fold {} started'.format(fold))

    valid_ = train.loc[val_idx,:].reset_index(drop=True)
    valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(), output_label=False)
    
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
    test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)
    
    
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device('cuda:0')
    model = CustomSEResNext101(pretrained=None).to(device)
    
    val_preds = []
    serespreds101 = []
    
    #for epoch in range(CFG['epochs']-3):
    for i, fold in enumerate(CFG.used_folds):    
        model.load_state_dict(torch.load(MODEL_DIR+f'{CFG.model_name}_fold{fold}_best.pth')['model'])
        
        with torch.no_grad():
            for _ in range(CFG.tta):
                #val_preds += [CFG.weights[i]/sum(CFG.weights)/CFG.tta*inference_one_epoch(model, val_loader, device)]
                serespreds101 += [CFG.weights[i]/sum(CFG.weights)/CFG.tta*inference_one_epoch(model, tst_loader, device)]

    #val_preds = np.mean(val_preds, axis=0) 
    serespreds101 = np.sum(serespreds101, axis=0) 
    
    #print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
    #print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
    
    del model
    torch.cuda.empty_cache()


# 2. SEResNeXt50
# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    num_workers=4
    model_name='seresnext50_32x4d'
    size=512
    batch_size=16
    seed=42
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=False
    inference=True
    tta=3
    used_folds= [0,1,2,3,4]
    weights= [1,1,1,1,1]

class CustomSEResNext50(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=False):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        n_features = backbone.fc.in_features
        self.model = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, CFG.target_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward_features(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats


MODEL_DIR = '../input/cassava-seresnext-results-1/seresnext50_with_snapmix/'

def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                #y_preds = model(images)
                y_preds, _ = model(images) #for snapmix
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs

model = CustomSEResNext50(CFG.model_name, pretrained=False)
states = [torch.load(MODEL_DIR+f'{CFG.model_name}_fold{fold}_best.pth') for fold in range(5)]
test_dataset = TestDataset(test, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
serespreds50 = inference(model, states, test_loader, device)

seresblend = 0.7*serespreds50 + 0.3*serespreds101

PREDS = 0.22*eff_preds + 0.23*vit_preds + 0.12*res_preds + 0.18*effsnapmix_preds + 0.25*seresblend

# for i in PREDS:
#     i[-1] = 0.975*i[-1]

print(eff_preds,vit_preds,res_preds,effsnapmix_preds,seresblend,'',PREDS,sep='\n')

test['label'] = np.argmax(PREDS, axis=1)
test.to_csv('submission.csv', index=False)