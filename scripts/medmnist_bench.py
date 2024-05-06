import json
import argparse
import numpy as np
import torch
from torch.nn.functional import softmax
from torchmetrics import AUROC
from torch.utils.data import DataLoader, ConcatDataset
from utils import torch_acc, count_parameters, WarmupExpSchedule, train_transform, test_transform
from medmnist import OrganMNIST3D, NoduleMNIST3D, FractureMNIST3D, AdrenalMNIST3D, VesselMNIST3D, SynapseMNIST3D
from tqdm import tqdm
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

epochs = 50
lr = 1e-3

parser = argparse.ArgumentParser(description='Train a U-Net model on different preprocessed datasets')
parser.add_argument('run_id', type=int, help='id of the run', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

run_id = parser.parse_args().run_id
run_ids = [run_id]

train_transform = train_transform
test_transform = test_transform

all_datasets = {
    'OrganMNIST3D': {
        'train': OrganMNIST3D(split='train', download=True, transform=train_transform),
        'val': OrganMNIST3D(split='val', download=True, transform=train_transform),
        'test': OrganMNIST3D(split='test', download=True, transform=test_transform),
        'n_classes': 11
    },
    'OrganMNIST3D_64': {
        'train': OrganMNIST3D(split='train', download=True, size=64, transform=train_transform),
        'val': OrganMNIST3D(split='val', download=True, size=64, transform=train_transform),
        'test': OrganMNIST3D(split='test', download=True, size=64, transform=test_transform),
        'n_classes': 11
    },
    'NoduleMNIST3D': {
        'train': NoduleMNIST3D(split='train', download=True, transform=train_transform),
        'val': NoduleMNIST3D(split='val', download=True, transform=train_transform),
        'test': NoduleMNIST3D(split='test', download=True, transform=test_transform),
        'n_classes': 2
    },
    'NoduleMNIST3D_64': {
        'train': NoduleMNIST3D(split='train', download=True, size=64, transform=train_transform),
        'val': NoduleMNIST3D(split='val', download=True, size=64, transform=train_transform),
        'test': NoduleMNIST3D(split='test', download=True, size=64, transform=test_transform),
        'n_classes': 2
    },
    'FractureMNIST3D': {
        'train': FractureMNIST3D(split='train', download=True, transform=train_transform),
        'val': FractureMNIST3D(split='val', download=True, transform=train_transform),
        'test': FractureMNIST3D(split='test', download=True, transform=test_transform),
        'n_classes': 3
    },
    #'FractureMNIST3D_64': {
    #    'train': FractureMNIST3D(split='train', download=True, size=64, transform=train_transform),
    #    'val': FractureMNIST3D(split='val', download=True, size=64, transform=train_transform),
    #    'test': FractureMNIST3D(split='test', download=True, size=64, transform=test_transform),
    #    'n_classes': 3
    #},
    'AdrenalMNIST3D': {
        'train': AdrenalMNIST3D(split='train', download=True, transform=train_transform),
        'val': AdrenalMNIST3D(split='val', download=True, transform=train_transform),
        'test': AdrenalMNIST3D(split='test', download=True, transform=test_transform),
        'n_classes': 2
    },
    #'AdrenalMNIST3D_64': {
    #    'train': AdrenalMNIST3D(split='train', download=True, size=64, transform=train_transform),
    #    'val': AdrenalMNIST3D(split='val', download=True, size=64, transform=train_transform),
    #    'test': AdrenalMNIST3D(split='test', download=True, size=64, transform=test_transform),
    #    'n_classes': 2
    #},
    'VesselMNIST3D': {
        'train': VesselMNIST3D(split='train', download=True, transform=train_transform),
        'val': VesselMNIST3D(split='val', download=True, transform=train_transform),
        'test': VesselMNIST3D(split='test', download=True, transform=test_transform),
        'n_classes': 2
    },
    #'VesselMNIST3D_64': {
    #    'train': VesselMNIST3D(split='train', download=True, size=64, transform=train_transform),
    #    'val': VesselMNIST3D(split='val', download=True, size=64, transform=train_transform),
    #    'test': VesselMNIST3D(split='test', download=True, size=64, transform=test_transform),
    #    'n_classes': 2
    #},
    'SynapseMNIST3D': {
        'train': SynapseMNIST3D(split='train', download=True, transform=train_transform),
        'val': SynapseMNIST3D(split='val', download=True, transform=train_transform),
        'test': SynapseMNIST3D(split='test', download=True, transform=test_transform),
        'n_classes': 2
    },
    'SynapseMNIST3D_64': {
        'train': SynapseMNIST3D(split='train', download=True, size=64, transform=train_transform),
        'val': SynapseMNIST3D(split='val', download=True, size=64, transform=train_transform),
        'test': SynapseMNIST3D(split='test', download=True, size=64, transform=test_transform),
        'n_classes': 2
    },
}

models = ['Vanilla_3D', 'ACS', 'Vanilla_2D', 'QNet']

for dataset in all_datasets:
    if not '64' in dataset:
        continue
    for model_name in models:
        for run_id in run_ids:

            torch.cuda.empty_cache()
            print(f'Running {model_name} on {dataset} with run_id {run_id}...')

            if model_name == 'Vanilla_3D':
                from models import Vanilla_3DCNN
                model = Vanilla_3DCNN(n_classes=all_datasets[dataset]['n_classes'], channel_factor=1.)
            elif model_name == 'ACS':
                from models import ACS_CNN
                model = ACS_CNN(n_classes=all_datasets[dataset]['n_classes'], channel_factor=1.7)
            elif model_name == 'Vanilla_2D':
                from models import Vanilla_2DCNN
                if '64' in dataset:
                    model = Vanilla_2DCNN(n_classes=all_datasets[dataset]['n_classes'], in_channels=64, channel_factor=1.66)
                else:
                    model = Vanilla_2DCNN(n_classes=all_datasets[dataset]['n_classes'], in_channels=28, channel_factor=1.68)
            elif model_name == 'QNet':
                from models import QNet
                if '64' in dataset:
                    model = QNet(n_classes=all_datasets[dataset]['n_classes'], width=64, channel_factor=0.8, isotrope=False)
                else:
                    model = QNet(n_classes=all_datasets[dataset]['n_classes'], width=28, channel_factor=1.68, isotrope=False)
            elif model_name == 'QNet_iso':
                from models import QNet
                if '64' in dataset:
                    model = QNet(n_classes=all_datasets[dataset]['n_classes'], width=64, channel_factor=0.8, isotrope=True)
                else:
                    model = QNet(n_classes=all_datasets[dataset]['n_classes'], width=28, channel_factor=1.75, isotrope=True)

            dl_train = DataLoader(ConcatDataset([all_datasets[dataset]['train'], 
                                                all_datasets[dataset]['val']]), batch_size=32, shuffle=True, 
                                                num_workers=4)
            dl_test = DataLoader(all_datasets[dataset]['test'], batch_size=32, shuffle=False, num_workers=4)

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = WarmupExpSchedule(optimizer, warmup_steps=5, decay_rate=0.96, t_total=epochs)
            loss_fn = torch.nn.CrossEntropyLoss()
            metric = AUROC(task='multiclass', num_classes=all_datasets[dataset]['n_classes'])

            train_time = []
            loop_train_time = []
            loop_test_time = []
            train_losses = []
            test_losses = []
            train_acc = []
            train_auc = []
            test_acc = []
            test_auc = []
            start_time = time.time()
            for epoch in tqdm(range(epochs)):
                
                # TRAINING
                
                model.train()
                losses = []
                acc = 0
                auc = 0
                N = len(dl_train)
                loop_train = 0.
                for i,(x,y) in enumerate(dl_train):
                    x, y = x.type(torch.float32).to(device), y.type(torch.float32).squeeze().to(device)

                    start_train_time = time.time()

                    if model_name == 'Vanilla_2D' or model_name == 'QNet' or model_name == 'QNet_iso':
                        x = x[:,0,:,:,:] # Take only one channel, consider dimensions as (batch, channel, x, y)

                    y_pred = model(x)
                    loss = loss_fn(y_pred, y.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach().cpu().numpy())
                    acc += torch_acc(y_pred, y).item() / len(dl_train)
                    y_pred = softmax(y_pred, dim=1)
                    auc += metric(y_pred.detach().cpu(), 
                                y.type(torch.int32).detach().cpu()).item() / len(dl_train)
                    
                    loop_train += time.time() - start_train_time
                
                train_losses.append(float(np.mean(losses)))
                train_acc.append(float(acc))
                train_auc.append(float(auc))
                train_time.append(time.time() - start_time)
                loop_train_time.append(loop_train)
                scheduler.step()
                
                # TESTING
                
                model.eval()
                losses = []
                acc = 0
                auc = 0
                N = len(dl_test)
                loop_test = 0.
                for i,(x,y) in enumerate(dl_test):
                    x, y = x.type(torch.float32).to(device), y.type(torch.float32).squeeze().to(device)

                    start_test_time = time.time()

                    if model_name == 'Vanilla_2D' or model_name == 'QNet' or model_name == 'QNet_iso':
                        x = x[:,0,:,:,:] # Take only one channel, consider dimensions as (batch, channel, x, y)
                    
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor))
                    losses.append(loss.detach().cpu().numpy())
                    acc += torch_acc(y_pred, y).item() / len(dl_test)
                    y_pred = softmax(y_pred, dim=1)
                    auc += metric(y_pred.detach().cpu(), 
                                y.type(torch.int32).detach().cpu()).item() / len(dl_test)
                    
                    loop_test += time.time() - start_test_time
                
                test_losses.append(float(np.mean(losses)))
                test_acc.append(float(acc))
                test_auc.append(float(auc))
                loop_test_time.append(loop_test)

            summary = {
                'best_epoch': int(np.argmin(test_losses)),
                'best_test_acc': float(np.max(test_acc)),
                'best_test_auc': float(np.max(test_auc)),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_acc': train_acc,
                'train_auc': train_auc,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'n_parameters': int(count_parameters(model)),
                'train_time': train_time,
                'loop_train_time': loop_train_time,
                'loop_test_time': loop_test_time,
                'time': float(time.time() - start_time)
            }

            # Dump the summary to a file
            with open(f'./results/{model_name}_{dataset}_{str(run_id)}.json', 'w') as f:
                json.dump(summary, f, indent=4)

            print(f'Finished {model_name} on {dataset} with run_id {run_id}...')
            print(f'Best acc: {summary["best_test_acc"]}, Best auc: {summary["best_test_auc"]}')
