'''Train MVTec with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import numpy as np
from models import *
from loader import Loader, Loader2
from utils import progress_bar
from sklearn.metrics import roc_auc_score
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch MVTec Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--loss_type', type=str, default = None)
args = parser.parse_args()

label_dict = {'bottle': {'good': 0, 'anomaly': 1},
                'cable': {'good': 0, 'anomaly': 1},
                'capsule': {'good': 0, 'anomaly': 1},
                'carpet': {'good': 0, 'anomaly': 1},
                'grid': {'good': 0, 'anomaly': 1},
                'hazelnut': {'good': 0, 'anomaly': 1},
                'metal_nut': {'good': 0, 'anomaly': 1},
                'screw': {'good': 0, 'anomaly': 1},
                'zipper': {'good': 0, 'anomaly': 1},
                'leather': {'good': 0, 'anomaly': 1},
                'pill': {'good': 0, 'anomaly': 1},
                'tile': {'good': 0, 'anomaly': 1},
                'toothbrush': {'good': 0, 'anomaly': 1},
                'transistor': {'good': 0, 'anomaly': 1},
                'wood': {'good': 0, 'anomaly': 1}
            } 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_auroc = 0 
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# MVTec Data
print('==> Preparing data..')
transform_train = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])

transform_test = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])

#grid, screw, zipper 
transform_train_grayscale = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.Grayscale(num_output_channels=3), #grid, screw, zipper
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])

transform_test_grayscale = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.Grayscale(num_output_channels=3), #grid, screw, zipper 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])


print('==> Building model..')



# Training
def train(net, criterion, optimizer, epoch, trainloader, cls_name):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Test
def test(net, criterion, epoch, cycle, testloader, cls_name):
    global best_acc
    global best_auroc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predicted_scores = []  # Store predicted scores for AUROC calculation
    all_targets = []  # Store true labels for AUROC calculation    

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_predicted_scores.extend(outputs[:, 1].cpu().numpy())  # Assuming the positive class is index 1
            all_targets.extend(targets.cpu().numpy())

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    roc_score = roc_auc_score(all_targets, all_predicted_scores)
    print(f"Epoch: {epoch}, AUROC: {roc_score:.4f}")
    
    # Save checkpoint acc.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(f'checkpoint/{args.loss_type}'):
            os.mkdir(f'checkpoint/{args.loss_type}')
        torch.save(state, f'./checkpoint/{args.loss_type}/main_acc_{cls_name}_{cycle}.pth')
        best_acc = acc
    
    # Save checkpoint auroc.
    if roc_score > best_auroc:  
        print('Saving AUROC checkpoint..')
        state = {
            'net': net.state_dict(),
            'auroc': roc_score,  
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(f'checkpoint/{args.loss_type}'):
            os.mkdir(f'checkpoint/{args.loss_type}')
        torch.save(state, f'./checkpoint/{args.loss_type}/main_auroc_{cls_name}_{cycle}.pth')  
        best_auroc = roc_score  



# Sampling
def get_plabels2(net, samples, cycle, cls_name, label_dict, num_classes):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(num_classes)]

    if cls_name== 'grid' or cls_name == 'screw' or cls_name == 'zipper':
        sub5k = Loader2(is_train=False,  transform=transform_test_grayscale, path_list=samples, class_name=cls_name, label_dict = label_dict)
    else:
        sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples, class_name=cls_name, label_dict = label_dict)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            # save top1 Score
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            top1_scores.append(probs[0][predicted.item()].cpu().numpy())
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    # The number of sampling
    k_samples = 1
    return samples[idx[:k_samples]]



if __name__ == '__main__':

    # class name
    def get_subfolders(folder_path):
        subfolders = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
        return subfolders

    target_folder = './DATA'
    class_list = get_subfolders(target_folder)

    for cls_name in class_list:
        print(f'Current class is {cls_name}')

        CYCLES = 10
        labeled = []

        pretrained_resnet = models.resnet18(pretrained=True)
        pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, len(label_dict[cls_name]))
        pretrained_resnet = pretrained_resnet.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(pretrained_resnet)
            cudnn.benchmark = True
            
        
        for cycle in range(CYCLES):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

            best_acc = 0
            best_auroc = 0 
            print('Cycle ', cycle)

            # open 5k batch (sorted high->low)
            with open(f'./loss_batch/{args.loss_type}/{cls_name}/batch0.txt', 'r') as f:
                samples0 = f.readlines()
            with open(f'./loss_batch/{args.loss_type}/{cls_name}/batch1.txt', 'r') as f:
                samples1 = f.readlines()
            with open(f'./loss_batch/{args.loss_type}/{cls_name}/batch2.txt', 'r') as f:
                samples2 = f.readlines()
            with open(f'./loss_batch/{args.loss_type}/{cls_name}/batch3.txt', 'r') as f:
                samples3 = f.readlines()
            with open(f'./loss_batch/{args.loss_type}/{cls_name}/batch4.txt', 'r') as f:
                samples4 = f.readlines()
                
            if cycle > 0:
                print('>> Getting previous checkpoint')

                with open(f'./loss_batch/{args.loss_type}/{cls_name}/batch.txt', 'r') as f:
                    samples = f.readlines()
                
                checkpoint = torch.load(f'./checkpoint/{args.loss_type}/main_acc_{cls_name}_{cycle-1}.pth')
                net.load_state_dict(checkpoint['net'])

                sample1k = get_plabels2(net, samples, cycle, cls_name, label_dict[cls_name], len(label_dict[cls_name]))
            else:
                samples0 = np.array(samples0)
                samples1 = np.array(samples1)
                samples2 = np.array(samples2)
                samples3 = np.array(samples3)
                samples4 = np.array(samples4)

                sample1k = []
                # First data from each batch
                sample1k.append(samples0[int(0)])
                sample1k.append(samples1[int(0)])
                sample1k.append(samples2[int(0)])
                sample1k.append(samples3[int(0)])
                sample1k.append(samples4[int(0)])
                # Second data from each batch
                sample1k.append(samples0[int(1)])
                sample1k.append(samples1[int(1)])
                sample1k.append(samples2[int(1)])
                sample1k.append(samples3[int(1)])
                sample1k.append(samples4[int(1)])
                print(f'The number of sample1k is {len(sample1k)}')


            # add 1k samples to labeled set
            labeled.extend(sample1k)
            print(f'>> Labeled length: {len(labeled)}')
            if cls_name== 'grid' or cls_name == 'screw' or cls_name == 'zipper':

                trainset = Loader2(is_train=True, transform=transform_train_grayscale, path_list=labeled, class_name=cls_name, label_dict=label_dict[cls_name])
            else:
                trainset = Loader2(is_train=True, transform=transform_train, path_list=labeled, class_name=cls_name, label_dict=label_dict[cls_name])

            n_train_batches = 10
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_train_batches, shuffle=True, num_workers=2)
            if cls_name== 'grid' or cls_name == 'screw' or cls_name == 'zipper':

                testset = Loader(is_train=False,  transform=transform_test_grayscale, class_name=cls_name, label_dict=label_dict[cls_name])
            else:
                testset = Loader(is_train=False,  transform=transform_test, class_name=cls_name, label_dict=label_dict[cls_name])

            n_test_batches = 10
            test_batch_size = len(testset) // n_test_batches
            testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


            for epoch in range(20):
                train(net, criterion, optimizer, epoch, trainloader, cls_name)
                test(net, criterion, epoch, cycle, testloader, cls_name)
                scheduler.step()

            
            if not os.path.isdir('main_best_acc'):
                os.mkdir('main_best_acc')
            if not os.path.isdir('main_best_auroc'):
                os.mkdir('main_best_auroc')
            if not os.path.isdir(f'main_best_acc/{args.loss_type}'):
                os.mkdir(f'main_best_acc/{args.loss_type}')
            if not os.path.isdir(f'main_best_auroc/{args.loss_type}'):
                os.mkdir(f'main_best_auroc/{args.loss_type}')

            with open(f'./main_best_acc/{args.loss_type}/main_best_acc_{cls_name}.txt', 'a') as f:
                f.write(str(cycle) + ' ' + str(best_acc)+'\n')
            with open(f'./main_best_auroc/{args.loss_type}/main_best_auroc_{cls_name}.txt', 'a') as f:
                f.write(str(cycle) + ' ' + str(best_auroc)+'\n')
            