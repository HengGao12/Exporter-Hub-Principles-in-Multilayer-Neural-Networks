"""
 This training code is also developed based on https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/lif_fc_mnist.py in Fang et al.'s work.
 
 Wei Fang et al., SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence.Sci. Adv.9,eadi1480(2023).DOI:10.1126/sciadv.adi1480
"""
import time
import argparse
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
import torchvision
import numpy as np
import random
import logging
from spikingjelly.activation_based import neuron, encoding, surrogate, layer, functional
from tqdm import tqdm
from snn import SNN
import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import ipdb

# set seed
def set_seed(seed_value):
    random.seed(seed_value)  
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed_value)  
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False 

set_seed(0)

# set logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler('./log_file/snn_training_log.txt')) # save the log to file

parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('-device', default='cuda:6', help='device')
parser.add_argument('-epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-data-dir', type=str, default='./data', help='root dir of MNIST dataset')
parser.add_argument('-out-dir', type=str, default='./model_output', help='root dir for saving logs and checkpoint')
parser.add_argument('-amp', default=True, help='automatic mixed precision training')
parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

args = parser.parse_args()  
# data loader
train_dataset = torchvision.datasets.MNIST(
    root=args.data_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root=args.data_dir,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
train_data_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True 
)
test_data_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True 
)


def evaluation(net, test_data_loader):
    encoder = encoding.PoissonEncoder()
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)  # (64,)
            # label_onehot = F.one_hot(label, 10).float()  # (64, 10)
            out_fr = 0.
            for t in range(args.T):
                encoded_img = encoder(img)
                out_fr += net(encoded_img)
            out_fr = out_fr / args.T  # (64, 10)
            # loss = F.mse_loss(out_fr, label_onehot)
            loss = F.cross_entropy(out_fr, label)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
    test_time = time.time()
    # test_speed = test_samples / (test_time - train_time)
    test_loss /= test_samples
    test_acc /= test_samples
    print(f'Final accuracy of the SNN on the test images: {test_acc}')    


def main():
    '''
        train SNN
    '''
    net = SNN(tau=args.tau)
    ipdb.set_trace()
    net.to(args.device)
    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    encoder = encoding.PoissonEncoder()
    
    print('Start training.')
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in tqdm(train_data_loader):
            # ipdb.set_trace()
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            # label_onehot = F.one_hot(label, 10).float()

            out_fr = 0.
            for t in range(args.T):
                encoded_img = encoder(img)
                out_fr += net(encoded_img)
            out_fr = out_fr / args.T
            # loss = F.mse_loss(out_fr, label_onehot)
            loss = F.cross_entropy(out_fr, label)
            
            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # evaluation in training
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                # label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                # loss = F.mse_loss(out_fr, label_onehot)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc*100:.4f}, test_loss: {test_loss:.4f}, test_acc: {test_acc*100:.4f}, max_test_acc: {max_test_acc*100:.4f}')
    
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(net.state_dict(), './model_output/best_snn_model_mnist_check.pth')
            print(f'Save best model at epoch {epoch} with test_acc: {test_acc*100: .4f}.')
    
    print('Training complete. Enjoy!')
    
    # evaluate the model   
    evaluation(net, test_data_loader)

if __name__ == '__main__':
    main()