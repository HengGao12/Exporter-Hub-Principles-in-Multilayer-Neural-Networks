import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from ann import ANN
from snn import SNN

from spikingjelly.activation_based import neuron, encoding, surrogate, layer, functional

import argparse

import random

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import ipdb

'''
    The code is developed based on: https://github.com/Jingkang50/OpenOOD/blob/main/tools/plot/tsne_tools.py
'''

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

def plot(x, colors, args, layer_idx):
    # Choosing color palette
    palette = np.array(sns.color_palette("pastel", 10))
    # pastel, husl, and so on
    # Create a scatter plot.
    f = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    # ax = plt.subplot(aspect='equal')
    ax.set_aspect('equal')
    ax.set_xlim([-120.0, 120.0])
    ax.set_ylim([-120.0, 120.0])
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    
    plt.axis('off')
    plt.savefig('./figures/tsne/tsne_{}_{}_layer{}.png'.format(args.net, args.delete_type, layer_idx), dpi=700, bbox_inches="tight", pad_inches=0)
    
def tsne(args):
    '''
        TSNE visualization for ANN and SNN
    '''
    feature_list = []
    label_list = []
    if args.net=="ann":
        # load the model
        if args.delete_type == "control":
            model.load_state_dict(torch.load('./model_output/best_ann_model_mnist.pth'))
        elif args.delete_type == "weak":
            model.load_state_dict(torch.load('./model_output/best_ann_model_mnist_retrain_prune_weak_neurons_ratio_1.0.pth'))
        elif args.delete_type == "mid":
            model.load_state_dict(torch.load('./model_output/best_ann_model_mnist_retrain_prune_mid_neurons_ratio_1.0.pth'))
        else:
            model.load_state_dict(torch.load('./model_output/best_ann_model_mnist_retrain_prune_hub_neurons_ratio_1.0.pth'))
        
        # get feature lists and plot tsne
        model.eval()                  
        if args.layer_num == 1: # the input layer
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):
                    images = images.view(images.shape[0],-1).cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    outputs = model(images)
                    l1_feature = model.l1  # Layer 1 feature, torch.Size([64, 1200])
                    feature_list.append(l1_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break    
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])  # (704)
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num)                
        elif args.layer_num == 2: # hidden layer
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):
                    images = images.view(images.shape[0],-1).cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    outputs = model(images)
                    l2_feature = model.l2  # Layer 2 feature, torch.Size([64, 1200])
                    feature_list.append(l2_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break    
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num) 
        elif args.layer_num == 3:
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):
                    images = images.view(images.shape[0],-1).cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    outputs = model(images)
                    l3_feature = model.l3  # Layer 3 feature, torch.Size([64, 1200])
                    feature_list.append(l3_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break    
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num) 
        else:  # the output layer
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):
                    images = images.view(images.shape[0],-1).cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    outputs = model(images)
                    l4_feature = model.out  # Layer 4 feature, torch.Size([64, 10])
                    feature_list.append(l4_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break    
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num) 
    else:
        '''
            TSNE for SNN
        '''
        if args.delete_type == "control":
            model.load_state_dict(torch.load('./model_output/best_snn_model_mnist.pth', map_location='cuda:0'))
        elif args.delete_type == "weak":
            model.load_state_dict(torch.load('./model_output/best_snn_model_mnist_retrain_prune_weak_neurons_ratio_1.0.pth', map_location='cuda:0'))
        elif args.delete_type == "mid":
            model.load_state_dict(torch.load('./model_output/best_snn_model_mnist_retrain_prune_mid_neurons_ratio_1.0.pth', map_location='cuda:0'))
        else:
            model.load_state_dict(torch.load('./model_output/best_snn_model_mnist_retrain_prune_hub_neurons_ratio_1.0.pth', map_location='cuda:0'))
        
        encoder = encoding.PoissonEncoder()
        # get feature lists and plot tsne
        model.eval()                  
        if args.layer_num == 1: # the input layer
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):
                    images = images.cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    # outputs = model(images)
                    out_fr = 0.
                    for t in range(100):
                        encoded_img = encoder(images)
                        out_fr += model(encoded_img)
                    out_fr = out_fr / 100  # (64, 10)
                    l1_feature = model.x1  
                    feature_list.append(l1_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break    
                    functional.reset_net(model)
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])  # (704)
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num)                
        elif args.layer_num == 2: # hidden layer
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):
                    images = images.cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    # outputs = model(images)
                    out_fr = 0.
                    for t in range(100):
                        encoded_img = encoder(images)
                        out_fr += model(encoded_img)
                    out_fr = out_fr / 100  # (64, 10)
                    l2_feature = model.x2 
                    feature_list.append(l2_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break  
                    functional.reset_net(model)  
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num) 
        elif args.layer_num == 3:
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):
                    images = images.cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    # outputs = model(images)
                    out_fr = 0.
                    for t in range(100):
                        encoded_img = encoder(images)
                        out_fr += model(encoded_img)
                    out_fr = out_fr / 100  # (64, 10)
                    l3_feature = model.x3  
                    feature_list.append(l3_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break    
                    functional.reset_net(model)
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num) 
        else:  # the output layer
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_data_loader):  # len(test_data_loader)=157
                    images = images.cuda()  # torch.Size([64, 784])
                    # labels = labels.cuda()  # torch.Size([64])
                    # outputs = model(images)
                    out_fr = 0.
                    for t in range(100):
                        encoded_img = encoder(images)
                        out_fr += model(encoded_img)
                    out_fr = out_fr / 100  # (64, 10)
                    l4_feature = out_fr  # (64, 10)
                    feature_list.append(l4_feature.detach().cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if i == 99:  # here we take 100 batches of data
                        break
                    functional.reset_net(model)    
                f = np.stack(np.array(feature_list),axis=0)
                f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])  # [704, 1200]
                l = np.stack(np.array(label_list),axis=0)
                l = l.reshape(l.shape[0]*l.shape[1])
                if args.n_components < f.shape[1]:
                    pca = PCA(n_components=50)
                    f = pca.fit_transform(f)
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=2000)
                tsne_pos = tsne.fit_transform(f)
                plot(tsne_pos, l, args, layer_idx=args.layer_num) 


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prunning Framework for ANN/SNN')
    parser.add_argument('--net', type=str, default='ann', help='ann or snn')
    parser.add_argument('--tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('--delete_type', default='hub', type=str, help='choose to remove which group of neuron: control, weak, mid, hub')
    parser.add_argument('--layer_num', default=1, type=int, help='choose which layer to be visualized')
    parser.add_argument('--n_components', default=50, type=int, help='tsne parameter')
    args = parser.parse_args() 
    
    # load the data
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
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
    
    if args.net == 'ann': 
        model = ANN().cuda()
    else:    
        model = SNN(tau=args.tau).cuda()
        
    tsne(args=args)    