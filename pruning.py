import argparse
import numpy as np
import random
import torch
from ann import ANN
from snn import SNN
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, encoding, surrogate, layer, functional
import torchvision
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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


# Evaluation for ANN
def evaluate_ann(net, test_data_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.view(images.shape[0], -1).cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Evaluation for SNN
def evaluate_snn(net, test_data_loader):
    encoder = encoding.PoissonEncoder()
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.cuda()
            label = label.cuda()  # (64,)
            # label_onehot = F.one_hot(label, 10).float()  # (64, 10)
            out_fr = 0.
            for t in range(100):
                encoded_img = encoder(img)
                out_fr += net(encoded_img)
            out_fr = out_fr / 100  # (64, 10)
            # loss = F.mse_loss(out_fr, label_onehot)
            loss = F.cross_entropy(out_fr, label)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
    # test_time = time.time()
    # test_speed = test_samples / (test_time - train_time)
    test_loss /= test_samples
    test_acc /= test_samples
    # print(f'Final accuracy of the SNN on the test images: {test_acc}%') 
    return test_acc * 100   



def retrain(model, model_type, neuron_type, ratio, exp_round):
    '''
        retrain the model
    '''
    if model_type == "ann":
        '''
            Retraining for ANN
        '''
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        best_accuracy = 0
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_data_loader):
                images = images.view(images.shape[0], -1).cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # save the model with the best accuracy
            acc = evaluate_ann(model, test_data_loader)
            print("Epoch: {}, Acc: {}%".format(epoch, acc))
            if acc > best_accuracy:
                best_accuracy = acc  
                torch.save(model.state_dict(), './model_output/best_ann_model_mnist_retrain_prune_{}_neurons_ratio_{}.pth'.format(neuron_type, ratio))
    else:
        '''
            Retraining for SNN
        '''
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        encoder = encoding.PoissonEncoder()
        num_epochs = 10
        max_test_acc = -1
        for epoch in range(num_epochs):
            model.train()
            for images, labels in tqdm(train_data_loader):
                optimizer.zero_grad()
                images = images.cuda()
                labels = labels.cuda()
            
                out_fr = 0.
                for t in range(100):
                    encoded_img = encoder(images)
                    out_fr += model(encoded_img)
                out_fr = out_fr / 100
                
                loss = criterion(out_fr, labels)
                loss.backward()
                optimizer.step()
                functional.reset_net(model)
            acc = evaluate_snn(model, test_data_loader)
            print("Epoch: {}, Acc: {}%".format(epoch, acc))
            
            if acc > max_test_acc:
                max_test_acc = acc
                # save the model with the best accuracy
                torch.save(model.state_dict(), './model_output/best_snn_model_mnist_retrain_prune_{}_neurons_ratio_{}_exp{}.pth'.format(neuron_type, ratio, exp_round))



def base_pruning(model, model_type, exp_round, del_type="hub"):
    '''
        pruning without retraining
    '''
    # The pipeline refers to this link: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html 
    if model_type == "ann":
        '''
            Pruning for ANN
        '''
        model.eval()  
        for (name, module) in model.named_modules():
            if name == "layer1":
                weight_size = module.weight.data.shape
                weights = module.weight.data  # torch.Size([1200, 784]), device = cuda
                # weight_sums = torch.sum(torch.abs(weights), dim=0) # definition 1
                weight_sums = torch.abs(torch.sum(weights, dim=0)) # definition 2
                B, I = torch.sort(weight_sums)
                weaksize = 261 
                midsize = 262 
                hubsize = 261 
                if del_type == "weak":    
                    '''
                        drop weak neurons in Layer 1
                    '''
                    weak_drop_acc_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        weak_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * weaksize)).int()
                        rand_perm = torch.randperm(weaksize)
                        idx = I[rand_perm[:drop_num]]
                        weak_drop_nn.layer1.weight.data[:,idx] = torch.zeros_like(weak_drop_nn.layer1.weight.data[:,idx]).cuda()
                        weak_drop_acc = evaluate_ann(weak_drop_nn, test_data_loader)
                        print("Acc after dropping weak neurons: {}%".format(weak_drop_acc))    # Gradually deactivating weak neurons   
                        weak_drop_acc_list.append(weak_drop_acc)
                    df = pd.DataFrame(weak_drop_acc_list, columns=['Weak'])
                    df.to_excel('./pruning_results/weak_drop_acc_d2.xlsx', index=False) 
                elif del_type == "mid":
                    '''
                        drop mid neurons in Layer 1
                    '''
                    mid_drop_acc_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        mid_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * midsize)).int()
                        rand_perm = torch.randperm(midsize)
                        idx = I[weaksize:][rand_perm[:drop_num]]
                        mid_drop_nn.layer1.weight.data[:,idx] = torch.zeros_like(mid_drop_nn.layer1.weight.data[:,idx]).cuda()
                        mid_drop_acc = evaluate_ann(mid_drop_nn, test_data_loader)
                        print("Acc after dropping mid neurons: {}%".format(mid_drop_acc))    # Gradually deactivating mid neurons   
                        mid_drop_acc_list.append(mid_drop_acc)
                    df = pd.DataFrame(mid_drop_acc_list, columns=['Mid'])
                    df.to_excel('./pruning_results/mid_drop_acc_d2.xlsx', index=False)
                else:
                    '''
                        drop hub neurons in Layer 1
                    '''
                    hub_drop_acc_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        hub_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * hubsize)).int()
                        rand_perm = torch.randperm(hubsize)
                        idx = I[weaksize+midsize:][rand_perm[:drop_num]]
                        hub_drop_nn.layer1.weight.data[:,idx] = torch.zeros_like(hub_drop_nn.layer1.weight.data[:,idx]).cuda()
                        hub_drop_acc = evaluate_ann(hub_drop_nn, test_data_loader)
                        print("Acc after dropping hub neurons: {}%".format(hub_drop_acc))    # Gradually deactivating hub neurons   
                        hub_drop_acc_list.append(hub_drop_acc)
                    df = pd.DataFrame(hub_drop_acc_list, columns=['Hub'])
                    df.to_excel('./pruning_results/hub_drop_acc_d2.xlsx', index=False)  
    else:
        '''
            Pruning for SNN
        '''
        model.eval()  
        for (name, module) in model.named_modules():
            if name == "layer1.1":
                weight_size = module.weight.data.shape
                weights = module.weight.data  # torch.Size([1200, 784]), device = cuda
                # weight_sums = torch.sum(torch.abs(weights), dim=0) # definition 1
                weight_sums = torch.abs(torch.sum(weights, dim=0)) # definition 2
                B, I = torch.sort(weight_sums)
                weaksize = 261 
                midsize = 262  
                hubsize = 261  
                if del_type == "weak":    
                    '''
                        drop weak neurons in Layer 1
                    '''
                    weak_drop_acc_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        weak_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * weaksize)).int()
                        rand_perm = torch.randperm(weaksize)
                        idx = I[rand_perm[:drop_num]]
                        for (name, module) in weak_drop_nn.named_modules():
                            if name == "layer1.1":
                                module.weight.data[:,idx] = torch.zeros_like(module.weight.data[:,idx]).cuda()
                        weak_drop_acc = evaluate_snn(weak_drop_nn, test_data_loader)
                        print("Acc after dropping {} % weak neurons: {}%".format(100*ratio, weak_drop_acc))    # Gradually deactivating weak neurons   
                        weak_drop_acc_list.append(weak_drop_acc)
                    df = pd.DataFrame(weak_drop_acc_list, columns=['Weak'])
                    df.to_excel('./pruning_results/snn_weak_drop_acc_d2_'+ exp_round +'_check.xlsx', index=False) 
                elif del_type == "mid":
                    '''
                        drop mid neurons in Layer 1
                    '''
                    mid_drop_acc_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        mid_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * midsize)).int()
                        rand_perm = torch.randperm(midsize)
                        idx = I[weaksize:][rand_perm[:drop_num]]
                        for (name, module) in mid_drop_nn.named_modules():
                            if name == "layer1.1":
                                module.weight.data[:,idx] = torch.zeros_like(module.weight.data[:,idx]).cuda()
                        mid_drop_acc = evaluate_snn(mid_drop_nn, test_data_loader)
                        print("Acc after dropping mid neurons: {}%".format(mid_drop_acc))    # Gradually deactivating mid neurons   
                        mid_drop_acc_list.append(mid_drop_acc)
                    df = pd.DataFrame(mid_drop_acc_list, columns=['Mid'])
                    df.to_excel('./pruning_results/snn_mid_drop_acc_d2_'+ exp_round +'.xlsx', index=False)
                else:
                    '''
                        drop hub neurons in Layer 1
                    '''
                    hub_drop_acc_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        hub_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * hubsize)).int()
                        rand_perm = torch.randperm(hubsize)
                        idx = I[weaksize+midsize:][rand_perm[:drop_num]]
                        for (name, module) in hub_drop_nn.named_modules():
                            if name == "layer1.1":
                                module.weight.data[:,idx] = torch.zeros_like(module.weight.data[:,idx]).cuda()
                        hub_drop_acc = evaluate_snn(hub_drop_nn, test_data_loader)
                        print("Acc after dropping hub neurons: {}%".format(hub_drop_acc))    # Gradually deactivating hub neurons   
                        hub_drop_acc_list.append(hub_drop_acc)
                    df = pd.DataFrame(hub_drop_acc_list, columns=['Hub'])
                    df.to_excel('./pruning_results/snn_hub_drop_acc_d2_'+ exp_round +'.xlsx', index=False) 
        
     

def pruning_with_retrain(model, model_type, exp_round, del_type="hub"):
    '''
        pruning with retraining
    '''
    if model_type == "ann":
        '''
            Pruning for ANN
        '''
        model.eval()  
        for (name, module) in model.named_modules():
            if name == "layer1":
                weight_size = module.weight.data.shape
                weights = module.weight.data  # torch.Size([1200, 784]), device = cuda
                # weight_sums = torch.sum(torch.abs(weights), dim=0) # definition 1  l1_norm
                weight_sums = torch.abs(torch.sum(weights, dim=0)) # definition 2
                B, I = torch.sort(weight_sums)
                weaksize = 261 
                midsize = 262 
                hubsize = 261 
                if del_type == "weak":    
                    '''
                        drop weak neurons in Layer 1
                    '''
                    weak_drop_acc_retrain_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        print("Dropping weak neurons with ratio: {}".format(ratio))
                        weak_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * weaksize)).int()
                        rand_perm = torch.randperm(weaksize)
                        idx = I[rand_perm[:drop_num]]
                        weak_drop_nn.layer1.weight.data[:,idx] = torch.zeros_like(weak_drop_nn.layer1.weight.data[:,idx]).cuda()
                        for name, param in weak_drop_nn.named_parameters():
                            if "layer1" in name:
                                param.requires_grad = False  # fix the parameters of layer1
                        retrain(weak_drop_nn, model_type, del_type, ratio) # retrain the network
                        # load the model with the best accuracy
                        weak_drop_nn.load_state_dict(torch.load('./model_output/best_ann_model_mnist_retrain_prune_{}_neurons_ratio_{}.pth'.format(del_type, ratio)))
                        # evaluation for the best model in retraining
                        weak_drop_acc_retrain = evaluate_ann(weak_drop_nn, test_data_loader)
                        print("Acc after dropping {} weak neurons and retraining: {}%".format(ratio, weak_drop_acc_retrain))    # Gradually deactivating weak neurons   
                        weak_drop_acc_retrain_list.append(weak_drop_acc_retrain)
                    df = pd.DataFrame(weak_drop_acc_retrain_list, columns=['Weak'])
                    df.to_excel('./pruning_results/weak_drop_acc_d2_w_retrain.xlsx', index=False) 
                elif del_type == "mid":
                    '''
                        drop mid neurons in Layer 1
                    '''
                    mid_drop_acc_retrain_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        print("Dropping mid neurons with ratio: {}".format(ratio))
                        mid_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * midsize)).int()
                        rand_perm = torch.randperm(midsize)
                        idx = I[weaksize:][rand_perm[:drop_num]]
                        mid_drop_nn.layer1.weight.data[:,idx] = torch.zeros_like(mid_drop_nn.layer1.weight.data[:,idx]).cuda()
                        for name, param in mid_drop_nn.named_parameters():
                            if "layer1" in name:
                                param.requires_grad = False   # fix the parameters of layer1
                        retrain(mid_drop_nn, model_type, del_type, ratio)
                        mid_drop_nn.load_state_dict(torch.load('./model_output/best_ann_model_mnist_retrain_prune_{}_neurons_ratio_{}.pth'.format(del_type, ratio)))
                        mid_drop_acc_retrain = evaluate_ann(mid_drop_nn, test_data_loader)
                        print("Acc after dropping {} mid neurons and retraining: {}%".format(ratio, mid_drop_acc_retrain))    # Gradually deactivating mid neurons   
                        mid_drop_acc_retrain_list.append(mid_drop_acc_retrain)
                    df = pd.DataFrame(mid_drop_acc_retrain_list, columns=['Mid'])
                    df.to_excel('./pruning_results/mid_drop_acc_d2_w_retrain.xlsx', index=False) 
                else:
                    '''
                        drop hub neurons in Layer 1
                    '''
                    hub_drop_acc_retrain_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        print("Dropping hub neurons with ratio: {}".format(ratio))
                        hub_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * hubsize)).int()
                        rand_perm = torch.randperm(hubsize)
                        idx = I[weaksize+midsize:][rand_perm[:drop_num]]
                        hub_drop_nn.layer1.weight.data[:,idx] = torch.zeros_like(hub_drop_nn.layer1.weight.data[:,idx]).cuda()
                        for name, param in hub_drop_nn.named_parameters():
                            if "layer1" in name:
                                param.requires_grad = False   # fix the parameters of layer1
                        retrain(hub_drop_nn, model_type, del_type, ratio)
                        hub_drop_nn.load_state_dict(torch.load('./model_output/best_ann_model_mnist_retrain_prune_{}_neurons_ratio_{}.pth'.format(del_type, ratio)))
                        hub_drop_acc_retrain = evaluate_ann(hub_drop_nn, test_data_loader)
                        print("Acc after dropping {} hub neurons and retraining: {}%".format(ratio, hub_drop_acc_retrain))    # Gradually deactivating hub neurons   
                        hub_drop_acc_retrain_list.append(hub_drop_acc_retrain)
                    df = pd.DataFrame(hub_drop_acc_retrain_list, columns=['Hub'])
                    df.to_excel('./pruning_results/hub_drop_acc_d2_w_retrain.xlsx', index=False)  
    else:
        '''
            Pruning for SNN
        '''
        model.eval()  
        for (name, module) in model.named_modules():
            if name == "layer1.1":
                weight_size = module.weight.data.shape
                weights = module.weight.data  # torch.Size([1200, 784]), device = cuda
                # weight_sums = torch.sum(torch.abs(weights), dim=0) # definition 1
                weight_sums = torch.abs(torch.sum(weights, dim=0)) # definition 2
                B, I = torch.sort(weight_sums)
                weaksize = 261 
                midsize = 262  
                hubsize = 261 
                if del_type == "weak":    
                    '''
                        drop weak neurons in Layer 1
                    '''
                    weak_drop_acc_retrain_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        print("Dropping weak neurons with ratio: {}".format(ratio))
                        weak_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * weaksize)).int()
                        rand_perm = torch.randperm(weaksize)
                        idx = I[rand_perm[:drop_num]]
                        for (name, module) in weak_drop_nn.named_modules():
                            if name == "layer1.1":
                                module.weight.data[:,idx] = torch.zeros_like(module.weight.data[:,idx]).cuda()
                        for name, param in weak_drop_nn.named_parameters():
                            if "layer1.1" in name:
                                param.requires_grad = False  # fix the parameters of layer1
                        retrain(weak_drop_nn, model_type, del_type, ratio, exp_round) # retrain the network
                        weak_drop_nn.load_state_dict(torch.load('./model_output/best_snn_model_mnist_retrain_prune_{}_neurons_ratio_{}_exp{}.pth'.format(del_type, ratio, exp_round)))
                        weak_drop_acc_retrain = evaluate_snn(weak_drop_nn, test_data_loader)
                        print("Acc after dropping {} weak neurons and retraining: {}%".format(ratio, weak_drop_acc_retrain))    # Gradually deactivating weak neurons   
                        weak_drop_acc_retrain_list.append(weak_drop_acc_retrain)
                    df = pd.DataFrame(weak_drop_acc_retrain_list, columns=['Weak'])
                    df.to_excel('./pruning_results/snn_weak_drop_acc_d2_w_retrain_'+ exp_round +'.xlsx', index=False) 
                elif del_type == "mid":
                    '''
                        drop mid neurons in Layer 1
                    '''
                    mid_drop_acc_retrain_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        print("Dropping mid neurons with ratio: {}".format(ratio))
                        mid_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * midsize)).int()
                        rand_perm = torch.randperm(midsize)
                        idx = I[weaksize:][rand_perm[:drop_num]]
                        for (name, module) in mid_drop_nn.named_modules():
                            if name == "layer1.1":
                                module.weight.data[:,idx] = torch.zeros_like(module.weight.data[:,idx]).cuda()
                        for name, param in mid_drop_nn.named_parameters():
                            if "layer1.1" in name:
                                param.requires_grad = False   # fix the parameters of layer1
                        retrain(mid_drop_nn, model_type, del_type, ratio, exp_round)
                        mid_drop_nn.load_state_dict(torch.load('./model_output/best_snn_model_mnist_retrain_prune_{}_neurons_ratio_{}_exp{}.pth'.format(del_type, ratio, exp_round)))
                        mid_drop_acc_retrain = evaluate_snn(mid_drop_nn, test_data_loader)
                        print("Acc after dropping {} mid neurons and retraining: {}%".format(ratio, mid_drop_acc_retrain))    # Gradually deactivating mid neurons   
                        mid_drop_acc_retrain_list.append(mid_drop_acc_retrain)
                    df = pd.DataFrame(mid_drop_acc_retrain_list, columns=['Mid'])
                    df.to_excel('./pruning_results/snn_mid_drop_acc_d2_w_retrain_'+ exp_round +'.xlsx', index=False) 
                else:
                    '''
                        drop hub neurons in Layer 1
                    '''
                    hub_drop_acc_retrain_list = []
                    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        print("Dropping hub neurons with ratio: {}".format(ratio))
                        hub_drop_nn = model
                        drop_num = torch.round(torch.tensor(ratio * hubsize)).int()
                        rand_perm = torch.randperm(hubsize)
                        idx = I[weaksize+midsize:][rand_perm[:drop_num]]
                        for (name, module) in hub_drop_nn.named_modules():
                            if name == "layer1.1":
                                module.weight.data[:,idx] = torch.zeros_like(module.weight.data[:,idx]).cuda()
                        for name, param in hub_drop_nn.named_parameters():
                            if "layer1.1" in name:
                                param.requires_grad = False   # fix the parameters of layer1
                        retrain(hub_drop_nn, model_type, del_type, ratio, exp_round)
                        hub_drop_nn.load_state_dict(torch.load('./model_output/best_snn_model_mnist_retrain_prune_{}_neurons_ratio_{}_exp{}.pth'.format(del_type, ratio, exp_round)))
                        hub_drop_acc_retrain = evaluate_snn(hub_drop_nn, test_data_loader)
                        print("Acc after dropping {} hub neurons and retraining: {}%".format(ratio, hub_drop_acc_retrain))    # Gradually deactivating hub neurons   
                        hub_drop_acc_retrain_list.append(hub_drop_acc_retrain)
                    df = pd.DataFrame(hub_drop_acc_retrain_list, columns=['Hub'])
                    df.to_excel('./pruning_results/snn_hub_drop_acc_d2_w_retrain_'+ exp_round +'.xlsx', index=False)
 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prunning Framework for ANN/SNN')
    parser.add_argument('--net', type=str, default='ann', help='ann or snn')
    parser.add_argument('--neuron_type', type=str, default='weak', help='weak, mid or hub')
    parser.add_argument('--tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('--T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('--retrain', action='store_true', help='retrain the network or not')
    parser.add_argument('--exp_round', type=str, default='1', help='Experiment round, 1,2. The original experiment is run without indexing')
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
        model.load_state_dict(torch.load('./model_output/best_ann_model_mnist.pth'))
    else:    
        model = SNN(tau=args.tau).cuda()
        model.load_state_dict(torch.load('./model_output/best_snn_model_mnist_check.pth', map_location='cuda:0'))
    

    if args.retrain:
        pruning_with_retrain(model, args.net, args.exp_round, args.neuron_type)  # pruning with retraining
    else:
        base_pruning(model, args.net, args.exp_round, args.neuron_type) # pruning without retraining