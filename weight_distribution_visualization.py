import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from ann import ANN
from snn import SNN
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import ipdb

# set the font of the label
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '14'

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

def weight_distribution(model, net_type):
    if net_type=='ann':  # Done
        for (name, module) in model.named_modules():
            if name == "layer1":
                weight_size = module.weight.data.shape
                weights = module.weight.data
                weight_sums = torch.abs(torch.sum(weights, dim=0))  # without 0/1 binarization  
                B, I = torch.sort(weight_sums)
                # ipdb.set_trace()
                plt.hist(B.cpu().numpy(), bins=20, facecolor='lightgrey', edgecolor='w', density=True)
                plt.xlim((0,55.00))
                plt.ylim((0.00, 0.16))
                plt.savefig('./figures/weight_hist_plot_{}.png'.format(net_type), dpi=700, bbox_inches='tight', pad_inches=0)
    else:
        for (name, module) in model.named_modules():
            
            if name == "layer1.1":  # layer1.1
                weight_size = module.weight.data.shape
                weights = module.weight.data
                weight_sums = torch.abs(torch.sum(weights, dim=0))
                B, I = torch.sort(weight_sums)
                # ipdb.set_trace()
                plt.hist(B.cpu().numpy(), bins=20, facecolor='lightgrey', edgecolor='w', density=True)
                plt.xlim((0, 55))
                plt.ylim((0.00, 0.16))
                plt.savefig('./figures/weight_hist_plot_{}.png'.format(net_type), dpi=700, bbox_inches='tight', pad_inches=0)
      
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weight Distribution Plot for ANN/SNN')
    parser.add_argument('--net', type=str, default='ann', help='ann or snn')
    parser.add_argument('--tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    args = parser.parse_args()  
    
    if args.net == 'ann':
        model = ANN().cuda()
        model.load_state_dict(torch.load('./model_output/best_ann_model_mnist.pth'))
    else:
        model = SNN(tau=args.tau).cuda()
        model.load_state_dict(torch.load('./model_output/best_snn_model_mnist.pth', map_location='cuda:0'))
    weight_distribution(model, net_type=args.net)




# ipdb> for (name, module) in model.named_modules():                                                                                            
#         print(name)                                                                                                                           
                                                                                                                                              
                                                                                                                                              
# layer1                                                                                                                                        
# layer1.0                                                                                                                                      
# layer1.1                                                                                                                                      
# layer1.2                                                                                                                                      
# layer1.2.surrogate_function                                                                                                                   
# layer2                                                                                                                                        
# layer2.0                                                                                                                                      
# layer2.1                                                                                                                                      
# layer2.2                                                                                                                                      
# layer2.2.surrogate_function                                                                                                                   
# layer3                                                                                                                                        
# layer3.0                                                                                                                                      
# layer3.1                                                                                                                                      
# layer3.2                                                                                                                                      
# layer3.2.surrogate_function                                                                                                                   
# layer4        



# ipdb> print(module)
# Sequential(
#   (0): Flatten(start_dim=1, end_dim=-1, step_mode=s)
#   (1): Linear(in_features=784, out_features=1200, bias=False)
#   (2): LIFNode(
#     v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s, backend=torch, tau=2.0
#     (surrogate_function): ATan(alpha=2.0, spiking=True)
#   )
# )
# ipdb> name
# 'layer1'
 
#ipdb> print(name)
# layer1.0
# ipdb> print(module)
# Flatten(start_dim=1, end_dim=-1, step_mode=s)

# ipdb> print(name)
# layer1.1
# ipdb> print(module)
# Linear(in_features=784, out_features=1200, bias=False)

# ipdb> print(name)
# layer1.2
# ipdb> print(module)
# LIFNode(
#   v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s, backend=torch, tau=2.0
#   (surrogate_function): ATan(alpha=2.0, spiking=True)
# )

# ipdb> print(name)
# layer1.2.surrogate_function
# ipdb> print(module)
# ATan(alpha=2.0, spiking=True)