import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import torch.utils.data as data
from ann import ANN
from tqdm import tqdm
import logging
import random
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('./log_file/ann_training_log.txt')) # save the log to file

# load the MNIST
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


model = ANN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def evaluate_model(model, test_data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.view(images.shape[0], -1).cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train_model(model, train_data_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_data_loader):
            images = images.view(images.shape[0], -1).cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(train_data_loader)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

        # save the model with the best accuracy
        accuracy = evaluate_model(model, test_data_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), './model_output/best_ann_model_mnist.pth')
            logger.info(f'Saved the model with the best accuracy: {best_accuracy:.4f}')

if __name__ == '__main__':
    # training
    num_epochs = 10

    logger.info('Start training.')

    train_model(model, train_data_loader, criterion, optimizer, num_epochs)

    logger.info('Training complete. Enjoy!')

    # evaluate the model
    final_accuracy = evaluate_model(model, test_data_loader)
    logger.info(f'Final accuracy of the network on the test images: {final_accuracy}%')