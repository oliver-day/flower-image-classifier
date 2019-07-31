import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import models

from collections import OrderedDict
from utility import (
    get_data_loaders,
    parse_cmd_line_train,
    save_nn,
    set_device_type
)

# Trains network -----------------------------------------------------------------------------------------------------------------
def train_network(model, total_epochs, criterion, optimizer, dataloaders, dataset_sizes, device):
    model.to(device)
    print_every = 30

    for epoch in range(total_epochs):
        
        print("Epoch: {}/{}".format(epoch+1, total_epochs))
        print("-" * 25)

        for phase in ['train', 'valid']:
            running_loss = 0
            running_correct = 0

            for ii, (inputs, labels) in enumerate(dataloaders[phase]):
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):             
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Get the maximum value in one dimension
                    max_tensor, predicted_classes = torch.max(outputs, 1)
                    
                    # Backward pass and optimization step
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += (predicted_classes == labels.data).sum().item()

            print("{} Loss: {:.5f}".format(phase.capitalize(), running_loss / dataset_sizes[phase]))
            print("{} Accuracy: {:.3f}%".format(phase.capitalize(), 100 * running_correct / dataset_sizes[phase]))
            print()
            
        print()
        print()
                    
    return model 
    
# Trains a new network on a image dataset and saves it as a checkpoint ---------------------------------------------------------------
def main():
    # Get command line arguments
    cmdl_args = parse_cmd_line_train()

    # Get dataloaders and their respective data set sizes
    dataloaders, dataset_sizes, class_to_idx = get_data_loaders(cmdl_args.data_dir)
    
    # Set Hyperparameters
    arch = cmdl_args.arch
    learning_rate = cmdl_args.learning_rate
    total_epochs = cmdl_args.epochs
    total_num_of_classes = 102
    classifier_layers = []
    
    # Set hidden layrs based off given arch
    if arch == 'vgg19':
        classifier_layers = [25088, 4096, cmdl_args.hidden_units]
    elif arch == 'resnet152':
        classifier_layers = [2048, 1000, cmdl_args.hidden_units]
    elif arch == 'densenet121':
        classifier_layers = [1024, 1000, cmdl_args.hidden_units]
    
    # Load pretrained network and freeze parameters
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    # Define and set classifier
    classifier = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(classifier_layers[0], classifier_layers[1])),
                    ('relu', nn.ReLU()),
                    ('dropout', nn.Dropout(p=0.5)),
                    ('fc2', nn.Linear(classifier_layers[1], classifier_layers[2])),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(classifier_layers[2], total_num_of_classes)),
                    ('output', nn.LogSoftmax(dim=1))
                    ]))
    model.classifier = classifier
    
    # Set device type for model
    device = set_device_type(cmdl_args.gpu, 'Training')
    
    # Set loss and optimizer functions
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train network
    trained_network = train_network(model, total_epochs, criterion, optimizer, dataloaders, dataset_sizes, device)
    
    # Save network
    save_nn(trained_network, class_to_idx, arch, classifier_layers, optimizer, learning_rate, total_epochs, cmdl_args.save_dir)
    
# =================================================================================================================
    
if __name__ == "__main__":
    main()
