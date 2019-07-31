import argparse
import os
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import numpy as np

# Parse command line arguments for train.py ---------------------------------------------------------------------------------------------------------
def parse_cmd_line_train():
    parser = argparse.ArgumentParser(add_help=True)
    
    # Add positional arguments
    parser.add_argument('data_dir', action='store', type=str)
    
    # Add optional arguments
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='')
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg19', 
                        choices=['vgg19', 'resnet152', 'densenet121'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=1024)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=8)
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False)

    return parser.parse_args()

# Parse command line arguments for predict.py -------------------------------------------------------------------------------------------------------
def parse_cmd_line_predict():
    parser = argparse.ArgumentParser(add_help=True)
    
    # Add positional arguments
    parser.add_argument('path_to_image', action='store', type=str)
    parser.add_argument('checkpoint', action='store', type=str)
    
    # Add optiional arguments
    parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False)
    
    return parser.parse_args()
    
# Get Data loaders ------------------------------------------------------------------------------------------------------------------------------
def get_data_loaders(data_dir):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                              data_transforms['train']),
                      'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'),
                                              data_transforms['valid']),
                      'test': datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                              data_transforms['test'])
    }
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    dataset_sizes = {'train': len(image_datasets['train']),
                     'valid': len(image_datasets['valid']),
                     'test': len(image_datasets['test'])
    }

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],
                                                  batch_size=64,
                                                  shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                                  batch_size=32,
                                                  shuffle=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'],
                                                  batch_size=32,
                                                  shuffle=True)
    }

    return dataloaders, dataset_sizes, class_to_idx

# Set device type ------------------------------------------------------------------------------------------------------------------
def set_device_type(request_gpu, model_activity):
    
    if request_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('{} with CUDA\n'.format(model_activity))
        else:
            device = torch.device('cpu')
            print('Unable to access CUDA, {} with CPU\n'.format(model_activity.lower()))
    else:
        device = torch.device('cpu')
        print('{} with CPU\n'.format(model_activity))
        
    return device
    
# Save checkpoint --------------------------------------------------------------------------------------------------------------------
def save_nn(trained_nn, class_to_idx, arch, classifier_layers, optimizer, learning_rate, total_epochs, checkpoint_path):
    
    trained_nn.class_to_idx = class_to_idx
    checkpoint_dict = {'arch': arch,
                       'classifier': trained_nn.classifier,
                       'class_to_idx': trained_nn.class_to_idx,
                       'hidden_layers': classifier_layers,
                       'optimizer': optimizer,
                       'lr': learning_rate,
                       'state_dict': trained_nn.state_dict(),
                       'total_epochs': total_epochs
                      }
    torch.save(checkpoint_dict, checkpoint_path)
    print('Network saved successfully saved as a checkpoint at {}'.format(checkpoint_path))

# Load checkpoint --------------------------------------------------------------------------------------------------------------------------------
def load_nn(checkpoint_path):
    ''' Loads a checkpointed model
    '''
    print('Loading network...')
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.state_dict = checkpoint['state_dict']
#     model.to(device)
    print('Successfully loaded the network.')
    
    return model

# Preprocess image --------------------------------------------------------------------------------------------------------------------------------
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path).convert('RGB')
    
    # Resize to 256 pixels
    width = image.width
    height = image.height
    aspect_ratio = width / height

    if width >= height:
        image = image.resize((int(256 * aspect_ratio), 256))
    else:
        image = image.resize((256, int(256 / aspect_ratio)))

    # Center crop 224x224 portion
    left_margin = (image.width - 224) / 2
    right_margin = left_margin + 224
    bottom_margin = (image.height - 224) / 2
    top_margin = bottom_margin + 224
    cropped_image = image.crop((left_margin, bottom_margin, 
                        right_margin, top_margin))
    
    # Normalize image and reorder color channel dimension
    cropped_image_np = np.array(cropped_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])

    normal_image = (cropped_image_np - mean) / std_dev
    normal_image = normal_image.transpose((2, 0, 1))
    
    return normal_image 

# Print inference results -----------------------------------------------------------------------------------------------------------------------
def print_inference_results(top_k_probs, top_k_labels, k):
    print()
    print('TOP {} INFERENCE RESULTS'.format(k))
    print('=' * 40)
    print('{:^20}{:^20}'.format('Class', 'Class Probability'))
    print('=' * 40)
    
    for index in range(len(top_k_labels)):
        print('{:<20}{:>20.2%}'.format(top_k_labels[index], top_k_probs[index]))

# *****************************************************************************************************************

# if __name__ == "__main__":
