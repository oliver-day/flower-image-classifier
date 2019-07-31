import torch
import torch.nn.functional as F
import json

from utility import (
    parse_cmd_line_predict,
    load_nn,
    print_inference_results,
    process_image,
    set_device_type
)

# Returns the top k probabilites and respective labels for a provided image -----------------------------------------------------------------------
def predict(image_path, model, category_names, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Process given image
    processed_image_np = process_image(image_path)
    
    # Convert numpy array into a Torch tensor
    if device == torch.device('cuda'):
        image_tensor_type = torch.cuda.FloatTensor
    else:
        image_tensor_type = torch.FloatTensor
        
    processed_image_tensor = torch.from_numpy(processed_image_np).type(image_tensor_type)
    
    # Indicate a batch size of 1, shape of tensor is now (batch_size, rgb, width, height)
    processed_image_tensor.unsqueeze_(0)

    # Get ouput from model
    model.to(device)
    output = model(processed_image_tensor)
    
    # Get top k probabilities and indices, convert PyTorch autograd.Variable to numpy array
    top_k_probs, top_k_indices = torch.topk(input=(F.softmax(output, dim=1)), k=topk, sorted=True)
    top_k_probs = [probability.item() for probability in top_k_probs[0].data]
    
    # Invert class_to_idx dictionary
    idx_to_class = { idx:class_name for class_name, idx in model.class_to_idx.items() }
    
    # Get class_to_label dictionary
    with open(category_names, 'r') as f:
        class_to_label = json.load(f)
    
    # Get top k labels
    top_k_classes = [idx_to_class[index.item()] for index in top_k_indices[0]]
    top_k_labels = [class_to_label[index] for index in top_k_classes]
    
    return top_k_probs, top_k_labels

# Loads a pretrained network and predicts the top k class probabilities for an image. --------------------------------------------------------------
def main():
    # Get command line arguments
    cmdl_args = parse_cmd_line_predict()
    
    # Load pretrained network
    loaded_network = load_nn(cmdl_args.checkpoint)
    
    # Set device type for network
    device = set_device_type(cmdl_args.gpu, 'Predicting')
    loaded_network.to(device)
    
    # Predict top k labels and their probabilities
    top_k_probs, top_k_labels = predict(cmdl_args.path_to_image, loaded_network, cmdl_args.category_names, device, cmdl_args.top_k )

    # Print results    
    print_inference_results(top_k_probs, top_k_labels, cmdl_args.top_k)

# =================================================================================================================
    
if __name__ == "__main__":
    main() 
