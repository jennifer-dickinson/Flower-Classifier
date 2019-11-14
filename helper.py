import argparse
import torch

def get_input_args():
    parser = argparse.ArgumentParser(description = "Train a neural network")
    parser.add_argument('data_directory', type = str)
    parser.add_argument('--save_dir', type = str, default = ".")
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--hidden_units', type = int, default = 512)
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--arch', type = str, default = "densenet121")
    parser.add_argument('--gpu', action='store_true')

    return parser.parse_args()

def validation(model, testloader, criterion, device):
    training = model.training
    if(not training): model.eval()
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
            
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        first = False
    if training: model.train()
    return test_loss, accuracy