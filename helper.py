import argparse
from torch import exp, FloatTensor
from torchvision import transforms

def get_train_args():
    parser = argparse.ArgumentParser(description = "Train a neural network")
    parser.add_argument('data_directory', type = str)
    parser.add_argument('--save_dir', type = str, default = ".")
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--hidden_units', type = int, default = 512)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--arch', type = str, default = "densenet121")
    parser.add_argument('--gpu', action='store_true')

    return parser.parse_args()

standard_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]
                                    )])


def validation(model, testloader, criterion, device):
    training = model.training
    if(not training): model.eval()
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = exp(output)
            
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(FloatTensor).mean()

    if training: model.train()

    return test_loss, accuracy

def get_predict_args():
    parser = argparse.ArgumentParser(description = "Predict an image from a pretrained neural network")
    parser.add_argument('image_path', type = str)
    parser.add_argument('checkpoint', type = str)
    parser.add_argument('--top_k', type = int, default = 3)
    parser.add_argument('--category_names', type = str)
    parser.add_argument('--gpu', action='store_true') 

    return parser.parse_args()