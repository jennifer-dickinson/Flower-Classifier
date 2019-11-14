import helper
import Image
import model as Model
import torch

# ## Get arguments
args = helper.get_predict_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

# ## Load model
model = Model.load(args.checkpoint)

# ## Process Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    return helper.standard_transforms(Image.open(image))

# ## Map classes to indices

class_idx = dict((key, cl) for (cl, key) in model.class_to_idx.items())


# ## Predict Function

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image.unsqueeze_(0)
    image = image.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)

    probs, cl = ps.topk(topk)
    probs, cl = probs.to('cpu'), cl.to('cpu')
    probs = probs.reshape(-1).numpy()
    cl = cl.reshape(-1).numpy()
    cl = [cat_to_name[class_idx[c]] for c in cl]
    return probs, cl


import json
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

probs, cls = predict(args.image_path, model)

print(probs, cls)