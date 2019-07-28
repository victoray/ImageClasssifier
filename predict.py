import numpy as np
import PIL.Image as Image
import torch
from predict_args import predict_args
from model import load_checkpoint
from torchvision import transforms, datasets, models
import json


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
#     img = image.resize((256, 256))
#     width, height = img.size

#     left = (width - 224) / 2
#     top = (height - 224) / 2
#     right = (width + 224) / 2
#     bottom = (height + 224) / 2

#     # Crop the center of the image
#     img_crop = img.crop((left, top, right, bottom))

#     np_image = np.array(img_crop) / 255
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     np_image = (np_image - mean) / std
#     np_trans = np_image.transpose(2, 0, 1)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_t = preprocess(image)
    np_trans = image_t.numpy()
    
    return np_trans


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)

    image = torch.from_numpy(image).float().to(device)
    image.unsqueeze_(0)
    image.to(device)
    
    image.requires_grad_(False)
    
    model.to(device)
    output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk)

    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    for label in top_class.tolist()[0]:
        mapped_classes.append(class_to_idx_inverted[label])

    top_p = top_p.tolist()[0]

    return top_p, mapped_classes


def cat_to_name(category_names):
    with open('cat_to_name.json', 'r') as f:
        category_names = json.load(f)

    return category_names


if __name__ == '__main__':
    input_args = predict_args()

    image_path = input_args.image_path
    checkpoint = input_args.checkpoint
    top_k = input_args.top_k
    category_names = input_args.category_names
    gpu = input_args.gpu

    device = 'cuda' if gpu else 'cpu'

    model = load_checkpoint(checkpoint)

    probabilities, classes = predict(image_path, model,device, top_k)

    max_p = probabilities.index(max(probabilities))
    max_label = classes[max_p]

    category_names = cat_to_name(category_names)
    
    labels = [category_names[str(i)] for i in classes]

    print('Name: ===>', category_names[max_label], '    Probability: ===>', probabilities[max_p])
    
    print("\nTop Classes and Probabilities")
    for a, b in zip(labels, probabilities):
        print(f"Name: {a}   Probability: {b}")
