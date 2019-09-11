# import numpy as np
import PIL.Image as Image
import torch
from classifier.predict_args import predict_args
from classifier.model import load_checkpoint
from torchvision import transforms, datasets, models
import json


def process_image(image):
    """
       Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns an Numpy array
        Arguments:
            image - image to be processed
        Returns:
            np_image - a numpy representation of the image
    """

    # TODO: Process a PIL image for use in a PyTorch model

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_t = preprocess(image)
    np_image = image_t.numpy()
    
    return np_image


def predict(image_path, model, device, topk=5):
    '''
        Predict the class (or classes) of an image using a trained deep learning model.
        Arguments:
            image_path - path to the image file
            model - a trained model
            device - The device for training can be cuda or cpu
            topk - number of top classes to return. Default is 5
        Returns:
             top_p - a list of the top probabilities
             classes - a list of top indexes

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
    classes = list()

    for label in top_class.tolist()[0]:
        classes.append(class_to_idx_inverted[label])

    top_p = top_p.tolist()[0]

    return top_p, classes


def cat_to_name(category_names):
    """

    :param
        category_names: a path to the json name matching
    :return:
        categories: returns a dictionary with the names
    """
    with open(category_names, 'r') as f:
        categories = json.load(f)

    return categories


if __name__ == '__main__':
    input_args = predict_args()

    image_path = input_args.image_path
    checkpoint = input_args.checkpoint
    top_k = input_args.top_k
    category_names = input_args.category_names
    gpu = input_args.gpu

    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'

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
