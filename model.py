import torch
from torch import optim, nn
from torchvision import transforms, datasets, models
import scipy

archs = dict()
archs['resnet18'] = [models.resnet18(pretrained=True), 512]
archs['alexnet'] = [models.alexnet(pretrained=True), 9216]
archs['vgg16'] = [models.vgg16(pretrained=True), 25088]
archs['densenet'] = models.densenet121(pretrained=True), 1024
# archs['squeezenet'] = [models.squeezenet1_0(pretrained=True), 512]
# archs['inception'] = models.inception_v3(pretrained=True), 2048
# archs['googlenet'] = models.googlenet(pretrained=True), 1024
# archs['shufflenet'] = models.shufflenet_v2_x1_0(pretrained=True), 1024
# archs['mobilenet'] = models.mobilenet_v2(pretrained=True), 1280
# archs['resnext50_32x4d'] = models.resnext50_32x4d(pretrained=True), 2048


fc = []
classifier = []

for name, model in archs.items():
    try:
        model[0].fc
        fc.append(name)
    except:
        pass

    try:
        model[0].classifier
        classifier.append(name)
    except:
        pass


def create_model(arch, lr=0.003, hidden_units=256):
    """

    :param arch: The pretrained model to create the training model
    :param lr: The learning rate
    :param hidden_units: The number of hidden units

    :return: model: The custom training model
    :return: criterion: Measure of training Loss
    :return: optimizer: The training optimizer
    :return: parameters: A dictionary of model parameters.
    """
    architectures = [arch for arch in archs.keys()]

    if arch not in architectures:
        raise Exception(f"Invalid Architecture. Supported archs are {', '.join(architectures)}")

    model = archs[arch][0]

    for param in model.parameters():
        param.requires_grad = False

    input = archs[arch][1]

    criterion = nn.NLLLoss()

    network = nn.Sequential(nn.Linear(input, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1))

    if arch in fc:
        model.fc = network
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    elif arch in classifier:
        model.classifier = network
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    parameters = {'model': arch, 'input': input, 'hidden_units': hidden_units, 'output': 102}

    return model, criterion, optimizer, parameters


def save_model(save_dir, train_datasets, model, parameters):
    """

    :param save_dir: location to save the model
    :param train_datasets: processed images for training.
    :param model: The trained model
    :param parameters: The parameters for the model
    :return: None
    """
    checkpoint = {'input_size': parameters['input'],
                  'output_size': parameters['output'],
                  'hidden_layer': parameters['hidden_units'],
                  'arch': parameters['model'],
                  'class_index': train_datasets.class_to_idx
                  }

    if parameters['model'] in fc:
        checkpoint['state_dict'] = model.fc.state_dict()
    else:
        checkpoint['state_dict'] = model.classifier.state_dict()

    torch.save(checkpoint, save_dir)


def load_checkpoint(path):
    """
    Loads  the saved model
    :param path: path to the checkpoint
    :return: model: loaded trained model
    """
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    arch = checkpoint['arch']

    model = archs[arch][0]

    network = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer']),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(checkpoint['hidden_layer'], checkpoint['output_size']),
                            nn.LogSoftmax(dim=1))
    if arch in fc:
        model.fc = network
        model.fc.load_state_dict(checkpoint['state_dict'])
    elif arch in classifier:
        model.classifier = network
        model.classifier.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_index']

    return model
