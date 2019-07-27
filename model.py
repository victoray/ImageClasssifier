import torch
from torch import optim, nn
from torchvision import transforms, datasets, models
import scipy

archs = {}
archs['resnet18'] = [models.resnet18(pretrained=True), 512]
archs['alexnet'] = [models.alexnet(pretrained=True), 9216]
archs['vgg16'] = [models.vgg16(pretrained=True), 25088]
archs['squeezenet'] = [models.squeezenet1_0(pretrained=True), 512]
archs['densenet'] = models.densenet161(pretrained=True), 2208
archs['inception'] = models.inception_v3(pretrained=True), 2048
archs['googlenet'] = models.googlenet(pretrained=True), 1024
archs['shufflenet'] = models.shufflenet_v2_x1_0(pretrained=True), 1024
archs['mobilenet'] = models.mobilenet_v2(pretrained=True), 1280
archs['resnext50_32x4d'] = models.resnext50_32x4d(pretrained=True), 2048

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
    architectures = [arch for arch in archs.keys()]

    if arch not in architectures:
        raise Exception(f"Invalid Architecture. Supported archs are {', '.join(architectures)}")

    model = archs[arch][0]

    for param in model.parameters():
        param.requires_grad = False

    input = archs[arch][1]

    criterion = nn.NLLLoss()

    if arch in fc:
        model.fc = nn.Sequential(nn.Linear(input, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))

        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    elif arch in classifier:
        model.classifier = nn.Sequential(nn.Linear(input, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))

        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)


    return model, criterion, optimizer
