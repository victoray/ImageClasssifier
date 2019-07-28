import argparse
from model import create_model, save_model
import torch
from torchvision import transforms, datasets, models
from train_args import train_args


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=64)

    return trainloader, testloader, validationloader, train_datasets



def train(model, device, epochs, criterion, optimizer, trainloader, validationloader):

    model.to(device)
    for i in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        else:
            test_losses = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validationloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model.forward(images)

                    test_losses += criterion(output, labels).item()

                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()

            print(f'Training Loss: {running_loss / len(trainloader):.3f} '
                  f'Validation Loss: {test_losses / len(validationloader):.3f} '
                  f'Validation Accuracy: {(accuracy / len(validationloader)) * 100:.3f}%')
    return model

def test(model, device, testloader):
    model.to(device)
    model.eval()
    accuracy = 0
    test_losses = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        else:
            print(f'Accuracy: {(accuracy / len(testloader)) * 100:.3f}% '
                  f'Test Losses: {test_losses / len(testloader):.3f}')



if __name__ == '__main__':
    input_args = train_args()

    arch = input_args.arch
    data_dir = input_args.data_dir
    epochs = input_args.epochs
    gpu = input_args.gpu
    hidden_units = input_args.hidden_units
    learning_rate = input_args.learning_rate
    save_dir = input_args.save_dir

    trainloader, testloader, validationloader, train_datasets= load_data(data_dir)

    model, criterion, optimizer, parameters = create_model(arch, lr=learning_rate, hidden_units=hidden_units)
    device = 'cuda' if gpu else 'cpu'

    print(f"Training Model with {arch}.......\n")
    trained_model = train(model=model, device=device, epochs=epochs, criterion=criterion,
          optimizer=optimizer, trainloader=trainloader, validationloader=validationloader)

    print("Training Complete\n")
    print("Testing Model")
    test(trained_model, device, testloader)
    print("Testing Complete\n")

    save_model(arch + save_dir, train_datasets, trained_model, parameters)
    
    print("Model Saved Succesfully")



