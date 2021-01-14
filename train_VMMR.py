import os, time, commentjson
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

def prepare_data(config):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(config["input_shape"]),
            transforms.CenterCrop(config["input_shape"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config["input_shape"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join("./data", x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    assert (len(class_names) == config["num_classes"])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=config["batch_size"], shuffle=True,
                                                  num_workers=config["num_workers"]) for x in ['train', 'val']}

    return dataloaders, class_names

def plot_figures(metrics, title):
    plt.figure()
    for mode in metrics:
        plt.plot(metrics[mode], label = mode)
    plt.xlabel("Epochs")
    plt.ylabel(title.split(" ")[3])
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join("./VMMR_model", title.split(" ")[3]+".png"))

def train(dataloaders, classes, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config["model"] == "resnet18":
        model = models.resnet50(pretrained=True)
    elif config["model"] == "resnet50":
        model = models.resnet50(pretrained=True)
    elif config["model"] == "resnet101":
        model = models.resnet101(pretrained=True)
    else:
        print("Unrecognized model. Please pick resnet18, resnet50 or resnet101 in the configuration file")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=config["Lr"], momentum=config["momentum"])

    val_acc_history = []
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []

    best_acc = 0.0
    t0 = time.time()
    num_epochs = config["num_epochs"]

    for epoch in tqdm(range(num_epochs), position = 0):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print("Training ...")
            else:
                model.eval()  # Set model to evaluate mode
                print("Validation ...")

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join("./VMMR_model/best_model.pth"))
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

            # lr decay
            if epoch == config["lr_decay_iter"]:
                optimizer_ft.param_groups[0]["lr"] *= config["lr_decay_factor"]
        print()

    time_elapsed = time.time() - t0
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Visualization of the results
    plot_figures({"train": train_acc_history, "validation": val_acc_history},
                 "Evolution of the accuracy during the training")
    plot_figures({"train": train_loss_history, "validation": val_loss_history},
                 "Evolution of the loss during the training")

if __name__ == '__main__':
    with open("./config.json") as json_file:
        config = commentjson.load(json_file)
    print("Preparing the data ...")
    dataloaders, class_names = prepare_data(config)
    print("Model training initiation ...")
    print()
    train(dataloaders, class_names, config)
