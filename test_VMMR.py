import os, time, argparse, commentjson, glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser()
parser.add_argument('--Test_Data_DIR', default='./data/test')
parser.add_argument('--config_Path', default='./config.json')
parser.add_argument('--Model_Path', default='./results/best_model.pth')

def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_names = []
    with open("./data/class_names.txt", 'r') as f:
        for line in f:
            class_names.append(line.rstrip('\n'))

    if args.config_Path["model"] == "resnet18":
        model = models.resnet50(pretrained=True)
    elif args.config_Path["model"] == "resnet50":
        model = models.resnet50(pretrained=True)
    elif args.config_Path["model"] == "resnet101":
        model = models.resnet101(pretrained=True)
    else:
        print("Unrecognized model. Please pick resnet18, resnet50 or resnet101 in the configuration file")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    model.load_state_dict(torch.load(args.Model_Path))

    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.config_Path["input_shape"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_images = glob.glob(args.Test_Data_DIR+"/*.*")
    with torch.no_grad():
        model.eval()
        predictions = []
        for test_img in test_images:
            img = plt.imread(test_img, 0)
            img = data_transforms(transforms.ToPILImage()(TF.to_tensor(img))).reshape((1, 3, args.config_Path["input_shape"], args.config_Path["input_shape"]))
            img = img.to(device).float()
            output = model(img)
            pred = np.argmax(output.cpu().numpy())
            print("Predicted label for test image {}: {}".format(test_img.split("\\")[-1], class_names[pred]))

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_Path) as json_file:
        args.config_Path = commentjson.load(json_file)
    print("Generating predictions ...")
    test(args)