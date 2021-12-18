import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
# from torchsummary import summary
# print(model, (channels, input, shape))

from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_writer = SummaryWriter('all_runs/runs_4/train_fish')
test_writer = SummaryWriter('all_runs/runs_4/test_fish')

means, stds = [0.37674451, 0.4075019,  0.4550148], [0.24352683, 0.18432788, 0.16501085]

IMG_SIZE = 256

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(size=IMG_SIZE*2),
        transforms.Normalize(means, stds),
        transforms.GaussianBlur(kernel_size=5),
        # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), 
                            # scale=(0.75, 1.25), shear=0.2),
        # transforms.RandomRotation(degrees=180, fill=(0,)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((IMG_SIZE, IMG_SIZE))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(size=IMG_SIZE*2),
        transforms.Normalize(means, stds),
        transforms.Resize((IMG_SIZE, IMG_SIZE))
    ])
}

data_dir = './NA_Fish_Dataset'
sets = ['train', 'test']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x]) for x in sets}

data_loaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0, drop_last=True)
                            for x in sets}

dataset_sizes = {x: len(image_datasets[x]) for x in sets}
class_names = image_datasets['train'].classes

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, len(class_names))
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.95)

# Plot predictions

def matplotlib_imshow(img, one_channel=False):
    img = transforms.ToTensor()(img)
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def get_class_img():
    # for each class, get random image 
    test = image_datasets['test']

    # starting index of every class
    start_idx = np.unique(test.targets, return_index=True)[1]
    start_idx = np.append(start_idx, len(test))

    imgs = []
    for i in range(len(start_idx)-1):
        rand_idx = np.random.choice(np.arange(start_idx[i], start_idx[i+1]))
        img_path = test.samples[rand_idx][0]
        img = Image.open(img_path)
        imgs.append(img)
    return imgs


def plot_classes_preds(model):
    # for each class, get random image, get predictions, and plot
    imgs = get_class_img()

    fig = plt.figure(figsize=(12, 12))
    num_classes = 9
    for idx in np.arange(num_classes):
        ax = fig.add_subplot(3, 3, idx+1, xticks=[], yticks=[])
        img = imgs[idx]
        matplotlib_imshow(img)

        img_proc = data_transforms['test'](img).to(device).unsqueeze(dim=0)
        model.eval()
        with torch.no_grad():
            output = model(img_proc)
            probs = F.softmax(output, dim=1)
            best_pred = probs.max(1, keepdim=True)

            prob = best_pred[0].item()
            pred_idx = best_pred[1].item()
            
            pred_class = image_datasets['test'].classes[pred_idx]
            true_class = image_datasets['test'].classes[idx]

            ax.set_title("Predicted: {}-({:.2f}%)\nTrue: {}".format(
                        pred_class, prob * 100, true_class
                        ),
                        color=("green" if pred_class == true_class else "red")
                        )
    return fig


def train(model, epoch):
    start = time.time()
    model.train()

    running_loss = 0
    correct = 0
    for i, (x, y) in enumerate(data_loaders['train']):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        # step_lr_scheduler.step()

        running_loss += loss.item()
        correct += outputs.max(1)[1].eq(y).sum()
        step = epoch * len(data_loaders['train']) + i+1
        train_writer.add_scalar('Loss',
                                running_loss / (i + 1),
                                step)

        train_writer.add_scalar('Accuracy',
                                correct / (i+1) / len(y),
                                step)

        train_writer.add_figure('Prediction per Class',
                        plot_classes_preds(model),
                        global_step=step)
    batch_loss = running_loss / len(data_loaders['train'])
    batch_accuracy = correct / len(data_loaders['train']) / len(y)
    end = time.time()
    print(f"[Training - {end-start}s] Epoch {epoch+1}: Loss - {batch_loss}, Accuracy - {batch_accuracy}")

def test(model, epoch):
    model.eval()
    with torch.no_grad():
        start = time.time()
        test_loss = 0
        test_correct = 0
        for i, (x, y) in enumerate(data_loaders['test']):
            x = x.to(device)
            y = y.to(device)
            pred = model(x) 
            test_loss += criterion(pred, y).item()
            test_correct += pred.max(1)[1].eq(y).sum()

        # Match step with train
        step = epoch * len(data_loaders['train']) + len(data_loaders['test'])

        test_writer.add_scalar('Loss',
                                test_loss / (i+1),
                                step)
        test_writer.add_scalar('Accuracy',
                                test_correct / (i+1) / len(y),
                                step)
        batch_loss = test_loss / len(data_loaders['test'])
        batch_accuracy = test_correct / len(data_loaders['test']) / len(y)
        end = time.time()
        print(f"[Testing - {end-start}s] Epoch {epoch+1}: Loss - {batch_loss}, Accuracy - {batch_accuracy}")


n_epochs = 50
start = time.time()
for i in range(n_epochs):
    train(model, i)
    test(model, i)
end = time.time()
print(f"Total Traning time: {end-start}")

train_writer.close()
test_writer.close()

torch.save(model.state_dict(), 'result/v4.pth')