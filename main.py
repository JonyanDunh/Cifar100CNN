import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
data_path = '/data/cifar_100/'
transformed_cifar100_val = datasets.CIFAR100(data_path, train=False, download=False, transform=transforms.ToTensor())
transformed_cifar100 = datasets.CIFAR100(data_path, train=True, download=False, transform=transforms.ToTensor())

cifar2 = [(img.cuda(), torch.zeros(1, 100).scatter_(1, torch.tensor([label]).unsqueeze(1), 1.0).cuda(), label) for
          img, label in transformed_cifar100]
cifar2_val = [(img.cuda(), torch.zeros(1, 100).scatter_(1, torch.tensor([label]).unsqueeze(1), 1.0).cuda(), label) for
              img, label in transformed_cifar100_val]
# label_name =["airplane",
#              "automobile",
#              "bird",
#              "cat",
#              "deer",
#              "dog",
#              "frog",
#              "horse",
#              "ship",
#              "truck"]
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(cifar2[i][0].cpu().permute(1, 2, 0), cmap=plt.cm.binary)
#     plt.xlabel(label_name[cifar2[i][1].argmax(1).cpu()])
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4_batchnorm = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5_batchnorm = nn.BatchNorm2d(num_features=128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6_batchnorm = nn.BatchNorm2d(num_features=128)
        self.fc1 = nn.Linear(128 * 4 * 4, 400)
        self.fc2 = nn.Linear(400, 100)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv1_batchnorm(out)# 32*32*32

        out = self.conv2(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, 2)# 32*16*16
        out = self.conv2_batchnorm(out)

        out = self.conv3(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, 2)
        out = torch.relu(out)# 64*8*8
        out = self.conv3_batchnorm(out)

        out = self.conv4(out)
        out = torch.relu(out)# 64*8*8
        out = self.conv4_batchnorm(out)

        out = self.conv5(out)
        out = torch.relu(out)# 128*8*8
        out = self.conv5_batchnorm(out)

        out = self.conv6(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, 2)# 128*4*4
        out = self.conv6_batchnorm(out)

        out = out.view(-1, 128 * 4 * 4)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)

        return out


model = Net().to(device=torch.device('cuda'))
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=4096, shuffle=True)


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, one_hot_labels, _ in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, one_hot_labels.squeeze_(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))
        validate(model, train_loader, val_loader)


val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=4096, shuffle=False)


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, _, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]

                correct += int((predicted == labels.cuda()).sum())
        print("Accuracy {}: {:.4f}".format(name, correct / total))

#torch.save(model.state_dict(),'/data/saved_tensor/Cifar-100/Cifar-100.pt')


validate(model, train_loader, val_loader)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss().cuda()
training_loop(
    n_epochs=25,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
)
validate(model, train_loader, val_loader)
