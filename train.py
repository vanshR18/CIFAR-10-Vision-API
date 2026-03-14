import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from model import CNN

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)

model = CNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(25):

    for images, labels in trainloader:

        optimizer.zero_grad()       # Optimizer

        outputs = model(images)

        loss = criterion(outputs, labels)      # Loss function

        loss.backward()         # Backward loss

        optimizer.step()        # optimizer step by step

print("training complete")

torch.save(model.state_dict(), "saved_model.pth")
