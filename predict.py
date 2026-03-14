import torch
from torchvision import transforms
from PIL import Image

from model import CNN

model = CNN()
model.load_state_dict(torch.load("saved_model.pth"))
model.eval()

classes = [
'airplane','automobile','bird','cat',
'deer','dog','frog','horse','ship','truck'
]

transform = transforms.Compose([
transforms.Resize((32,32)),
transforms.ToTensor()
])

def predict_image(image):

    img = transform(image).unsqueeze(0)

    output = model(img)

    _, predicted = torch.max(output,1)

    return classes[predicted.item()]