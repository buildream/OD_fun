import torch
import PIL
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
#import playsound

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load('./test2.pt', map_location=device)

# Image transform
imgs=PIL.Image.open('./fig1.jpg')
imgs=imgs.convert("L")
imgs.show()

tf=transforms.ToTensor()
tfs=transforms.Resize([28,28])

imgst=tf(imgs)
imgss=tfs(imgst)
imgss=imgss.unsqueeze(0)
imgss=imgss.to(device)

# Inference
model.eval()
with torch.no_grad():
   results = model(imgss)
   print(results)
   ans=results[0].argmax(0)

# Check if the answer is correct
if ans == 5:
    print("5!")
    #playsound.playsound('Succeed.mp3')
else:
    print("Wrong!")
    #playsound.playsound('Fail.mp3')
