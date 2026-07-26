import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


#########################################
# 1. CNN model definition
#########################################

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


#########################################
# 2. Device setting
#########################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Current device:", device)


#########################################
# 3. Load the trained model
#########################################

script_directory = Path(__file__).resolve().parent
model_path = script_directory / "test2.pt"

print("Model path:", model_path)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file was not found: {model_path}"
    )

# This loading method corresponds to:
# torch.save(model, "test2.pt")
try:
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

except TypeError:
    # For older PyTorch versions
    model = torch.load(
        model_path,
        map_location=device,
    )

model = model.to(device)
model.eval()

print("The trained model was loaded.")


#########################################
# 4. Image transform
#########################################

tf = transforms.ToTensor()
tfs = transforms.Resize((28, 28))


#########################################
# 5. Open the camera
#########################################

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open the camera.")

print()
print("ESC: Capture the current image and run inference")
print("TAB: Exit the program")


#########################################
# 6. Camera and inference loop
#########################################

try:
    while True:

        while True:
            ret, imgs = cap.read()

            if not ret:
                print("Failed to read a camera image.")
                break

            # Convert the camera image to grayscale.
            imgs = cv2.cvtColor(
                imgs,
                cv2.COLOR_BGR2GRAY,
            )

            cv2.imshow("Test", imgs)

            key = cv2.waitKey(10) & 0xFF

            # ESC key: capture the current frame.
            if key == 27:
                break

            # TAB key: terminate the program.
            if key == 9:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

        if not ret:
            break

        cv2.destroyAllWindows()

        #################################
        # Convert the image for the CNN
        #################################

        imgst = tf(imgs)

        # Resize from the camera resolution to 28 x 28.
        imgss = tfs(imgst)

        # [1, 28, 28] -> [1, 1, 28, 28]
        imgss = imgss.unsqueeze(0)

        imgss = imgss.to(device)

        print("Model input shape:", imgss.shape)

        #################################
        # Run inference
        #################################

        model.eval()

        with torch.no_grad():
            results = model(imgss)

            print("Model output:")
            print(results)

            ans = results[0].argmax(dim=0).item()

        print("Predicted digit:", ans)

        #################################
        # Check the result
        #################################

        if ans == 5:
            print("5!")
        else:
            print("Fail")

        print()
        print("The camera will open again.")
        print("ESC: Run inference")
        print("TAB: Exit")

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Program ended.")