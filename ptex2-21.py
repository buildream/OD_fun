from pathlib import Path

import cv2
import numpy as np
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

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )

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
# 2. Device configuration
#########################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Current device: {device}")


#########################################
# 3. Load the trained model
#########################################

script_directory = Path(__file__).resolve().parent
model_path = script_directory / "test2.pt"

print(f"Model path: {model_path}")

if not model_path.exists():
    raise FileNotFoundError(
        f"Could not find the trained model:\n{model_path}"
    )

# test2.pt was created using:
# torch.save(model, "test2.pt")
#
# In recent PyTorch versions, weights_only=False may be required
# when loading an entire model object.
try:
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
except TypeError:
    # Compatibility with older PyTorch versions.
    model = torch.load(
        model_path,
        map_location=device,
    )

model = model.to(device)
model.eval()

print("The trained model was loaded successfully.")


#########################################
# 4. Image preprocessing
#########################################

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


def preprocess_image(
    grayscale_roi,
    invert_image=True,
):
    """
    Convert a webcam ROI to an MNIST-compatible tensor.

    Input:
        grayscale_roi: Grayscale OpenCV image
        invert_image:
            True  -> dark digit on bright paper is inverted
            False -> bright digit on dark background is retained

    Output:
        input_tensor: [1, 1, 28, 28]
        processed_image: 28 x 28 image for visualization
    """

    # Reduce noise slightly.
    blurred_image = cv2.GaussianBlur(
        grayscale_roi,
        (5, 5),
        0,
    )

    # Convert the image to black and white using Otsu thresholding.
    if invert_image:
        threshold_type = (
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU
        )
    else:
        threshold_type = (
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

    _, binary_image = cv2.threshold(
        blurred_image,
        0,
        255,
        threshold_type,
    )

    input_tensor = transform(binary_image)

    # Add the batch dimension:
    # [1, 28, 28] -> [1, 1, 28, 28]
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    processed_image = (
        input_tensor
        .squeeze(0)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    processed_image = (
        processed_image * 255
    ).astype(np.uint8)

    return input_tensor, processed_image


#########################################
# 5. Inference function
#########################################

def predict_digit(input_tensor):
    """
    Predict a handwritten digit from 0 to 9.
    """

    model.eval()

    with torch.inference_mode():
        output = model(input_tensor)

        # The model returns log probabilities.
        probabilities = torch.exp(output)

        predicted_digit = (
            output.argmax(dim=1).item()
        )

        confidence = probabilities[
            0,
            predicted_digit,
        ].item()

    return predicted_digit, confidence, output


#########################################
# 6. Open the webcam
#########################################

camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera index {camera_index}."
    )

# Optional camera resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print()
print("Camera controls")
print("-------------------------------")
print("SPACE : Run inference")
print("I     : Toggle image inversion")
print("R     : Reset prediction")
print("Q     : Quit")
print("ESC   : Quit")
print("-------------------------------")


#########################################
# 7. Camera loop
#########################################

predicted_digit = None
confidence = None

# Most handwritten digits are written in black on white paper.
# MNIST normally contains bright digits on a dark background,
# so inversion is enabled initially.
invert_image = True

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read an image from the camera.")
            break

        frame_height, frame_width = frame.shape[:2]

        #################################
        # Define the center ROI
        #################################

        roi_size = int(
            min(frame_width, frame_height) * 0.55
        )

        center_x = frame_width // 2
        center_y = frame_height // 2

        x1 = center_x - roi_size // 2
        y1 = center_y - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        # Draw the ROI rectangle.
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Extract the ROI.
        roi = frame[y1:y2, x1:x2]

        grayscale_roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        #################################
        # Display status information
        #################################

        inversion_text = (
            "ON" if invert_image else "OFF"
        )

        cv2.putText(
            frame,
            f"Inversion: {inversion_text}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Place one digit inside the green box",
            (20, frame_height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "SPACE: predict | I: invert | Q: quit",
            (20, frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if predicted_digit is not None:
            result_text = (
                f"Prediction: {predicted_digit} "
                f"({confidence * 100:.1f}%)"
            )

            cv2.putText(
                frame,
                result_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        #################################
        # Show camera and grayscale ROI
        #################################

        cv2.imshow(
            "MNIST Camera Inference",
            frame,
        )

        cv2.imshow(
            "Grayscale ROI",
            grayscale_roi,
        )

        key = cv2.waitKey(1) & 0xFF

        #################################
        # SPACE: Run inference
        #################################

        if key == 32:
            input_tensor, processed_image = (
                preprocess_image(
                    grayscale_roi,
                    invert_image=invert_image,
                )
            )

            predicted_digit, confidence, output = (
                predict_digit(input_tensor)
            )

            print()
            print("Model output:")
            print(output.detach().cpu())

            print(
                f"Predicted digit: {predicted_digit}"
            )
            print(
                f"Confidence: {confidence:.2%}"
            )

            cv2.imshow(
                "Model Input 28x28",
                cv2.resize(
                    processed_image,
                    (280, 280),
                    interpolation=cv2.INTER_NEAREST,
                ),
            )

            if predicted_digit == 5:
                print("5!")
            else:
                print("The predicted digit is not 5.")

        #################################
        # I: Toggle inversion
        #################################

        elif key == ord("i"):
            invert_image = not invert_image

            print(
                f"Image inversion: "
                f"{'ON' if invert_image else 'OFF'}"
            )

        #################################
        # R: Reset prediction
        #################################

        elif key == ord("r"):
            predicted_digit = None
            confidence = None

            print("Prediction was reset.")

        #################################
        # Q or ESC: Quit
        #################################

        elif key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("The camera program has ended.")