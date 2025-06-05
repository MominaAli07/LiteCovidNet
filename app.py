import os
from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DeepCNNModel class (same as your training code)
class BlockA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.maxpool(x)

class BlockB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.avgpool(x)

class BlockC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.avgpool(x)

class BlockE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.global_avgpool(x)

class DeepCNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(DeepCNNModel, self).__init__()
        self.block_a = BlockA(3, 64)
        self.block_b = BlockB(64, 128)
        self.block_c = BlockC(128, 256)
        self.block_d = BlockC(256, 256)
        self.block_e = BlockE(256, 768)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x).view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)

# Load the pre-trained model
model = DeepCNNModel(num_classes=4)
model.load_state_dict(torch.load('final_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names
class_names = ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia']

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file uploaded!", 400
#     file = request.files['file']
#     if file.filename == '':
#         return "No file selected!", 400

#     # Save the file
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Load the image
#     image = Image.open(filepath).convert('RGB')

#     # Preprocess the image
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     print("Preprocessed image shape:", image.shape)

#     # Perform prediction
#     with torch.no_grad():
#         outputs = model(image)
#         print("Model Outputs (Logits):", outputs)
#         probabilities = torch.nn.functional.softmax(outputs, dim=1)
#         print("Class Probabilities:", probabilities)
#         _, predicted = torch.max(probabilities, 1)
#         print("Predicted Class:", predicted.item())

#     class_name = class_names[predicted.item()]
#     return render_template('result.html', class_name=class_name)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    # Save the file in the 'static/uploads' directory
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess the image
    image = Image.open(filepath).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
        

    class_name = class_names[predicted.item()]

    # Pass the relative path to the uploaded image
    uploaded_image_path = f'uploads/{file.filename}'
    return render_template('result.html', class_name=class_name, uploaded_image_path=uploaded_image_path)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
