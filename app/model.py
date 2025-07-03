import torch
from torchvision import models
from PIL import Image
import os
import requests
from app.utils import preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_labels = [
    'Cassava - bacterial blight', 'Cassava - brown spot', 'Cassava - green mite', 'Cassava - healthy', 'Cassava - mosaic',
    'Maize - leaf spot', 'Maize - leaf blight', 'Maize - fall armyworm', 'Maize - grasshopper', 'Maize - streak virus',
    'Maize - leaf beetle', 'Maize - healthy',
    'Tomato - leaf curl', 'Tomato - leaf blight', 'Tomato - septoria leaf spot', 'Tomato - verticillium wilt', 'Tomato - healthy',
    'Cashew - gummosis', 'Cashew - red rust', 'Cashew - anthracnose', 'Cashew - leaf miner', 'Cashew - healthy'
]

MODEL_PATH = 'models/best_model_v1_75percent.pth'
MODEL_URL = 'https://github.com/steveAzo/ccmt-api/releases/download/v1.0/best_model_v1_75percent.pth'

def download_model():
    try:
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.isfile(MODEL_PATH):
            print("Downloading model...")
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)
                print("Model downloaded successfully!")
            else:
                raise Exception(f"Failed to download model: HTTP {response.status_code}")
    except Exception as e:
        raise Exception(f"Model download failed: {str(e)}")

def load_model():
    try:
        download_model()
        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device).eval()
        return model, class_labels
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def predict_image(image: Image.Image, model, class_labels):
    try:
        image = preprocess_image(image).to(device)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            predicted_label = class_labels[predicted_idx.item()]
        return predicted_label, round(confidence.item(), 4)
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")