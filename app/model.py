import torch
from torchvision import models, transforms
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_labels = [
    'Cassava - bacterial blight', 'Cassava - brown spot', 'Cassava - green mite', 'Cassava - healthy', 'Cassava - mosaic',
    'Maize - leaf spot', 'Maize - leaf blight', 'Maize - fall armyworm', 'Maize - grasshoper', 'Maize - streak virus',
    'Maize - leaf beetle', 'Maize - healthy',
    'Tomato - leaf curl', 'Tomato - leaf blight', 'Tomato - septoria leaf spot', 'Tomato - verticulium wilt', 'Tomato - healthy',
    'Cashew - gumosis', 'Cashew - red rust', 'Cashew - anthracnose', 'Cashew - leaf miner', 'Cashew - healthy'
]

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))
    model.load_state_dict(torch.load('models/best_model_v1_75percent.pth', map_location=device))
    model.eval()
    return model, class_labels

def predict_image(image: Image.Image, model, class_labels):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_label = class_labels[predicted_idx.item()]

    return predicted_label, round(confidence.item(), 4)
