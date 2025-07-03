from PIL import Image
import io
from torchvision import transforms

def read_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)