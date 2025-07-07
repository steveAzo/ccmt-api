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
        transforms.Resize((380, 380)),  # Match B2 input size
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
    ])
    return transform(image).unsqueeze(0)