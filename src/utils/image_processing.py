from torchvision import transforms
from PIL import Image

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    return transform(image).unsqueeze(0)  # Add batch dimension
