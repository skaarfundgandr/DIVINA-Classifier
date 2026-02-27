import sys
import os
from models.resnet import ResNet50Classifier
from utils.image_processing import load_image
from globals import DEVICE

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <path_to_image>")
        return

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    print(f"Initializing ResNet50 model...")
    classifier = ResNet50Classifier()

    print(f"Processing image: {image_path}...")
    try:
        input_tensor = load_image(image_path).to(DEVICE)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Running inference...")
    result = classifier.predict(input_tensor)

    print("-" * 30)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Class ID:   {result['class_id']}")
    print("-" * 30)

if __name__ == "__main__":
    main()
