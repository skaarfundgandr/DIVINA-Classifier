from PIL import Image
import sys
import os
from divina_classifier.models.resnet import ResNet50Classifier
from divina_classifier.models.vgg16 import VGG16Classifier


def main():
    if len(sys.argv) < 2:
        print("Usage: divina-classify <path_to_image>  OR  python -m divina_classifier <path_to_image>")
        return

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    print(f"Initializing VGG16 model...")
    classifier = VGG16Classifier()

    print(f"Processing image: {image_path}...")
    try:
        # Pass raw PIL image directly to predict
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Running inference...")
    result = classifier.predict(image)

    print("-" * 30)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Class ID:   {result['class_id']}")
    print("-" * 30)

if __name__ == "__main__":
    main()
