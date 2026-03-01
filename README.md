# DIVINA-Classifier
Marine life classification engine for the DIVINA Project.

## Installation

```bash
pip install git+https://github.com/skaarfundgandr/DIVINA-Classifier.git
```

Or using `uv`:

```bash
uv add "divina-classifier @ git+https://github.com/skaarfundgandr/DIVINA-Classifier.git"
```

## Usage as a Library

```python
from divina_classifier import ResNet50Classifier, load_image, DEVICE

# Initialize the classifier
classifier = ResNet50Classifier()

# Load and process an image
input_tensor = load_image("path/to/image.jpg").to(DEVICE)

# Run inference
result = classifier.predict(input_tensor)

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## CLI Usage

After installation, you can use the `divina-classify` command:

```bash
divina-classify path/to/image.jpg
```
