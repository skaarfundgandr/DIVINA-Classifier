import torch
from torchvision.models import vgg16, VGG16_Weights
from divina_classifier.globals import DEVICE
from divina_classifier.utils.image_processing import preprocess_image
from PIL import Image
from typing import Union

class VGG16Classifier:
    def __init__(self, compile=False):
        self.weights = VGG16_Weights.DEFAULT
        self.model = vgg16(weights=self.weights)

        if compile:
            self.model = torch.compile(self.model)

        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        self.categories = self.weights.meta["categories"]

    def predict(self, input_data: Union[torch.Tensor, Image.Image], confidence_min=0.5):
        """Run prediction on a preprocessed tensor OR a raw PIL image."""
        if isinstance(input_data, Image.Image):
            input_tensor = preprocess_image(input_data).to(DEVICE)
        else:
            input_tensor = input_data

        with torch.no_grad():
            output = self.model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        top_prob, top_catid = torch.topk(probabilities, 1)

        if top_prob[0].item() >= confidence_min:
            return {
                "class_id": top_catid[0].item(),
                "label": self.categories[top_catid[0]],
                "confidence": top_prob[0].item()
            }
        else:
            return {
                "class_id": -1,
                "label": "Unknown",
                "confidence": 0.0
            }
