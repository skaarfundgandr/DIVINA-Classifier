import torch
from torchvision.models import resnet50, ResNet50_Weights
from globals import DEVICE

class ResNet50Classifier:
    def __init__(self, compile=False):
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights).to(DEVICE)

        if compile:
            self.model = torch.compile(self.model)

        self.model.eval()
        
        self.categories = self.weights.meta["categories"]

    def predict(self, input_tensor, confidence_min=0.5):
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