import torch
from torchvision import transforms
from PIL import Image
import json
import sys
import os

parent_dir = os.path.abspath (os.path.join(os.path.dirname(__file__), '../train'))
sys.path.insert(0, parent_dir)

# model
from train import CloudClassifier, CLASS_INDEX_PATH, MODEL_PATH

with open(CLASS_INDEX_PATH, 'r') as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_CLASSES = len(class_to_idx)
model = CloudClassifier(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

def predict(image_path, topk=3):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=topk)

        print("Top predictions:")
        for prob, idx in zip(top_probs[0], top_indices[0]):
            print(f"{idx_to_class[idx.item()]}: {prob.item()*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py path_to_image.jpg")
        sys.exit(1)
    image_path = sys.argv[1]
    predict(image_path)
