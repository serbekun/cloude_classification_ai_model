import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
MODEL_PATH = os.path.join("..", "models", "cloud_model.pth")
CLASS_INDEX_PATH = os.path.join("..", "models", "class_to_idx.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")

class CloudClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CloudClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CloudDataset(Dataset):
    def __init__(self, images_dir, labels_file, class_index_file=None, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open(labels_file, 'r') as f:
            raw_meta = json.load(f)
        self.labels_data = {}

        with open("../data_packs/Clouds-1000/meta.json", 'r') as meta_file:
            meta = json.load(meta_file)
        class_titles = [c['title'] for c in meta['classes']]
        self.class_to_idx = {title: idx for idx, title in enumerate(class_titles)}

        if class_index_file:
            with open(class_index_file, 'w') as f:
                json.dump(self.class_to_idx, f)

        for fname, label_str in raw_meta.items():
            if label_str in self.class_to_idx:
                self.labels_data[fname] = self.class_to_idx[label_str]

        self.image_files = list(self.labels_data.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        label = self.labels_data[image_name]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

images_dir = "../data_packs/Clouds-1000/all_images_and_labels/images"
labels_file = "../data_packs/Clouds-1000/image_labels.json" 

with open("../data_packs/Clouds-1000/meta.json", 'r') as f:
    class_titles = [c['title'] for c in json.load(f)['classes']]

NUM_CLASSES = len(class_titles)

dataset = CloudDataset(images_dir, labels_file, CLASS_INDEX_PATH, transform=transform)

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,          
    pin_memory=True    
)

def load_or_create_model():
    model = CloudClassifier(NUM_CLASSES)
    
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        print("Model loaded from file.")
    else:
        print("Created new model.")
    return model


def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        save_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(save_model.state_dict(), MODEL_PATH)
        print("Model saved.")

if __name__ == "__main__":
    model = load_or_create_model()
    train_model(model)
