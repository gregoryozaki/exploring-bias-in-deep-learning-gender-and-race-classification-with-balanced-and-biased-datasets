# CELULA 01
from google.colab import drive
import zipfile
import os

drive.mount('/content/drive')

drive_base_path = '/content/drive/MyDrive/datasets'
celeba_zip_drive = os.path.join(drive_base_path, 'dataset-celeba', 'celeba.zip')

celeba_zip_local = '/content/celeba.zip'
celeba_extract_path = '/content/celeba'

!cp "{celeba_zip_drive}" "{celeba_zip_local}"

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        print(f"Extraindo {zip_path} para {extract_to} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extração concluída.")
    else:
        print(f"📂 Já existe a pasta extraída: {extract_to}")

extract_zip(celeba_zip_local, celeba_extract_path)

celeba_csv_train_drive = '/content/drive/MyDrive/datasets/dataset-celeba/celeba_train.csv'
celeba_csv_val_drive   = '/content/drive/MyDrive/datasets/dataset-celeba/celeba_val.csv'

!cp "{celeba_csv_train_drive}" /content/
!cp "{celeba_csv_val_drive}" /content/

# CELULA 02
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os

class FaceDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Corrige capitalização
        self.data['gender'] = self.data['gender'].str.lower().str.strip()
        self.data['race'] = self.data['race'].str.lower().str.strip()

        # Mapeamento de classes
        self.gender_map = {'male': 0, 'female': 1}
        self.race_map = {
            'white': 0, 'black': 1, 'indian': 2, 'east asian': 3,
            'southeast asian': 4, 'middle eastern': 5, 'latino_hispanic': 6
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['file'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        gender_label = self.gender_map[row['gender']]
        race_label = self.race_map[row['race']]

        return image, gender_label, race_label

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)

transform = weights.transforms()

celeba_train_dataset = FaceDataset('/content/celeba_train.csv', '/content/celeba/celeba', transform)
celeba_val_dataset   = FaceDataset('/content/celeba_val.csv', '/content/celeba/celeba', transform)

from torch.utils.data import DataLoader
from tqdm import tqdm

batch_size = 16
num_workers = 2

celeba_train_loader = DataLoader(celeba_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,  pin_memory=True)
celeba_val_loader   = DataLoader(celeba_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)

# CELULA 03
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

class MultiTaskConvNeXtTiny(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskConvNeXtTiny, self).__init__()
        
        # Carrega o backbone
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        
        # Remove a camada final padrão
        self.backbone.classifier = nn.Identity()
        hidden_dim = 768  # Para ConvNeXt-Tiny

        # Cabeça para gênero
        self.gender_head = nn.Sequential(
            nn.Flatten(),  # Corrige o shape
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2)
        )

        # Cabeça para raça
        self.race_head = nn.Sequential(
            nn.Flatten(),  # Corrige o shape
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 7)
        )

    def forward(self, x):
        features = self.backbone(x)  # Saída será [batch_size, 768, 1, 1]
        gender_out = self.gender_head(features)
        race_out = self.race_head(features)
        return gender_out, race_out

# Correto agora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskConvNeXtTiny(pretrained=True).to(device)

#CELULA 04
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import torch
torch.cuda.empty_cache()

# Configurações
epochs = 5
learning_rate = 1e-4

# Funções de perda
loss_fn_gender = nn.CrossEntropyLoss()
loss_fn_race = nn.CrossEntropyLoss()

# Otimizador
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Loop de treino
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct_gender = 0
    correct_race = 0
    total = 0

    for images, gender_labels, race_labels in tqdm(dataloader, desc="Treinando"):
        images = images.to(device)
        gender_labels = gender_labels.to(device)
        race_labels = race_labels.to(device)

        optimizer.zero_grad()

        gender_preds, race_preds = model(images)

        loss_gender = loss_fn_gender(gender_preds, gender_labels)
        loss_race = loss_fn_race(race_preds, race_labels)
        loss = loss_gender + loss_race
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, gender_predicted = torch.max(gender_preds, 1)
        _, race_predicted = torch.max(race_preds, 1)

        correct_gender += (gender_predicted == gender_labels).sum().item()
        correct_race += (race_predicted == race_labels).sum().item()
        total += gender_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    gender_acc = correct_gender / total
    race_acc = correct_race / total
    return avg_loss, gender_acc, race_acc

# Loop de validação
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_gender = 0
    correct_race = 0
    total = 0

    with torch.no_grad():
        for images, gender_labels, race_labels in tqdm(dataloader, desc="Validando"):
            images = images.to(device)
            gender_labels = gender_labels.to(device)
            race_labels = race_labels.to(device)

            gender_preds, race_preds = model(images)

            loss_gender = loss_fn_gender(gender_preds, gender_labels)
            loss_race = loss_fn_race(race_preds, race_labels)
            loss = loss_gender + loss_race

            total_loss += loss.item()
            _, gender_predicted = torch.max(gender_preds, 1)
            _, race_predicted = torch.max(race_preds, 1)

            correct_gender += (gender_predicted == gender_labels).sum().item()
            correct_race += (race_predicted == race_labels).sum().item()
            total += gender_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    gender_acc = correct_gender / total
    race_acc = correct_race / total
    return avg_loss, gender_acc, race_acc

for epoch in range(epochs):
    print(f"\n🌟 Época {epoch+1}/{epochs}")

    train_loss, train_gender_acc, train_race_acc = train_one_epoch(model, celeba_train_loader, optimizer, device)
    val_loss, val_gender_acc, val_race_acc = validate(model, celeba_val_loader, device)

    print(f"🧠 Treino — Loss: {train_loss:.4f}, Gênero Acc: {train_gender_acc:.4f}, Raça Acc: {train_race_acc:.4f}")
    print(f"🔍 Validação — Loss: {val_loss:.4f}, Gênero Acc: {val_gender_acc:.4f}, Raça Acc: {val_race_acc:.4f}")
