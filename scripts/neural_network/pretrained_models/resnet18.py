import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models, transforms
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import time

# ─── GLOBAL SETTINGS ──────────────────────────────────────────
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── LOAD AND PREPROCESS DATA ────────────────────────────────
print("Loading MNIST data...")
train_df = pd.read_csv("../IA/datasets/mnist/mnist_train.csv")
test_df = pd.read_csv("../IA/datasets/mnist/mnist_test.csv")

X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
y_test = test_df.iloc[:, 0].values

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

print("Applying image transformations...")
X_train_tensor = torch.stack([transform(img) for img in X_train])
X_test_tensor = torch.stack([transform(img) for img in X_test])
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"Training samples: {len(X_train_tensor)} | Test samples: {len(X_test_tensor)}")

full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ─── LOAD PRETRAINED RESNET-18 ───────────────────────────────
print("Loading pretrained ResNet-18 model...")
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
model = resnet.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── TRAINING AND EVALUATION FUNCTIONS ───────────────────────
def train(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    print("\n[TRAINING]")
    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        _, preds = torch.max(outputs, 1)
        batch_correct = (preds == y_batch).sum().item()
        correct += batch_correct
        total += y_batch.size(0)

        print(f"  Batch {batch_idx+1:03}/{len(loader)} | "
              f"Loss: {loss.item():.4f} | "
              f"Correct: {batch_correct}/{y_batch.size(0)}")

    avg_loss = total_loss / total
    avg_acc = correct / total
    print(f"  → Epoch Train Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

def evaluate_loss(model, loader, mode="Validation"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    print(f"\n[{mode.upper()}]")
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            _, preds = torch.max(outputs, 1)
            batch_correct = (preds == y_batch).sum().item()
            correct += batch_correct
            total += y_batch.size(0)

            print(f"  Batch {batch_idx+1:03}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Correct: {batch_correct}/{y_batch.size(0)}")

    avg_loss = total_loss / total
    avg_acc = correct / total
    print(f"  → {mode} Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

# ─── LIVE PLOTTING SETUP ─────────────────────────────────────
plt.ion()
fig, ax = plt.subplots()
train_loss_list = []
val_loss_list = []

def update_plot():
    ax.clear()
    ax.plot(train_loss_list, label="Train Loss")
    ax.plot(val_loss_list, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    plt.draw()
    plt.pause(0.01)

# ─── TRAINING LOOP ───────────────────────────────────────────
print("\nStarting training loop...\n")
for epoch in range(EPOCHS):
    print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")
    start = time.time()

    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = evaluate_loss(model, val_loader, mode="Validation")

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    update_plot()

    print(f"[SUMMARY] Epoch {epoch+1} finished in {time.time() - start:.1f}s")
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ─── FINAL TEST EVALUATION ───────────────────────────────────
test_loss, test_acc = evaluate_loss(model, test_loader, mode="Test")
print(f"\n[FINAL TEST] Accuracy: {test_acc:.4f}")
