import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# ─── Global Config ─────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ───────────────────────────────────────────────────────────────

# ─── Load and Preprocess Data ──────────────────────────────────
train_df = pd.read_csv("../IA/datasets/mnist/mnist_train.csv")
test_df = pd.read_csv("../IA/datasets/mnist/mnist_test.csv")

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ─── Model Definitions ─────────────────────────────────────────
class MNISTNetNoBN(nn.Module):
    def __init__(self):
        super(MNISTNetNoBN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class MNISTNetBN(nn.Module):
    def __init__(self):
        super(MNISTNetBN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ─── Instantiate Models ────────────────────────────────────────
model_no_bn = MNISTNetNoBN().to(DEVICE)
model_bn = MNISTNetBN().to(DEVICE)

optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=0.001)
optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

# ─── Training and Evaluation Functions ─────────────────────────
def train(model, optimizer, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / total, correct / total

def evaluate_loss(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return total_loss / total, correct / total

# ─── Live Plot Setup ──────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots()

train_loss_no_bn = []
val_loss_no_bn = []
train_loss_bn = []
val_loss_bn = []

def update_plot():
    ax.clear()
    ax.plot(train_loss_no_bn, label="Train No BN")
    ax.plot(val_loss_no_bn, label="Val No BN")
    ax.plot(train_loss_bn, label="Train BN", linestyle="--")
    ax.plot(val_loss_bn, label="Val BN", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss (BN vs No BN)")
    ax.legend()
    plt.draw()
    plt.pause(0.01)

# ─── Training Loop ─────────────────────────────────────────────
for epoch in range(EPOCHS):
    t0 = time.time()

    train_nbn_loss, train_nbn_acc = train(model_no_bn, optimizer_no_bn, train_loader)
    val_nbn_loss, val_nbn_acc = evaluate_loss(model_no_bn, val_loader)

    train_bn_loss, train_bn_acc = train(model_bn, optimizer_bn, train_loader)
    val_bn_loss, val_bn_acc = evaluate_loss(model_bn, val_loader)

    train_loss_no_bn.append(train_nbn_loss)
    val_loss_no_bn.append(val_nbn_loss)
    train_loss_bn.append(train_bn_loss)
    val_loss_bn.append(val_bn_loss)

    update_plot()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"[No BN] Train Loss: {train_nbn_loss:.4f}, Acc: {train_nbn_acc:.4f} | Val Loss: {val_nbn_loss:.4f}, Acc: {val_nbn_acc:.4f}")
    print(f"[BN]    Train Loss: {train_bn_loss:.4f}, Acc: {train_bn_acc:.4f} | Val Loss: {val_bn_loss:.4f}, Acc: {val_bn_acc:.4f}")
    print(f"Time: {time.time() - t0:.1f}s\n")

# ─── Final Test Evaluation ─────────────────────────────────────
test_nbn_loss, test_nbn_acc = evaluate_loss(model_no_bn, test_loader)
test_bn_loss, test_bn_acc = evaluate_loss(model_bn, test_loader)

print(f"Test Accuracy [No BN]: {test_nbn_acc:.4f}")
print(f"Test Accuracy [BN]:    {test_bn_acc:.4f}")
