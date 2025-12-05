import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# ─── Global Config ─────────────────────────────────────────────
EPOCHS = 60
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
class MNISTNetNoDropout(nn.Module):
    def __init__(self):
        super(MNISTNetNoDropout, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class MNISTNetDropout(nn.Module):
    def __init__(self):
        super(MNISTNetDropout, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ─── Instantiate Models ────────────────────────────────────────
model_no_dropout = MNISTNetNoDropout().to(DEVICE)
model_dropout = MNISTNetDropout().to(DEVICE)

optimizer_no_dropout = optim.Adam(model_no_dropout.parameters(), lr=0.001)
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)

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

train_loss_no_dropout = []
val_loss_no_dropout = []
train_loss_dropout = []
val_loss_dropout = []

def update_plot():
    ax.clear()
    ax.plot(train_loss_no_dropout, label="Train No Dropout")
    ax.plot(val_loss_no_dropout, label="Val No Dropout")
    ax.plot(train_loss_dropout, label="Train Dropout", linestyle="--")
    ax.plot(val_loss_dropout, label="Val Dropout", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Comparison")
    ax.legend()
    plt.draw()
    plt.pause(0.01)


# ─── Training Loop ─────────────────────────────────────────────
for epoch in range(EPOCHS):
    t0 = time.time()
    
    # Train and Evaluate: No Dropout
    train_nd_loss, train_nd_acc = train(model_no_dropout, optimizer_no_dropout, train_loader)
    val_nd_loss, val_nd_acc = evaluate_loss(model_no_dropout, val_loader)

    # Train and Evaluate: Dropout
    train_do_loss, train_do_acc = train(model_dropout, optimizer_dropout, train_loader)
    val_do_loss, val_do_acc = evaluate_loss(model_dropout, val_loader)

    # Record Losses
    train_loss_no_dropout.append(train_nd_loss)
    val_loss_no_dropout.append(val_nd_loss)
    train_loss_dropout.append(train_do_loss)
    val_loss_dropout.append(val_do_loss)

    update_plot()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"[No Dropout] Train Loss: {train_nd_loss:.4f}, Acc: {train_nd_acc:.4f} | Val Loss: {val_nd_loss:.4f}, Acc: {val_nd_acc:.4f}")
    print(f"[Dropout]    Train Loss: {train_do_loss:.4f}, Acc: {train_do_acc:.4f} | Val Loss: {val_do_loss:.4f}, Acc: {val_do_acc:.4f}")
    print(f"Time: {time.time() - t0:.1f}s\n")

# ─── Final Test Evaluation ─────────────────────────────────────
test_nd_loss, test_nd_acc = evaluate_loss(model_no_dropout, test_loader)
test_do_loss, test_do_acc = evaluate_loss(model_dropout, test_loader)

print(f"Test Accuracy [No Dropout]: {test_nd_acc:.4f}")
print(f"Test Accuracy [Dropout]:    {test_do_acc:.4f}")
