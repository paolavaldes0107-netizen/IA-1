import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os

# ─── Configuración Global ─────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 3
MIN_DELTA = 1e-4
MODEL_PATH = "best_mnist_model.pt"
# ──────────────────────────────────────────────────────────────

# Carga y preprocesamiento de datos
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

# Definición del modelo
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = MNISTNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Funciones de entrenamiento y evaluación
def train(model, loader):
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

# Configuración de gráfica en vivo
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

# Ciclo de entrenamiento con early stopping a partir de época 10
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = evaluate_loss(model, val_loader)

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    update_plot()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Time: {time.time() - t0:.1f}s")

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("Mejor modelo guardado.")
    else:
        if epoch >= 10:
            patience_counter += 1
            print(f"Contador de paciencia: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping activado.")
                break

# Evaluación final con mejor modelo
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Mejor modelo cargado para evaluación final.")

test_loss, test_acc = evaluate_loss(model, test_loader)
print(f"Precisión final en test: {test_acc:.4f}")

# Guardar la gráfica final
plt.ioff()
fig.savefig("loss_plot.png")
print("Gráfica de pérdida guardada como 'loss_plot.png'")
