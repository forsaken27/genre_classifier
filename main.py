import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import train_loader, val_loader, test_loader
from model import CNNModel
from config import Config
import random

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

c = Config()
model = CNNModel(num_classes=10)
model = model.to(c.device)
optimizer = optim.Adam(model.parameters(), lr=c.learning_rate, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _, (inputs, labels) in enumerate(train_loader):

        # 1. Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        # 2. Zero the gradients
        optimizer.zero_grad()
        # 3. Forward pass
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        correct += (predictions==labels).sum().item()
        total += labels.size(0)

        # 4. Compute loss
        loss = criterion(outputs, labels)
        # 5. Backward pass
        loss.backward()
        # 6. Gradient clipping because of overfitting
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 7. Update weights
        optimizer.step()

        running_loss += loss.item()

    return running_loss/len(train_loader), correct, total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # for confusion matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            # 1. Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # 2. Forward pass
            outputs = model(inputs)
            # 3. Compute loss
            loss = criterion(outputs, labels)
            # 4. Compute accuracy
            predictions = outputs.argmax(dim=1)
            correct += (predictions==labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())

            running_loss += loss.item()
        cm = confusion_matrix(all_labels, all_preds)

    return running_loss/len(val_loader), correct, total, cm


def train_cnn(c, model, optimizer, scheduler):
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    for epoch in range(c.epochs):
        # training one epoch
        train_loss, train_correct, train_total= train_one_epoch(model, train_loader, c.criterion, optimizer, c.device)
        val_loss, val_correct, val_total, _ = validate(model, val_loader, c.criterion, c.device)
        train_accuracy = train_correct/train_total * 100
        val_accuracy = val_correct/val_total * 100
        # displaying and saving the results
        print(f"[{epoch+1}/{c.epochs}]: | train_loss={train_loss:.4f}; | train_accuracy={train_accuracy:.1f}% | val_loss={val_loss:.4f}; | val_accuracy={val_accuracy:.1f}% |")
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        # Step scheduler based on validation accuracy
        scheduler.step(val_accuracy)
        # Saving the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

    # plotting the training results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.ravel()
    epochs_axs = np.arange(1, c.epochs+1)
    axs[0].plot(epochs_axs, train_loss_list, label="train_loss")
    axs[0].plot(epochs_axs, val_loss_list, label="val_loss")
    axs[0].set_title("Loss Curves")
    axs[0].legend()
    axs[1].plot(epochs_axs, train_accuracy_list, label="train_accuracy")
    axs[1].plot(epochs_axs, val_accuracy_list, label="val_accuracy")
    axs[1].set_title("Accuracy Curves")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig("training_stats.png")

    # testing the best model
    print('\n')
    print(f"Training complete: best_val_accuracy - {best_val_accuracy}%")
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_correct, test_total, cm = validate(model, test_loader, c.criterion, c.device)
    test_accuracy = test_correct / test_total * 100
    print("Final Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.1f}% ({test_correct}/{test_total})")
    print('\n')

    # plotting confusion matrix of test
    plt.figure(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
    disp.plot(cmap="Blues")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")



if __name__=="__main__":
    print("     Training Information    ")
    print(f"Epochs: {c.epochs}")
    print(f"Learning Rate: {c.learning_rate}")
    print(f"Device: {c.device}")
    train_cnn(c, model, optimizer, scheduler)
