#!/usr/bin/env python
# coding: utf-8

# In[2]:

# Imports
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
import matplotlib.pyplot as plt

# Function to load images
def retrieve_images(filepath: str):
    return [
        cv2.imread(os.path.join(filepath, img))
        for img in os.listdir(filepath)
        if img.endswith(('.png', '.jpg', '.jpeg'))
    ]

# Dataset paths
ABS_FILE_PATHS = [
        '/home/ma2mp/STAT7400/Balanced5Class/Normal/images/',
        '/home/ma2mp/STAT7400/Balanced5Class/Lung_Opacity/images/',
        '/home/ma2mp/STAT7400/Balanced5Class/COVID/images/',
        '/home/ma2mp/STAT7400/Balanced5Class/Pneumonia/images/',
        '/home/ma2mp/STAT7400/Balanced5Class/TB/images'
        ]

labels = ['normal', 'afflicted', 'covid', 'pneumonia', 'tb']

# Load images and labels into a DataFrame
df_images = pd.concat([
    pd.DataFrame({
        'img': [os.path.join(ABS_FILE_PATHS[i], img) for img in os.listdir(ABS_FILE_PATHS[i]) if img.endswith(('.png', '.jpg', '.jpeg'))],
        'label': labels[i]
    })
    for i in range(len(labels))
], ignore_index=True)

# Encode labels
le = LabelEncoder()
df_images['label'] = le.fit_transform(df_images['label'])

# Debug the number of paths and labels
image_paths = df_images['img']
labels = df_images['label']
print(f"Number of image paths: {len(image_paths)}, Number of labels: {len(labels)}")

# Ensure consistency before splitting
assert len(image_paths) == len(labels), "Mismatch in image paths and labels"

# Split into training-validation and test sets
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42, shuffle=True
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval, test_size=0.125, stratify=Y_trainval, random_state=42, shuffle=True
)



    # Transformations for preprocessing images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_paths = image_paths.tolist()
        self.labels = labels.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve the image and corresponding label at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and label as tensors.
        """
        # Load the image from the file path
        img = cv2.imread(self.image_paths[idx])
        label = self.labels[idx]

        # Apply transformations (e.g., resizing, normalization)
        if self.transform:
            img = self.transform(img)

        # Convert label to tensor
        return img, torch.tensor(label).long()

# Lightning Data Module
class LungImgDataModule(pl.LightningDataModule):
    def __init__(self, image_paths_train, image_paths_val, image_paths_test, labels_train, labels_val, labels_test, transform=None, batch_size=64, num_workers=4):
        """
        Lightning Data Module for loading image data.
        
        Args:
            image_paths_train: List of training image file paths.
            image_paths_val: List of validation image file paths.
            image_paths_test: List of test image file paths.
            labels_train: List of training labels.
            labels_val: List of validation labels.
            labels_test: List of test labels.
            transform: Transformations to apply to images.
            batch_size: Number of samples per batch.
            num_workers: Number of workers for DataLoader.
        """
        super().__init__()
        self.image_paths_train = image_paths_train
        self.image_paths_val = image_paths_val
        self.image_paths_test = image_paths_test
        self.labels_train = labels_train
        self.labels_val = labels_val
        self.labels_test = labels_test
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        train_dataset = ImageDataset(self.image_paths_train, self.labels_train, transform=self.transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        val_dataset = ImageDataset(self.image_paths_val, self.labels_val, transform=self.transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        test_dataset = ImageDataset(self.image_paths_test, self.labels_test, transform=self.transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

# Instantiate the data module
data_module = LungImgDataModule(
    image_paths_train=X_train,
    image_paths_val=X_val,
    image_paths_test=X_test,
    labels_train=Y_train,
    labels_val=Y_val,
    labels_test=Y_test,
    transform=transform,
    batch_size=16
)

# In[ ]:


class BlockA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.maxpool(x)

class BlockB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.avgpool(x)

class BlockC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.avgpool(x)

class BlockE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.global_avgpool(x)


# In[ ]:


class DeepCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.block_a = BlockA(3, 64)
        self.block_b = BlockB(64, 128)
        self.block_c = BlockC(128, 256)
        self.block_d = BlockC(256, 256)
        self.block_e = BlockE(256, 768)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x).view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)


# In[ ]:


# SGLD Optimizer Definition
class SGLDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p + wd * p
                p.add_(d_p, alpha=-lr)
                noise = torch.randn_like(p) * (2 * lr) ** 0.5
                p.add_(noise)





class CustomModelLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, num_classes=5, switch_epoch=30, sgld_lr=1e-4):
        super().__init__()
        self.model = DeepCNNModel(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.sgld_lr = sgld_lr
        self.switch_epoch = switch_epoch

        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.train_f1score = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1score = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1score = F1Score(task='multiclass', num_classes=num_classes, average='macro')

        self.train_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.test_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')

        self.train_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.test_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')

        # For plotting learning curves
        self.train_step_losses = []
        self.val_step_losses = []
        self.train_step_accs = []
        self.val_step_accs = []

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        loss = self.criterion(Y_hat, Y)
        acc = self.train_accuracy(Y_hat, Y)
        f1score = self.train_f1score(Y_hat, Y)
        precision = self.train_precision(Y_hat, Y)
        recall = self.train_recall(Y_hat, Y)

        self.train_step_losses.append(loss)
        self.train_step_accs.append(acc)

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1score', f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        loss = self.criterion(Y_hat, Y)
        acc = self.val_accuracy(Y_hat, Y)
        f1score = self.val_f1score(Y_hat, Y)
        precision = self.val_precision(Y_hat, Y)
        recall = self.val_recall(Y_hat, Y)

        self.val_step_losses.append(loss)
        self.val_step_accs.append(acc)

        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1score', f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        loss = self.criterion(Y_hat, Y)
        acc = self.test_accuracy(Y_hat, Y)
        f1score = self.test_f1score(Y_hat, Y)
        precision = self.test_precision(Y_hat, Y)
        recall = self.test_recall(Y_hat, Y)

        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_f1score', f1score, on_step=False, on_epoch=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # For plotting the training learning curve
        avg_loss = torch.stack(self.train_step_losses).mean().item()
        avg_acc = torch.stack(self.train_step_accs).mean().item()
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        self.train_step_losses.clear()
        self.train_step_accs.clear()

    def on_validation_epoch_end(self):
        # For plotting the validation learning curve
        avg_loss = torch.stack(self.val_step_losses).mean().item()
        avg_acc = torch.stack(self.val_step_accs).mean().item()
        self.val_losses.append(avg_loss)
        self.val_accs.append(avg_acc)
        self.val_step_losses.clear()
        self.val_step_accs.clear()

    def configure_optimizers(self):
        if self.current_epoch < self.switch_epoch:
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            return SGLDOptimizer(self.parameters(), lr=self.sgld_lr)


# In[ ]:


custom_model = CustomModelLightning(learning_rate=1e-3, num_classes=5)


# In[ ]:


trainer = pl.Trainer(max_epochs=100)


# In[ ]:


trainer.fit(custom_model, data_module)


# In[ ]:


# Remove the first entries if they are placeholders
if len(custom_model.val_losses) > len(custom_model.train_losses):
    del custom_model.val_losses[0]
    del custom_model.val_accs[0]


# In[ ]:


def plot_metrics(model):
    epochs = range(len(model.train_losses))
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, model.train_losses, label='Train Loss')
    plt.plot(epochs, model.val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, model.train_accs, label='Train Accuracy')
    plt.plot(epochs, model.val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Plot and save the learning curves
plot_metrics(custom_model)
plt.savefig("CustomModelResults/custom_model_learning_curve.png")


def get_predictions_and_labels(model, datamodule):
    model.eval()
    predictions = []
    true_labels = []
    for batch in datamodule.test_dataloader():
        X, Y = batch
        with torch.no_grad():
            Y_hat = model(X)
            preds = torch.argmax(Y_hat, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(Y.cpu().numpy())
    return np.array(predictions), np.array(true_labels)


from tqdm import tqdm

def mc_predict(model, dataloader, n_passes: int = 20):
    """
    Run n_passes stochastic forward passes and return a stacked tensor of
    shape (n_passes, N, num_classes), where N = len(dataloader.dataset).
    """
    model.eval()
    preds_list = []
    for _ in tqdm(range(n_passes), desc="MC passes"):
        all_probs = []
        for x, _ in dataloader:
            x = x.to(model.device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
        preds_list.append(torch.cat(all_probs))
    return torch.stack(preds_list)  # (n_passes, N, C)


def plot_reliability_diagram(probs_tensor, labels_tensor, n_bins=15, save_path='reliability.png'):
    """
    probs_tensor : torch.tensor or np.ndarray, shape (N, C)
    labels_tensor: torch.tensor or np.ndarray, shape (N,)
    Saves a reliability diagram as PNG.
    """
    if isinstance(probs_tensor, np.ndarray):
        probs_tensor = torch.from_numpy(probs_tensor)
    if isinstance(labels_tensor, np.ndarray):
        labels_tensor = torch.from_numpy(labels_tensor)

    confidences, predictions = probs_tensor.max(dim=1)
    accuracies = predictions.eq(labels_tensor)

    bins = torch.linspace(0, 1, n_bins + 1)
    bin_centers, bin_acc = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = confidences.gt(lo) & confidences.le(hi)
        if idx.any():
            bin_centers.append(((lo + hi) / 2).item())
            bin_acc.append(accuracies[idx].float().mean().item())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    plt.plot(bin_centers, bin_acc, marker='o', label='Model')
    plt.xlabel('Confidence'); plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path); plt.close()




# In[ ]:


trainer.test(custom_model, data_module)
predictions, true_labels = get_predictions_and_labels(custom_model, data_module)
y_true = np.array(true_labels)
# -------------------------------------------------------------
# Uncertainty estimation using MC sampling
# -------------------------------------------------------------
N_PASSES = 20
val_loader = data_module.test_dataloader()

# MC Sampling
preds_stack = mc_predict(custom_model, val_loader, n_passes=N_PASSES)  # shape: (20, N, 5)
mean_probs = preds_stack.mean(dim=0)  # (N, 5)
y_pred = mean_probs.argmax(dim=1).numpy()

# 1. Predictive Entropy
entropy = -np.sum(mean_probs.numpy() * np.log(mean_probs.numpy() + 1e-8), axis=1)
avg_entropy = entropy.mean()

# 2. Variation Ratio
variation_ratio = 1.0 - np.max(mean_probs.numpy(), axis=1)
avg_variation_ratio = variation_ratio.mean()

# 3. ECE
confidences = np.max(mean_probs.numpy(), axis=1)
correctness = (y_pred == y_true).astype(float)
ece = np.abs(confidences - correctness).mean()

# Ensure directory
os.makedirs("CustomModelResults", exist_ok=True)

# 4. Save uncertainty metrics
with open("CustomModelResults/custom_model_uncertainty_metrics.txt", "w") as f:
    f.write(f"Average Predictive Entropy : {avg_entropy:.4f}\n")
    f.write(f"Average Variation Ratio    : {avg_variation_ratio:.4f}\n")
    f.write(f"Expected Calibration Error : {ece:.4f}\n")
print("✅ Saved uncertainty metrics.")

# 5. Save reliability diagram
plot_reliability_diagram(mean_probs, torch.tensor(y_true),
    save_path="CustomModelResults/custom_model_reliability.png")
print("✅ Saved reliability diagram.")




from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# --------- ROC Curve Code START ---------
# Number of classes (should be 5 for your case)
n_classes = mean_probs.shape[1]
# Binarize the labels for ROC computation
y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

# Compute ROC curve and AUC for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], mean_probs.numpy()[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Micro-average (all classes together)
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), mean_probs.numpy().ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot and save
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)],
             label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot(fpr["micro"], tpr["micro"], 'k--', lw=2, label=f"Micro-average (area = {roc_auc['micro']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()

# Save ROC to your results directory
os.makedirs("CustomModelResults", exist_ok=True)
plt.savefig('CustomModelResults/custom_model_roc_curve.png')
plt.close()
print("✅ Saved ROC curve.")
# --------- ROC Curve Code END ---------


# -------------------------------------------------------------------------
# Fetch predictions  →  confusion matrix  →  specificity  →  save files
# -------------------------------------------------------------------------

os.makedirs("CustomModelResults", exist_ok=True)

# Confusion matrix (raw counts)
cm = confusion_matrix(true_labels, predictions,
                      labels=list(range(len(le.classes_))))

# Macro specificity
TN = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
FP = cm.sum(axis=0) - np.diag(cm)
macro_spec = (TN / (TN + FP + 1e-8)).mean()
print(f"Macro specificity: {macro_spec:.4f}")

# Save CM counts
pd.DataFrame(cm, index=le.classes_, columns=le.classes_) \
    .to_csv("CustomModelResults/confusion_matrix.csv")

# Save CM heat‑map
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(np.arange(len(le.classes_)))
ax.set_yticks(np.arange(len(le.classes_)))
ax.set_xticklabels(le.classes_)
ax.set_yticklabels(le.classes_)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("CustomModelResults/confusion_matrix.png")
plt.close()

# Classification report + append specificity
report = classification_report(
    true_labels, predictions,
    target_names=le.inverse_transform(range(len(le.classes_))))
with open("CustomModelResults/custom_model_classification_report.txt", "w") as fh:
    fh.write(report)
    fh.write(f"\nMacro specificity: {macro_spec:.4f}\n")


# In[ ]:


# Create the directory if it doesn't exist
os.makedirs("CustomModelResults", exist_ok=True)

# Save the model state dictionary
torch.save(custom_model.state_dict(), "CustomModelResults/custom_model.pth")


# In[ ]:


