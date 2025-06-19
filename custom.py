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
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# Function to load imagesi
def retrieve_images(filepath: str):
    return [
        cv2.imread(os.path.join(filepath, img))
        for img in os.listdir(filepath)
        if img.endswith(('.png', '.jpg', '.jpeg'))
    ]

# Dataset paths
ABS_FILE_PATHS = [
        #Dataset path goes here
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



# Training Transform with augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation/Test Transform (no augmentation)
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
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
    def __init__(self, image_paths_train, image_paths_val, image_paths_test,
                 labels_train, labels_val, labels_test,
                 train_transform=None, eval_transform=None, batch_size=64, num_workers=4):
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
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        train_dataset = ImageDataset(self.image_paths_train, self.labels_train, transform=self.train_transform)
        # Compute class weights
        class_sample_counts = np.bincount(self.labels_train)
        class_weights = 1. / class_sample_counts
        sample_weights = [class_weights[label] for label in self.labels_train]

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          sampler=sampler,
                          num_workers=self.num_workers,
                          pin_memory=True)
    
    def val_dataloader(self):
        val_dataset = ImageDataset(self.image_paths_val, self.labels_val, transform=self.eval_transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        test_dataset = ImageDataset(self.image_paths_test, self.labels_test, transform=self.eval_transform)
      
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

# Instantiate the data module
data_module = LungImgDataModule(
    image_paths_train=X_train,
    image_paths_val=X_val,
    image_paths_test=X_test,
    labels_train=Y_train,
    labels_val=Y_val,
    labels_test=Y_test,
    train_transform=train_transform,
    eval_transform=eval_transform,
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



class CustomModelLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, num_classes=5):
        super().__init__()
        self.model = DeepCNNModel(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

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
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# In[ ]:
custom_model = CustomModelLightning(learning_rate=1e-3, num_classes=5)


# In[ ]:
trainer = pl.Trainer(max_epochs=100)
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


# In[ ]:
trainer.test(custom_model, data_module)


# In[ ]:
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

def generate_classification_report(model, datamodule, label_encoder):
    predictions, true_labels = get_predictions_and_labels(model, datamodule)
    target_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    report = classification_report(true_labels, predictions, target_names=target_names)
    return report

# Generate and print the classification report
custom_report = generate_classification_report(custom_model, data_module, le)
print("Classification Report for Custom Model:\n", custom_report)


from sklearn.metrics import confusion_matrix

def compute_specificity(y_true, y_pred, label_encoder):
    """
    Compute specificity (true negative rate) for each class in a multiclass setting.
    For each class, specificity = TN / (TN + FP), where:
      - TN is the sum of true negatives for that class.
      - FP is the sum of false positives for that class.
    """
    cm = confusion_matrix(y_true, y_pred)
    specificity_dict = {}
    for i, label in enumerate(label_encoder.classes_):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_dict[label] = specificity
    return specificity_dict








# Save the report
os.makedirs("CustomModelResults", exist_ok=True)
with open("CustomModelResults/custom_model_classification_report.txt", "w") as f:
    f.write(custom_report)


# Compute test set predictions and specificity
custom_predictions, custom_true_labels = get_predictions_and_labels(custom_model, data_module)
custom_specificity_dict = compute_specificity(custom_true_labels, custom_predictions, le)
custom_avg_specificity = np.mean(list(custom_specificity_dict.values()))
print("Test Set Specificity for Custom Model:")
print(f"  Average Specificity: {custom_avg_specificity:.4f}")
print(f"  Per Class Specificity: {custom_specificity_dict}")




# In[ ]:
# Create the directory if it doesn't exist
os.makedirs("CustomModelResults", exist_ok=True)

# Save the model state dictionary
torch.save(custom_model.state_dict(), "CustomModelResults/custom_model.pth")









from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


from sklearn.model_selection import StratifiedKFold
import numpy as np

# Combine your training and validation sets into a DataFrame for cross-validation
trainval_df = pd.DataFrame({'img': X_trainval.values, 'label': Y_trainval.values})

# Repeat the CV run (e.g., 10 times with different seeds)
for run in range(1):
    #seed = 42 + run
    seed = 42
    print(f"\n\n=========== Cross-Validation Run {run+1}/10 | Seed: {seed} ===========")

    # Create 10-fold stratified CV with the current seed
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cv_results = []  # To store metrics for each fold

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(trainval_df['img'], trainval_df['label'])):
        print(f"\n========== Fold {fold+1}/10 (Seed: {seed}) ==========")

        # Retrieve fold data
        X_fold_train = trainval_df.iloc[train_idx]['img']
        Y_fold_train = trainval_df.iloc[train_idx]['label']
        X_fold_val   = trainval_df.iloc[val_idx]['img']
        Y_fold_val   = trainval_df.iloc[val_idx]['label']

        # Create a data module for the fold
        fold_data_module = LungImgDataModule(
            image_paths_train=X_fold_train,
            image_paths_val=X_fold_val,
            image_paths_test=X_test,
            labels_train=Y_fold_train,
            labels_val=Y_fold_val,
            labels_test=Y_test,
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size=16
        )

        # Instantiate a new model instance for this fold
        fold_model = CustomModelLightning(learning_rate=1e-3, num_classes=5)

        # Create and run the trainer
        #fold_trainer = pl.Trainer(max_epochs=50)
        callbacks = [
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
                    ]

        fold_trainer = pl.Trainer(
            max_epochs=100,
            callbacks=callbacks,
            enable_checkpointing=True,
            #deterministic=True,  # Optional: for reproducibility
                )
        fold_trainer.fit(fold_model, fold_data_module)

        # Evaluate the fold using the validation set
        # (You can use fold_trainer.validate if your model logs 'val_acc', etc.)
        val_metrics = fold_trainer.validate(fold_model, fold_data_module)
        # Here we assume val_metrics[0] is a dictionary with validation accuracy; alternatively, compute metrics below.
        # We'll compute metrics directly from predictions.

        # Get predictions and true labels from the validation dataloader
        fold_model.eval()
        val_preds = []
        val_true = []
        for batch in fold_data_module.val_dataloader():
            X, Y = batch
            with torch.no_grad():
                Y_hat = fold_model(X)
                preds = torch.argmax(Y_hat, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(Y.cpu().numpy())
        val_preds = np.array(val_preds)
        val_true = np.array(val_true)

        # Compute metrics using sklearn
        acc    = accuracy_score(val_true, val_preds)
        f1     = f1_score(val_true, val_preds, average='macro')
        prec   = precision_score(val_true, val_preds, average='macro')
        rec    = recall_score(val_true, val_preds, average='macro')
        spec_dict = compute_specificity(val_true, val_preds, le)
        spec_avg  = np.mean(list(spec_dict.values()))
        

        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        # Compute and plot confusion matrix
        cm = confusion_matrix(val_true, val_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.tight_layout()

        # Save the figure
        os.makedirs("CustomModelResults/confusion_matrices", exist_ok=True)
        plt.savefig(f"CustomModelResults/confusion_matrices/fold_{fold+1}_confusion_matrix.png")
        plt.close()

        # Print fold metrics
        print(f"Fold {fold+1} Metrics:")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  Specificity: {spec_avg:.4f}")
        print(f"  Per Class Specificity: {spec_dict}")

        # Save current fold metrics
        cv_results.append({
            'val_acc': acc,
            'val_f1': f1,
            'val_precision': prec,
            'val_recall': rec,
            'val_specificity': spec_avg,
            'per_class_specificity': spec_dict
        })

        # Clean up: free memory if needed
        del fold_model, fold_trainer, fold_data_module
        torch.cuda.empty_cache()

    # Compute average metrics across folds for this run
    avg_acc = np.mean([m['val_acc'] for m in cv_results])
    avg_f1 = np.mean([m['val_f1'] for m in cv_results])
    avg_precision = np.mean([m['val_precision'] for m in cv_results])
    avg_recall = np.mean([m['val_recall'] for m in cv_results])
    avg_specificity = np.mean([m['val_specificity'] for m in cv_results])

    # Print the average cross-validation metrics
    print(f"\n>>> Custom Model - Average Cross-Validation Metrics (Seed: {seed}):")
    print(f"  Average Accuracy   : {avg_acc:.4f}")
    print(f"  Average F1 Score   : {avg_f1:.4f}")
    print(f"  Average Precision  : {avg_precision:.4f}")
    print(f"  Average Recall     : {avg_recall:.4f}")
    print(f"  Average Specificity: {avg_specificity:.4f}")

    # Optionally, save the results for this run to file
    os.makedirs("CustomModelResults", exist_ok=True)
    with open(f"CustomModelResults/cv_run_{run+1}_results.txt", "w") as f:
        for i, metrics in enumerate(cv_results):
            f.write(f"Fold {i+1} Metrics:\n")
            f.write(f"  Accuracy   : {metrics['val_acc']:.4f}\n")
            f.write(f"  F1 Score   : {metrics['val_f1']:.4f}\n")
            f.write(f"  Precision  : {metrics['val_precision']:.4f}\n")
            f.write(f"  Recall     : {metrics['val_recall']:.4f}\n")
            f.write(f"  Specificity: {metrics['val_specificity']:.4f}\n")
            f.write(f"  Per Class Specificity: {metrics['per_class_specificity']}\n\n")
        f.write(f"Average Cross-Validation Metrics:\n")
        f.write(f"  Average Accuracy   : {avg_acc:.4f}\n")
        f.write(f"  Average F1 Score   : {avg_f1:.4f}\n")
        f.write(f"  Average Precision  : {avg_precision:.4f}\n")
        f.write(f"  Average Recall     : {avg_recall:.4f}\n")
        f.write(f"  Average Specificity: {avg_specificity:.4f}\n")
