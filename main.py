import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
import time
from tqdm import tqdm
from pathlib import Path
import csv
from PIL import Image


class BasicAttention(nn.Module):
    """Attention mechanism for focusing on important features in images."""

    def __init__(self, in_channels):
        """Initialize the attention module.

        Args:
            in_channels (int): Number of input channels
        """
        super(BasicAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass for attention module.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Attention weighted features
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicResNeXt(nn.Module):
    """ResNeXt-based model with attention for image classification."""

    def __init__(self, num_classes=100, dropout_rate=0.5):
        """Initialize the ResNeXt model.

        Args:
            num_classes (int, optional): Number of output classes.
            Defaults to 100.
            dropout_rate (float, optional): Dropout probability.
            Defaults to 0.5.
        """
        super(BasicResNeXt, self).__init__()

        # Load the ResNeXt50 backbone
        self.backbone = models.resnext50_32x4d(pretrained=True)
        feature_dim = 2048

        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # Simple attention mechanism
        self.attention = BasicAttention(feature_dim)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """Forward pass for classification model.

        Args:
            x (torch.Tensor): Input tensor (images)

        Returns:
            torch.Tensor: Class logits
        """
        # Extract features using backbone
        x = self.features(x)

        # Apply attention
        x = self.attention(x)

        # Global pooling
        x = self.global_pool(x).view(x.size(0), -1)

        # Dropout and classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class NumericFolderDataset(torch.utils.data.Dataset):
    """Dataset for loading images from folders with numeric names."""

    def __init__(self, root, transform=None):
        """Initialize the dataset.

        Args:
            root (str): Root directory containing class folders
            transform (callable, optional): Transform to apply to images
        """
        self.root = root
        self.transform = transform
        self.classes = self._find_classes(root)
        self.class_to_idx = {cls_name: int(cls_name)
                             for cls_name in self.classes}
        self.samples = self._make_dataset(root, self.class_to_idx)

    def _find_classes(self, dir_path):
        """Find all subdirectories and sort them numerically.

        Args:
            dir_path (str): Path to root directory

        Returns:
            list: Sorted list of class names
        """
        classes = [d.name for d in os.scandir(dir_path) if d.is_dir()]
        classes.sort(key=int)  # Sort numerically
        return classes

    def _make_dataset(self, dir_path, class_to_idx):
        """Create a list of (sample path, class_index) tuples.

        Args:
            dir_path (str): Path to root directory
            class_to_idx (dict): Mapping from class name to index

        Returns:
            list: List of (path, target) tuples
        """
        samples = []
        for target_class in self.classes:
            target_dir = os.path.join(dir_path, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(
                    os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if self._is_valid_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target_class])
                        samples.append(item)
        return samples

    def _is_valid_file(self, filename):
        """Check if a file is a valid image file.

        Args:
            filename (str): File name to check

        Returns:
            bool: True if file is a valid image
        """
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        return filename.lower().endswith(valid_extensions)

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Args:
            index (int): Index of the sample

        Returns:
            tuple: (sample, target) where target is class_index
        """
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        """Get dataset length.

        Returns:
            int: Number of samples in dataset
        """
        return len(self.samples)


class TestImageDataset(torch.utils.data.Dataset):
    """Dataset for test images without labels."""

    def __init__(self, test_dir, transform=None):
        """Initialize the test dataset.

        Args:
            test_dir (str): Directory containing test images
            transform (callable, optional): Transform to apply to images
        """
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = []

        # Get all image files in the test directory
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):
                    self.image_paths.append(os.path.join(root, file))

        self.image_paths.sort()  # Sort paths for deterministic results

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Args:
            index (int): Index of the sample

        Returns:
            tuple: (sample, image_name)
        """
        path = self.image_paths[index]
        filename = os.path.basename(path)
        sample = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, filename

    def __len__(self):
        """Get dataset length.

        Returns:
            int: Number of samples in dataset
        """
        return len(self.image_paths)


def count_parameters(model):
    """Count the number of trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_transforms():
    """Create data transformations for training and testing.

    Returns:
        tuple: (train_transform, test_transform)
    """
    train_transform = transforms.Compose(
        [
            # Resize and crop
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # Basic augmentations
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            # Normalization
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
            # Occasional random erasing
            transforms.RandomErasing(p=0.1),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    device="cuda",
):
    """Train the model.

    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (Optimizer): Optimizer
        scheduler (Scheduler): Learning rate scheduler
        num_epochs (int, optional): Number of epochs. Defaults to 25.
        device (str, optional): Device to use. Defaults to 'cuda'.

    Returns:
        nn.Module: Trained model
    """
    model.to(device)
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Use tqdm for progress bar
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        # Use tqdm for validation progress bar
        val_desc = f"Epoch {epoch+1}/{num_epochs} [Val]"
        val_pbar = tqdm(val_loader, desc=val_desc)
        for inputs, labels in val_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            val_pbar.set_postfix({"loss": loss.item()})

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        # Update learning rate
        scheduler.step()

        # Calculate time
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | "
            f"Time: {epoch_time:.1f}s Total: {total_time/60:.1f}m"
        )

        # Save the best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            print(
                f"New best validation accuracy: {best_acc:.4f} "
                f"- Saving model..."
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                },
                "best_model.pth",
            )

    # Print training summary
    total_time_mins = (time.time() - start_time) / 60
    print(f"Training completed in {total_time_mins:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f}")

    return model


class FineTuner:
    """Class for fine-tuning a model on a dataset."""

    def __init__(
        self,
        train_dir,
        val_dir,
        test_dir,
        num_classes=100,
        batch_size=32,
        num_workers=4,
    ):
        """Initialize the fine-tuner.

        Args:
            train_dir (str): Training data directory
            val_dir (str): Validation data directory
            test_dir (str): Test data directory
            num_classes (int, optional): Number of classes. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Get data transforms
        self.train_transform, self.val_transform = get_data_transforms()

        # Create model
        self.model = BasicResNeXt(num_classes=num_classes)

        # Move model to device
        self.model.to(self.device)

        # Print parameter count
        num_params = count_parameters(self.model)
        print(f"Model parameters: {num_params:,}")

    def load_data(self):
        """Load datasets and create data loaders.

        Returns:
            tuple: (train_dataset, val_dataset)
        """
        print("Loading datasets...")

        # Load train dataset
        train_dataset = NumericFolderDataset(
            root=self.train_dir, transform=self.train_transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Load validation dataset
        val_dataset = NumericFolderDataset(
            root=self.val_dir, transform=self.val_transform
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Print dataset info
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")

        # Update num_classes from dataset if needed
        if len(train_dataset.classes) != self.num_classes:
            print(
                f"Updating num_classes from {self.num_classes} to "
                f"{len(train_dataset.classes)}"
            )
            self.num_classes = len(train_dataset.classes)
            # Recreate model with correct number of classes
            self.model = BasicResNeXt(num_classes=self.num_classes)
            self.model.to(self.device)

        return train_dataset, val_dataset

    def train(self, num_epochs=30, lr=0.001):
        """Train the model.

        Args:
            num_epochs (int, optional): Number of epochs. Defaults to 30.
            lr (float, optional): Learning rate. Defaults to 0.001.

        Returns:
            nn.Module: Trained model
        """
        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        criterion = nn.CrossEntropyLoss()

        # Train the model
        print(f"Training on {self.device}...")
        self.model = train_model(
            self.model,
            self.train_loader,
            self.val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=num_epochs,
            device=self.device,
        )

        return self.model

    def load_best_model(self, model_path="best_model.pth"):
        """Load the best model from training.

        Args:
            model_path (str, optional): Path to model.
            Defaults to 'best_model.pth'.

        Returns:
            nn.Module: Loaded model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model from {model_path} "
            f"(epoch {checkpoint['epoch']}, "
            f"acc: {checkpoint['best_acc']:.4f})"
        )
        return self.model

    def generate_predictions(self, output_file="prediction.csv"):
        """Generate predictions for test images.

        Args:
            output_file (str, optional): Output file path.
            Defaults to 'prediction.csv'.
        """
        print(f"Generating predictions to {output_file}...")

        # Create test dataset and loader
        test_dataset = TestImageDataset(
            self.test_dir, transform=self.val_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=0
        )

        self.model.eval()
        predictions = []

        with torch.no_grad():
            loop = tqdm(test_loader, desc="Testing")
            for inputs, filenames in loop:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for filename, pred in zip(filenames, preds):
                    # Remove extension (if present)
                    clean_filename = Path(filename).stem
                    predictions.append([clean_filename, pred.item()])

        # Write predictions to CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "pred_label"])
            writer.writerows(predictions)

        print(f"Predictions saved to {output_file}")


def main():
    """Main function to run the fine-tuning process."""
    # Set paths to data directories
    train_dir = r"C:\Users\xingt\OneDrive\Desktop\data\train"
    val_dir = r"C:\Users\xingt\OneDrive\Desktop\data\val"
    test_dir = r"C:\Users\xingt\OneDrive\Desktop\data\test"

    # Initialize the model trainer
    tuner = FineTuner(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        num_classes=100,  # Will be updated automatically from dataset
        batch_size=32,
    )

    # Load datasets
    tuner.load_data()

    # Train the model
    tuner.train(num_epochs=40)

    # Generate predictions for test images
    tuner.generate_predictions(output_file="prediction.csv")


if __name__ == "__main__":
    main()
