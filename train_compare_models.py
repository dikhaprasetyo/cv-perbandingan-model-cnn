import os
import time
import copy
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ============================================================
# KONFIGURASI UTAMA
# ============================================================

DATA_DIR = "dataset"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0

FREEZE_FEATURE_EXTRACTOR = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Device yang digunakan:", DEVICE)

if torch.cuda.is_available():
    print("Nama GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# SEED AGAR HASIL LEBIH STABIL
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ============================================================
# CEK STRUKTUR DATASET
# ============================================================

def check_dataset():
    required_folders = [
        os.path.join(DATA_DIR, "train"),
        os.path.join(DATA_DIR, "val"),
        os.path.join(DATA_DIR, "test")
    ]

    for folder in required_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"Folder tidak ditemukan: {folder}\n"
                f"Pastikan struktur dataset seperti ini:\n"
                f"dataset/train/nama_kelas\n"
                f"dataset/val/nama_kelas\n"
                f"dataset/test/nama_kelas"
            )


# ============================================================
# LOAD DATASET
# ============================================================

def get_dataloaders(input_size):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "test"),
        transform=val_test_transform
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Kelas dataset:", class_names)
    print("Jumlah kelas:", num_classes)
    print("Jumlah data train:", len(train_dataset))
    print("Jumlah data val:", len(val_dataset))
    print("Jumlah data test:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader, class_names, num_classes


# ============================================================
# FREEZE PARAMETER MODEL
# ============================================================

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


# ============================================================
# MEMBUAT MODEL
# ============================================================

def build_model(model_name, num_classes):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        if FREEZE_FEATURE_EXTRACTOR:
            freeze_model_parameters(model)

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        if FREEZE_FEATURE_EXTRACTOR:
            freeze_model_parameters(model)

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)

        if FREEZE_FEATURE_EXTRACTOR:
            freeze_model_parameters(model)

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            num_classes
        )

    elif model_name == "inceptionv3":
        weights = models.Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=weights, aux_logits=True)

        if FREEZE_FEATURE_EXTRACTOR:
            freeze_model_parameters(model)

        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(
                model.AuxLogits.fc.in_features,
                num_classes
            )

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)

        if FREEZE_FEATURE_EXTRACTOR:
            freeze_model_parameters(model)

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    elif model_name == "mobilenetv3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)

        if FREEZE_FEATURE_EXTRACTOR:
            freeze_model_parameters(model)

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )

    else:
        raise ValueError(f"Model tidak dikenali: {model_name}")

    return model


# ============================================================
# TRAINING MODEL
# ============================================================

def train_model(model, train_loader, val_loader, model_name):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    model = model.to(DEVICE)

    print("Model berada di:", next(model.parameters()).device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        print("\n" + "-" * 60)
        print(f"Model: {model_name.upper()} | Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)

        # ==========================
        # TRAINING
        # ==========================
        model.train()

        train_loss_total = 0.0
        train_correct_total = 0
        train_data_total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            if isinstance(outputs, tuple):
                main_outputs = outputs[0]
                aux_outputs = outputs[1]

                loss_main = criterion(main_outputs, labels)
                loss_aux = criterion(aux_outputs, labels)

                loss = loss_main + 0.4 * loss_aux
                outputs = main_outputs
            else:
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * inputs.size(0)
            train_correct_total += torch.sum(preds == labels.data).item()
            train_data_total += labels.size(0)

        train_loss = train_loss_total / train_data_total
        train_acc = train_correct_total / train_data_total

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()

        val_loss_total = 0.0
        val_correct_total = 0
        val_data_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss_total += loss.item() * inputs.size(0)
                val_correct_total += torch.sum(preds == labels.data).item()
                val_data_total += labels.size(0)

        val_loss = val_loss_total / val_data_total
        val_acc = val_correct_total / val_data_total

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    end_time = time.time()
    training_time = end_time - start_time

    model.load_state_dict(best_model_wts)

    return model, best_val_acc, training_time, history


# ============================================================
# EVALUASI MODEL PADA DATA TEST
# ============================================================

def evaluate_model(model, test_loader, class_names, model_name):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:", model_name)
    print(report)

    os.makedirs("reports", exist_ok=True)

    report_path = os.path.join("reports", f"classification_report_{model_name}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return accuracy, precision, recall, f1, report_path


# ============================================================
# SIMPAN GRAFIK TRAINING
# ============================================================

def save_training_plot(history, model_name):
    os.makedirs("plots", exist_ok=True)

    # Grafik Accuracy
    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="Train Accuracy")
    plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - {model_name}")
    plt.legend()
    plt.savefig(os.path.join("plots", f"accuracy_{model_name}.png"))
    plt.close()

    # Grafik Loss
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {model_name}")
    plt.legend()
    plt.savefig(os.path.join("plots", f"loss_{model_name}.png"))
    plt.close()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    check_dataset()

    model_names = [
        "resnet18",
        "resnet50",
        "vgg16",
        "inceptionv3",
        "densenet121",
        "mobilenetv3"
    ]

    results = []

    os.makedirs("saved_models", exist_ok=True)

    for model_name in model_names:
        print("\n" + "=" * 80)
        print(f"MEMULAI TRAINING MODEL: {model_name.upper()}")
        print("=" * 80)

        input_size = 299 if model_name == "inceptionv3" else 224

        train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(input_size)

        model = build_model(model_name, num_classes)

        trained_model, best_val_acc, training_time, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name
        )

        accuracy, precision, recall, f1, report_path = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            class_names=class_names,
            model_name=model_name
        )

        model_path = os.path.join("saved_models", f"{model_name}.pth")

        torch.save({
            "model_name": model_name,
            "model_state_dict": trained_model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "input_size": input_size
        }, model_path)

        save_training_plot(history, model_name)

        result = {
            "Model": model_name,
            "Best Validation Accuracy": round(best_val_acc, 4),
            "Test Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Training Time Seconds": round(training_time, 2),
            "Saved Model": model_path,
            "Report": report_path
        }

        results.append(result)

        print("\nHasil sementara:")
        print(result)

        del model
        del trained_model
        torch.cuda.empty_cache()
        gc.collect()

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("HASIL AKHIR PERBANDINGAN MODEL")
    print("=" * 80)
    print(df_results)

    df_results.to_csv("hasil_perbandingan_model.csv", index=False)

    print("\nFile hasil disimpan sebagai:")
    print("hasil_perbandingan_model.csv")
    print("saved_models/")
    print("reports/")
    print("plots/")


if __name__ == "__main__":
    main()