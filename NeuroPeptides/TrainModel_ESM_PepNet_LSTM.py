import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from ESM_PepNet_LSTM import AIMP
import warnings
from sklearn.metrics import matthews_corrcoef, f1_score


warnings.filterwarnings('ignore')
set_seed(500)  # set_seed(4)
device = "cuda:1"
model_checkpoint = "facebook/esm2_t6_8M_UR50D"

# Load data
df_train = pd.read_csv('training.csv')
df_val = pd.read_csv('testing.csv')

train_sequences = df_train["Seq"].tolist()
train_labels = df_train["Label"].tolist()
val_sequences = df_val["Seq"].tolist()
val_labels = df_val["Label"].tolist()



class MyDataset(Dataset):
    def __init__(self, dict_data) -> None:
        super(MyDataset, self).__init__()
        self.data = dict_data
        self.max_len = 50  # 设置最大序列长度
        self.standard_amino_acids = set('ARNDCQEGHILKMFPSTWYV')  # 标准氨基酸集合

    def __getitem__(self, index):
        sequence = self.data['text'][index]
        label = self.data['labels'][index]

        # 将序列中的非标准氨基酸替换为 'X'
        sequence = ''.join([aa if aa in self.standard_amino_acids else 'X' for aa in sequence])

        # 截断或填充序列到固定长度
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]  # 截断
        else:
            sequence = sequence.ljust(self.max_len, 'X')  # 填充

        return [sequence, label]

    def __len__(self):
        return len(self.data['text'])


# Prepare datasets and dataloaders
train_dict = {"text": train_sequences, 'labels': train_labels}
val_dict = {"text": val_sequences, 'labels': val_labels}

epochs = 50
learning_rate = 5e-5
batch_size = 1280

import numpy as np

def one_hot_encode_protein(sequence):
    """
    One-hot encode a protein sequence.

    Args:
        sequence (str): Protein sequence consisting of standard amino acid letters.

    Returns:
        np.ndarray: One-hot encoded representation of the sequence.
        dict: Mapping of amino acids to indices.
    """
    # Define the standard amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Initialize a zero matrix with shape (sequence_length, num_amino_acids)
    one_hot_matrix = np.zeros((len(sequence), len(amino_acids)), dtype=int)

    for i, aa in enumerate(sequence):
        if aa in amino_acid_to_index:
            one_hot_matrix[i, amino_acid_to_index[aa]] = 1
        else:
            raise ValueError(f"Invalid amino acid '{aa}' in sequence.")

    return one_hot_matrix, amino_acid_to_index

def get_onehot_features(sequences, theta=50):
    element = 'ALRKNMDFCPQSETGWHYIV'
    onehot_ref = np.eye(21)  # 生成一个 21x21 的单位矩阵，最后一个维度用于element未包含的氨基酸
    features = []
    for seq in sequences:
        sequence_len = len(seq)
        feature = np.zeros((theta, 21), dtype=float)  # 修改为 21 维
        for i in range(min(sequence_len, theta)):
            aa = seq[i]
            index = element.find(aa)
            if index == -1:
                # 如果未找到氨基酸，使用特殊索引 20
                index = 20
            feature[i, :] = onehot_ref[index, :]
        features.append(feature)
    features = np.array(features)
    return features

train_data = MyDataset(train_dict)
val_data = MyDataset(val_dict)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # Shuffle for training
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
print("Current device:", torch.cuda.current_device())

# Initialize model, criterion and optimizer hidden=640
model = AIMP(pre_feas_dim=1280, feas_dim=21, hidden=1024, n_lstm=5, dropout=0.5)
model = model.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# Track metrics
train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
train_epochs_acc = []
valid_epochs_acc = []
train_epochs_sensitivity = []
valid_epochs_sensitivity = []
train_epochs_specificity = []
valid_epochs_specificity = []
train_epochs_auc = []
valid_epochs_auc = []

best_acc = 0

for epoch in range(epochs):
    model.train()
    train_epoch_loss = []
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []

    # Training loop
    for index, batch in enumerate(train_dataloader):

        batch_protein_sequences = list(batch[0])
        batch_labels = batch[1]
        one_hot_embedding = get_onehot_features(batch_protein_sequences)
        one_hot_tensor = torch.tensor(one_hot_embedding, device=device, dtype=torch.float)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_protein_sequences,one_hot_tensor)

        # Convert labels to one-hot encoding (optional depending on your loss function)
        label = torch.nn.functional.one_hot(batch_labels, num_classes=2).float().cuda()

        # 计算损失
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # 记录损失
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())

        # 计算预测
        train_argmax = torch.argmax(outputs, dim=1)
        correct += (train_argmax == batch_labels.cuda()).sum().item()

        # 收集预测结果，用于评估
        all_labels.extend(batch_labels.cpu().numpy())
        all_preds.extend(train_argmax.cpu().numpy())
        all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[:, 1])

    # Training accuracy and metrics
    train_acc = correct / len(train_labels)
    train_epochs_acc.append(train_acc)
    train_epochs_loss.append(np.mean(train_epoch_loss))

    train_cm = confusion_matrix(all_labels, all_preds)
    train_sensitivity = train_cm[1, 1] / (train_cm[1, 1] + train_cm[1, 0])
    train_specificity = train_cm[0, 0] / (train_cm[0, 0] + train_cm[0, 1])
    train_auc = roc_auc_score(all_labels, all_probs)

    # 计算 MCC 和 F1-score
    train_mcc = matthews_corrcoef(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds)

    train_epochs_sensitivity.append(train_sensitivity)
    train_epochs_specificity.append(train_specificity)
    train_epochs_auc.append(train_auc)

    # Validation loop
    model.eval()
    valid_epoch_loss = []
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for index, batch in enumerate(val_dataloader):
            batch_protein_sequences = list(batch[0])
            batch_labels = batch[1]
            one_hot_embedding = get_onehot_features(batch_protein_sequences)
            one_hot_tensor = torch.tensor(one_hot_embedding, device=device, dtype=torch.float)

            outputs = model(batch_protein_sequences, one_hot_tensor)   #保留tcn_features特征，用于做特征可视化


            # Convert labels to one-hot encoding
            label = torch.nn.functional.one_hot(batch_labels, num_classes=2).float().cuda()

            loss = criterion(outputs, label)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())

            # Get predictions
            val_argmax = torch.argmax(outputs, dim=1)
            correct += (val_argmax == batch_labels.cuda()).sum().item()

            # Store all predictions for metrics calculation
            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(val_argmax.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[:, 1])

    # Validation accuracy and metrics
    valid_acc = correct / len(val_labels)
    valid_epochs_acc.append(valid_acc)
    valid_epochs_loss.append(np.mean(valid_epoch_loss))

    val_cm = confusion_matrix(all_labels, all_preds)
    val_sensitivity = val_cm[1, 1] / (val_cm[1, 1] + val_cm[1, 0])
    val_specificity = val_cm[0, 0] / (val_cm[0, 0] + val_cm[0, 1])
    val_auc = roc_auc_score(all_labels, all_probs)

    # 计算 MCC 和 F1-score
    valid_mcc = matthews_corrcoef(all_labels, all_preds)
    valid_f1 = f1_score(all_labels, all_preds)

    valid_epochs_sensitivity.append(val_sensitivity)
    valid_epochs_specificity.append(val_specificity)
    valid_epochs_auc.append(val_auc)

    # Save best model based on validation accuracy
    if valid_acc >= best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), f"my_best_model_2_lstm_data_split_train.pth")
    print(
        f'Epoch {epoch}, Train Loss: {np.mean(train_epoch_loss):.4f}, Train Acc: {train_acc:.4f}, Train Sensitivity: {train_sensitivity:.4f}, Train Specificity: {train_specificity:.4f}, Train AUC: {train_auc:.4f}, Train MCC: {train_mcc:.4f}, Train F1: {train_f1:.4f}')
    print(
        f'Epoch {epoch}, Valid Loss: {np.mean(valid_epoch_loss):.4f}, Valid Acc: {valid_acc:.4f}, Valid Sensitivity: {val_sensitivity:.4f}, Valid Specificity: {val_specificity:.4f}, Valid AUC: {val_auc:.4f}, Valid MCC: {valid_mcc:.4f}, Valid F1: {valid_f1:.4f}')

print(f"Best accuracy: {best_acc}")
