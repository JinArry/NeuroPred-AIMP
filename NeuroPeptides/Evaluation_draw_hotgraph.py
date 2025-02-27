import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, f1_score
from ESM_PepNet_LSTM_hotfigure import AIMP

warnings.filterwarnings('ignore')
torch.manual_seed(4)  # 设置 PyTorch 的随机种子
np.random.seed(4)  # 设置 NumPy 的随机种子
random.seed(4)  # 设置 Python 内置随机库的随机种子
device = "cuda:1"
model_checkpoint = "facebook/esm2_t6_8M_UR50D"

# Load data
df_val = pd.read_csv('testing.csv')
val_sequences = df_val["Seq"].tolist()
val_labels = df_val["Label"].tolist()


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


val_dict = {"text": val_sequences, 'labels': val_labels}

# Prepare datasets and dataloaders
val_data = MyDataset(val_dict)
val_dataloader = DataLoader(val_data, batch_size=128, shuffle=False)

# Initialize model, criterion and optimizer hidden=640
model = AIMP(pre_feas_dim=1280, feas_dim=21, hidden=1024, n_lstm=5, dropout=0.5)
model.load_state_dict(torch.load("my_best_model_2_lstm_data_split_train.pth"))
model = model.cuda(device)


def generate_heatmap_data(model, dataloader, device):
    model.eval()
    activations = {"tcn": [], "esm_conv": [], "sequences": []}

    with torch.no_grad():
        for batch in dataloader:
            seqs = batch[0]
            one_hot = get_onehot_features(seqs)
            one_hot_tensor = torch.tensor(one_hot, device=device, dtype=torch.float)

            # 前向传播，激活值自动保存到 model.tcn_activations 和 model.esm_conv_activations
            _ = model(seqs, one_hot_tensor)

            # 收集数据（假设batch_size=1）
            activations["tcn"].append(model.tcn_activations.squeeze(0).cpu().numpy())
            activations["esm_conv"].append(model.esm_conv_activations.squeeze(0).cpu().numpy())
            activations["sequences"].extend(seqs)

    return activations


def plot_heatmap(activation, sequence, title, save_path):
    """
    activation: numpy数组，形状为 (seq_len, feature_dim)
    sequence: 原始氨基酸序列（长度50）
    """
    # 沿特征维度取均值（或最大值）
    activation_agg = np.mean(activation, axis=0)  # 或 np.max(activation, axis=0)

    # 归一化到 [0,1]
    activation_norm = (activation_agg - np.min(activation_agg)) / (np.max(activation_agg) - np.min(activation_agg))
    # 创建热图
    plt.figure(figsize=(15, 2))
    plt.imshow(activation_norm.reshape(1, -1), cmap="viridis", aspect="auto")

    # 标注氨基酸序列
    plt.xticks(np.arange(len(sequence)), list(sequence), rotation=90)
    plt.yticks([])
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# 示例调用
activations = generate_heatmap_data(model, val_dataloader, device)
for i in range(3):  # 可视化前3个样本
    plot_heatmap(
        activation=activations["tcn"][i],
        sequence=activations["sequences"][i],
        title="TCN Activation Heatmap (Sample {})".format(i + 1),
        save_path=f"tcn_heatmap_sample_{i + 1}.png"
    )
    plot_heatmap(
        activation=activations["esm_conv"][i],
        sequence=activations["sequences"][i],
        title="ESM-Conv Activation Heatmap (Sample {})".format(i + 1),
        save_path=f"esm_conv_heatmap_sample_{i + 1}.png"
    )