import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
import matplotlib.pyplot as plt
from ESM_PepNet_LSTM import AIMP


warnings.filterwarnings('ignore')
set_seed(4)
device = "cuda:1"
model_checkpoint = "facebook/esm2_t6_8M_UR50D"

# df_val = pd.read_csv('val_data.csv')
# df_val = pd.read_csv('data/datasets/AMP/independent_test.csv')
# df_val = pd.read_csv('data/datasets/AMP/test.csv')
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

# epochs = 100
# learning_rate = 0.00001
batch_size = 128

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def collate_fn(batch):
    max_len = 30
    pt_batch = tokenizer([b[0] for b in batch], max_length=max_len, padding="max_length", truncation=True,
                         return_tensors='pt')
    labels = [b[1] for b in batch]
    return {'labels': labels, 'input_ids': pt_batch['input_ids'],
            'attention_mask': pt_batch['attention_mask']}


val_data = MyDataset(val_dict)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)



# model = AIMP(pre_feas_dim=5120, hidden=1024, n_transformer=2, dropout=0.5)
# model.load_state_dict(torch.load("my_best_model_15B.pth"))
# model = torch.load("my_best_model_2_trans_independent.pth", map_location="cpu")
print("Start Evaluation")
model = AIMP(pre_feas_dim=1280, feas_dim=21, hidden=1024, n_lstm=5, dropout=0.5)
model.load_state_dict(torch.load("my_best_model_2_lstm_data_split_train.pth"))

# model = torch.load("my_best_model_2_lstm_data_split_train.pth", map_location="cpu")
model = model.to(device)

valid_loss = []
valid_epochs_loss = []
valid_epochs_acc = []
best_acc = 0

predictions = []
labels = []
probabilities_list = []

model.eval()
with torch.no_grad():
    # 初始化一个列表来存储所有批次的 one-hot 特征
    all_one_hot_features = []

    # 初始化一个列表来存储所有批次的tcn特征
    tcn_features = []

    # 初始化一个列表来存储所有批次的esm特征
    esm_features = []

    # 初始化一个列表来存储所有批次的pre_feas特征
    esm_conv_features = []

    # 初始化一个列表来存储所有批次的onehot+esm特征
    combine_features = []

    #初始化一个列表来存储所有批次的lstm特征
    transformer_features = []


    currect = 0
    for index, batch in enumerate(val_dataloader):
        # batchs = {k: v for k, v in batch.items()}
        batch_protein_sequences = list(batch[0])

        batch_labels = batch[1]

        one_hot_embedding = get_onehot_features(batch_protein_sequences)
        one_hot_tensor = torch.tensor(one_hot_embedding, device=device, dtype=torch.float)

        outputs = model(batch_protein_sequences,one_hot_tensor)
        probabilities = outputs[:, 1]
        label = torch.nn.functional.one_hot(torch.tensor(batch_labels).to(torch.int64), num_classes=2).float()
        val_argmax = np.argmax(outputs.cpu(), axis=1)
        predictions.extend(val_argmax.numpy())
        labels.extend(batch_labels)
        probabilities_list.extend(probabilities.cpu().numpy())

        # 将 one-hot 特征添加到列表中
        # all_one_hot_features.append(one_hot_tensor.cpu().numpy())
        # 将 tcn 特征添加到列表中
        # tcn_features.append(tcn_features_batch.cpu().numpy())

        #将esm_features
        # esm_features.append(esm_features_batch.cpu().numpy())
        #将esm卷积之后的特征添加到列表中
        # esm_conv_features.append(esm_conv_features_batch.cpu().numpy())

        # 将onehot+esm的特征添加到列表中
        # combine_features.append(combine_features_batch.cpu().numpy())
        # 将lstm的特征添加到列表中
        # transformer_features.append(transformer_out_batch.cpu().numpy())




# Convert lists to numpy arrays
predictions = np.array(predictions)
labels = np.array(labels)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

# Accuracy
acc = accuracy_score(labels, predictions)

# Precision
prec = precision_score(labels, predictions)

# Sensitivity (Recall)
sens = recall_score(labels, predictions)

# Specificity
spec = tn / (tn + fp)

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(labels, predictions)

# AUC
auc = roc_auc_score(labels, probabilities_list)

# F1-score
f1 = f1_score(labels, predictions)

# Print the metrics
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Sensitivity (Recall): {sens:.4f}")
print(f"Specificity: {spec:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"AUC: {auc:.4f}")
# Print the F1-scores
print(f"F1-Score: {f1:.4f}")

# Add this after the evaluation loop where you collect predictions, labels, and probabilities

# Create a DataFrame to organize the results
results_df = pd.DataFrame({
    'Sequence': val_sequences,
    'True_Label': labels,
    'Predicted_Label': predictions,
    'Prediction_Probability': probabilities_list
})

# Save the results to a CSV file
output_file = 'test_predictions_results.csv'
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to {output_file}")



