import math

import torch
from torch import nn
import esm
from transformers import T5Tokenizer, T5EncoderModel





class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2),
                       nn.ReLU(),
                       nn.Dropout(dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))
        x = self.linear(x.permute(0, 2, 1))    # Reshape for linear layer
        return x

class ESM(nn.Module):
    def __init__(self):
        super(ESM, self).__init__()
        self.esm_model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, prot_seqs):
        data = [('seq{}'.format(i), seq) for i, seq in enumerate(prot_seqs)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33][:, 1:-1]
        prot_embedding = token_representations
        return prot_embedding

class T5FeatureExtractor(nn.Module):
    def __init__(self):
        super(T5FeatureExtractor, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # 将模型移动到指定设备

    def forward(self, prot_seqs):
        inputs = self.tokenizer(prot_seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 将输入数据移动到指定设备
        with torch.no_grad():
            outputs = self.model(**inputs)
        t5_features = outputs.last_hidden_state[:, 0, :]  # 取[CLS]标记的特征
        return t5_features



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        return lstm_out


class AIMP(torch.nn.Module):
    def __init__(self, pre_feas_dim, feas_dim, hidden, n_lstm, dropout):
        super(AIMP, self).__init__()

        self.esm = ESM()

        self.t5_extractor = T5FeatureExtractor()
        # 添加一个线性层来调整T5特征的维度
        self.t5_linear = nn.Linear(1024, hidden)  # 假设T5模型输出的特征维度为1024


        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # self.embedding = nn.Sequential(
        #     nn.Conv1d(hidden * 2, hidden, kernel_size=1),
        #     nn.BatchNorm1d(hidden),
        #     nn.ELU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(hidden, hidden, kernel_size=1),
        # )

        # 修改合并特征后的嵌入层
        self.embedding = nn.Sequential(
            nn.Conv1d(hidden * 3, hidden, kernel_size=1),  # 修改输入通道数为hidden*3
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.bn = nn.ModuleList([nn.BatchNorm1d(pre_feas_dim),
                                 nn.BatchNorm1d(feas_dim),
                                 # nn.BatchNorm1d(seq_feas_dim),
                                 ])

        self.n_lstm = n_lstm

        self.tcn = TCN(feas_dim, hidden, [hidden // 2, hidden // 2, hidden // 2], 21, dropout)
        self.tcn_res = nn.Sequential(
            nn.Conv1d(hidden + feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.lstm = LSTMModel(input_dim=hidden, hidden_dim=hidden, num_layers=n_lstm, dropout=dropout)

        self.transformer_act = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden + hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_pool = nn.AdaptiveAvgPool2d((1, None))
        # self.clf = nn.Sequential(
        #     nn.Linear(hidden, hidden),
        #     nn.BatchNorm1d(hidden),
        #     nn.ELU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden, 2),
        # )
        self.clf = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.pre_embedding, self.embedding, self.clf]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)
        for layer in [self.transformer_act, self.transformer_res]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def forward(self, peptide_sequence, one_hot_embedding):
        # batch_size = pre_feas.shape[0]
        bert_output = self.esm(peptide_sequence)
        pre_feas = self.bn[0](bert_output.permute(0, 2, 1)).permute(0, 2, 1)
        pre_feas = self.pre_embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)


        feas = self.bn[1](one_hot_embedding.permute(0, 2, 1)).permute(0, 2, 1)
        tcn_out = self.tcn(feas)
        tcn_out = self.tcn_res(torch.cat([tcn_out, feas], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        # 获取T5特征
        t5_features = self.t5_extractor(peptide_sequence)
        t5_features = self.t5_linear(t5_features)  # 调整T5特征的维度
        t5_features = t5_features.unsqueeze(1)
        # 将 t5_features 扩展到 (batch_size, sequence_length, hidden_size)
        t5_features = t5_features.expand(-1, pre_feas.size(1), -1)


        # feas_em = self.embedding(torch.cat([pre_feas, tcn_out], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        # 将T5特征与pre_feas和tcn_out合并
        feas_em = self.embedding(torch.cat([pre_feas, tcn_out, t5_features], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        # transformer_out = self.lstm(feas_em) 消融LSTM
        # transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1) 消融LSTM

        transformer_out = self.transformer_act(feas_em.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_res(torch.cat([transformer_out, feas_em], dim=-1).permute(0, 2, 1)).permute(
            0, 2, 1)
        transformer_out = self.transformer_pool(transformer_out).squeeze(1)

        out = self.clf(transformer_out)
        out = torch.nn.functional.softmax(out, -1)

        # 返回中间层的输出tcn_out,bert_output是esm的特征,pre_feas是esm卷积之后的特征
        # feas_em是esm和onehot合并后的特征
        return out