import torch
from torch import nn
import esm

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)

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
        x = self.linear(x.permute(0, 2, 1))
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
        self.cnn = CNN(input_dim=feas_dim, hidden_dim=hidden, kernel_size=3, dropout=dropout)

        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.embedding = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.bn = nn.ModuleList([nn.BatchNorm1d(pre_feas_dim),
                                 nn.BatchNorm1d(feas_dim)])

        self.n_lstm = n_lstm

        self.tcn = TCN(feas_dim, hidden, [hidden // 2, hidden // 2, hidden // 2], 23, dropout)
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
        self.clf = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2))

    def forward(self, peptide_sequence, one_hot_embedding):
        bert_output = self.esm(peptide_sequence)
        pre_feas = self.bn[0](bert_output.permute(0, 2, 1)).permute(0, 2, 1)
        feas = self.bn[1](one_hot_embedding.permute(0, 2, 1)).permute(0, 2, 1)

        # 添加 CNN 模块
        cnn_out = self.cnn(feas)

        pre_feas = self.pre_embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)

        tcn_out = self.tcn(cnn_out)  # 将 CNN 输出作为 TCN 输入
        tcn_out = self.tcn_res(torch.cat([tcn_out, cnn_out], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        feas_em = self.embedding(torch.cat([pre_feas, tcn_out], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        transformer_out = self.lstm(feas_em)
        transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_res(torch.cat([transformer_out, feas_em], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_pool(transformer_out).squeeze(1)

        out = self.clf(transformer_out)
        out = torch.nn.functional.softmax(out, -1)

        return out