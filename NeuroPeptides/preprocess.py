import pickle

import h5py
import numpy as np
import os


# parse_fasta_predict 函数:用于从FASTA文件中加载蛋白质序列，但不加载标签。
# 参数:
#       dir: FASTA文件的路径。
#       number: 读取的序列数量，如果为None，则读取所有序列。
# 返回:
#       names: 序列的名称。
#       sequences: 序列的氨基酸序列。
def parse_fasta_predict(dir, number=None):
    names = []
    sequences = []
    if number is None:
        number = -1
    with open(dir, 'r') as f:
        data = f.readlines()
        for i in range(0, len(data[:number]), 2):
            line = data[i]
            if line.startswith('>'):
                names.append(line.strip()[1:])
                sequences.append(data[i + 1].strip())
    return np.array(names), np.array(sequences)


# parse_fasta函数: 用于从FASTA文件中加载蛋白质序列及其标签。
# 参数:
#   dir: FASTA文件的路径。
#   number: 读取的序列数量，如果为None，则读取所有序列。
# 返回:
#   names: 序列的名称。
#   sequences: 序列的氨基酸序列。
#   labels: 序列的标签。
def parse_fasta(dir, number=None):
    names = []
    sequences = []
    labels = []
    if number is None:
        number = -1
    with open(dir, 'r') as f:
        data = f.readlines()
        for i in range(0, len(data[:number]), 2):
            line = data[i]
            if line.startswith('>'):
                label = int(line.split('|')[-1])
                labels.append(label)
                names.append(line.strip()[1:])
                sequences.append(data[i + 1].strip())
    return np.array(names), np.array(sequences), np.array(labels)

# get_pretrained_features函数： 从HDF5文件中加载预训练的特征，并截取前theta个氨基酸的特征。
# 参数:
# names: 序列的名称。
# sequences: 序列的氨基酸序列。
# pre_dict_path: 预训练特征的HDF5文件路径。
# theta: 截取的氨基酸数量，默认为50。
# 返回:
# features: 截取后的特征矩阵。
def get_pretrained_features(names, sequences, pre_dict_path, theta=50):
    features = []
    with h5py.File(pre_dict_path, "r") as h5fi:
        for name, seq in zip(names, sequences):
            pre_features_ref = h5fi[name][:]
            sequence_len = len(seq)
            feature = np.zeros((theta, pre_features_ref.shape[-1]), dtype=float)
            for i in range(min(sequence_len, theta)):
                feature[i, :] = pre_features_ref[i, :]
            features.append(feature)
    features = np.array(features)

    return features

def get_pretrained_features_predict(names, sequences, pre_dict_path, theta=50):
    features = []
    with h5py.File(pre_dict_path, "r") as h5fi:
        for name, seq in zip(names, sequences):
            pre_features_ref = h5fi[name][:]
            sequence_len = len(seq)
            feature = np.zeros((theta, pre_features_ref.shape[-1]), dtype=float)
            for i in range(min(sequence_len, theta)):
                feature[i, :] = pre_features_ref[i, :]
            features.append(feature)
    features = np.array(features)

    return features

# get_onehot_features函数：生成序列的One-Hot编码特征
# 参数:
#   sequences: 序列的氨基酸序列
#   theta: 截取的氨基酸数量，默认为50
# 返回:
#   features: One-Hot编码的特征矩阵
# def get_onehot_features(sequences, theta=50):
#     # if not os.path.exists(out_path):
#     element = 'ALRKNMDFCPQSETGWHYIV'
#     onehot_ref = np.eye(20)
#     features = []
#     for seq in sequences:
#         sequence_len = len(seq)
#         feature = np.zeros((theta, 20), dtype=float)
#         for i in range(min(sequence_len, theta)):
#             aa = seq[i]
#             index = element.find(aa)
#             feature[i, :] = onehot_ref[index, :]
#         features.append(feature)
#     features = np.array(features)
#     print(features.shape)
#     print(features)
#     return features
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
    print(features.shape)
    print(features)
    return features


def get_properties_features(sequences, theta=50):
    # if not os.path.exists(out_path):
    with open(f'../datasets/properties.pkl', 'rb') as f:
        properties = pickle.load(f)
    features = []
    for seq in sequences:
        sequence_len = len(seq)
        feature = np.zeros((theta, 14), dtype=float)
        for i in range(min(sequence_len, theta)):
            aa = seq[i]
            feature[i, :] = properties[aa]
        features.append(feature)
    features = np.array(features)
    # with open(out_path, 'wb') as f:
    #     pickle.dump(features, f)
    # else:
    #     with open(out_path, 'rb') as f:
    #         features = pickle.load(f)

    return features


def get_seq_features(names, out_path, pre_dict_path, theta=50):
    with open(pre_dict_path, 'rb') as f:
        pre_features_dict = pickle.load(f)
    features = []
    for name in names:
        pre_features_ref = pre_features_dict[name]
        features.append(pre_features_ref[None, :])
    features = np.array(features)
    return features


def get_seqs_len(sequences):
    # sequences, labels = load_fasta(dir)
    seqs_len = []
    for seq in sequences:
        seqs_len.append(len(seq))
    return seqs_len


def get_blosum_features(sequences, out_path, blosum_path, theta=50):
    '''
      Get PSSM features of each sequence according to directory acp_dir and pssm_dir.
      While some sequences may not have PSSM files, their features will be replaced by 0 matrices.

      :param acp_dir:     directory of acp dataset
      :param pssm_dir:    directory of generated PSSM features
      :param theta:       intercept the first theta amino acids of the sequence
      :return:            PSSM features of size acp length * theta * 20
      '''
    if not os.path.exists(out_path):
        with open(f'{blosum_path}', 'rb') as f:
            blosum_dict = pickle.load(f)
        features = []
        for seq in sequences:
            feature = np.zeros((theta, 20))
            for index, amino in enumerate(seq):
                feature[index] = blosum_dict[amino]
                if index == theta:
                    break
            feature = np.array(feature)
            features.append(feature)
        features = np.array(features)[:, :theta]
        with open(out_path, 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(out_path, 'rb') as f:
            features = pickle.load(f)
    return features

if __name__ == '__main__':
    get_onehot_features('SYSMEHFRWGKPVGRKRRPVKVYTSNGVEEESAEVFPGEMX',100)


