import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, f1_score
from ESM_PepNet_LSTM_hotfigure import AIMP
import random

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
model = model.to(device)


# First, let's fix the generate_saliency_map function
def generate_saliency_map(model, sequence, device):
    # Store original training state
    was_training = model.training

    # Temporarily set to training mode for backward pass
    model.train()

    # Properly format the sequence for the model
    seq_list = [sequence]

    # Get one-hot encoding
    one_hot = get_onehot_features(seq_list)
    one_hot_tensor = torch.tensor(one_hot, device=device, dtype=torch.float)
    one_hot_tensor.requires_grad_()

    # Forward pass
    output = model(seq_list, one_hot_tensor)
    pred_prob = output[0, 1]  # Class 1 probability

    # Backward pass to get gradient
    pred_prob.backward()

    # Get saliency map (average across one-hot dimensions)
    saliency = one_hot_tensor.grad.data.abs().mean(dim=2).squeeze().cpu().numpy()

    # Normalize saliency values between 0 and 1
    if saliency.max() != saliency.min():  # Avoid division by zero
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    # Restore original training state
    model.train(was_training)

    return saliency


def plot_saliency_map(sequence, saliency_map, title=None, save_path=None, colormap='viridis'):
    """
    Plot saliency map for a protein sequence with customizable colormap.

    Parameters:
    -----------
    sequence : str
        The protein sequence
    saliency_map : numpy.ndarray
        The saliency values for each position
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure
    colormap : str, optional
        Matplotlib colormap to use (default: 'viridis')
        Options include: 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                         'Blues', 'Greens', 'Reds', 'YlOrRd', 'RdPu',
                         'coolwarm', 'RdBu', 'seismic', 'jet', 'rainbow'
    """
    # Trim to actual sequence length or saliency map length
    seq_len = min(len(sequence), len(saliency_map))
    sequence = sequence[:seq_len]
    saliency_map = saliency_map[:seq_len]

    # Create position indices
    positions = list(range(seq_len))

    plt.figure(figsize=(max(10, seq_len / 5), 3))  # Adjust figure size based on sequence length

    # Plot heatmap with specified colormap
    plt.imshow([saliency_map], cmap=colormap, aspect='auto')
    plt.colorbar(label='Saliency', orientation='vertical', shrink=0.8)

    # Set x-axis ticks and labels
    plt.xticks(positions, list(sequence), fontsize=10, rotation=45 if seq_len > 20 else 0)
    plt.yticks([])  # Hide y-axis ticks

    # Add grid lines to separate amino acids
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add title
    if title:
        plt.title(title)
    else:
        plt.title('Saliency Map for Protein Sequence')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_saliency_for_sequences(model, dataset, num_examples=5, save_dir=None, colormap='viridis',
                                     random_seed=None):
    """
    Visualize saliency maps for randomly selected sequences from the dataset.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    dataset : Dataset
        The dataset containing sequences and labels
    num_examples : int, optional
        Number of examples to visualize
    save_dir : str, optional
        Directory to save visualizations
    colormap : str, optional
        Matplotlib colormap to use
    random_seed : int, optional
        Seed for random number generator (for reproducibility)
    """
    model.eval()

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Randomly select indices from the dataset
    dataset_size = len(dataset)
    if num_examples >= dataset_size:
        # If requesting more examples than available, use all indices
        selected_indices = list(range(dataset_size))
    else:
        # Randomly sample without replacement
        selected_indices = random.sample(range(dataset_size), num_examples)

    for i, idx in enumerate(selected_indices):
        sequence, label = dataset[idx]
        original_seq = sequence.replace('X', '')  # Remove padding for display

        # Generate saliency map
        saliency_map = generate_saliency_map(model, sequence, device)

        # Create title with prediction info and dataset index
        with torch.no_grad():
            one_hot = get_onehot_features([sequence])
            one_hot_tensor = torch.tensor(one_hot, device=device, dtype=torch.float)
            output = model([sequence], one_hot_tensor)
            pred_class = output.argmax(dim=1).item()
            pred_prob = output[0, pred_class].item()

        title = f"Sample #{idx} - True: {label}, Pred: {pred_class} (Prob: {pred_prob:.2f})"

        # Save path if requested
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/saliency_sample_{idx}.png"

        # Plot saliency map
        plot_saliency_map(original_seq, saliency_map, title=title, save_path=save_path, colormap=colormap)


def perform_feature_occlusion(model, sequence, device, threshold, mask_token='X'):
    """
    Perform feature occlusion by masking important regions identified by saliency map.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    sequence : str
        The protein sequence
    device : torch.device
        Device to run the model on
    threshold : float
        Threshold for determining important positions (0.0-1.0)
    mask_token : str
        Character used to mask important positions

    Returns:
    --------
    original_output : torch.Tensor
        Original model output
    occluded_output : torch.Tensor
        Model output after occlusion
    saliency_map : numpy.ndarray
        Saliency values for each position
    occluded_sequence : str
        Sequence with important positions masked
    """
    # Get original saliency map
    saliency_map = generate_saliency_map(model, sequence, device)

    # Get original prediction
    with torch.no_grad():
        one_hot = get_onehot_features([sequence])
        one_hot_tensor = torch.tensor(one_hot, device=device, dtype=torch.float)
        original_output = model([sequence], one_hot_tensor)

    # Find positions to occlude based on saliency threshold
    important_positions = np.where(saliency_map > threshold)[0]

    # Create occluded sequence
    seq_chars = list(sequence)
    for pos in important_positions:
        if pos < len(seq_chars):
            seq_chars[pos] = mask_token
    occluded_sequence = ''.join(seq_chars)

    # Get prediction with occluded sequence
    with torch.no_grad():
        occluded_one_hot = get_onehot_features([occluded_sequence])
        occluded_one_hot_tensor = torch.tensor(occluded_one_hot, device=device, dtype=torch.float)
        occluded_output = model([occluded_sequence], occluded_one_hot_tensor)

    return original_output, occluded_output, saliency_map, occluded_sequence, important_positions


def plot_occlusion_comparison(sequence, occluded_sequence, saliency_map, original_output, occluded_output,
                              important_positions, title=None, save_path=None, colormap='coolwarm'):
    """
    Plot comparison between original sequence and occluded sequence saliency maps.

    Parameters:
    -----------
    sequence : str
        The original protein sequence
    occluded_sequence : str
        The sequence with important positions masked
    saliency_map : numpy.ndarray
        Saliency values for each position
    original_output : torch.Tensor
        Original model output
    occluded_output : torch.Tensor
        Model output after occlusion
    important_positions : numpy.ndarray
        Indices of important positions that were masked
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure
    colormap : str, optional
        Matplotlib colormap to use
    """
    # Remove padding characters for display
    original_seq = sequence.replace('X', '')

    # Trim to actual sequence length
    seq_len = len(original_seq)
    saliency_map = saliency_map[:seq_len]

    # Get prediction information
    original_pred_class = original_output.argmax(dim=1).item()
    original_pred_prob = original_output[0, original_pred_class].item()

    occluded_pred_class = occluded_output.argmax(dim=1).item()
    occluded_pred_prob = occluded_output[0, occluded_pred_class].item()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, seq_len / 4), 7), sharex=True)

    # Plot original saliency map
    im1 = ax1.imshow([saliency_map], cmap=colormap, aspect='auto')
    ax1.set_yticks([])
    ax1.set_title(f"Original Sequence - Pred: {original_pred_class} (Prob: {original_pred_prob:.2f})")

    # Add colored grid lines to highlight important positions
    for pos in range(seq_len):
        ax1.axvline(x=pos - 0.5, color='black', linestyle='-', alpha=0.1)
    for pos in important_positions:
        if pos < seq_len:
            ax1.axvline(x=pos - 0.5, color='black', linestyle='-', alpha=0.8)
            ax1.axvline(x=pos + 0.5, color='black', linestyle='-', alpha=0.8)

    # Calculate occlusion effect (difference in probability)
    prob_diff = original_pred_prob - occluded_pred_prob

    # Plot occluded sequence
    # Create a visualization of occlusion effect
    occlusion_effect = np.zeros(seq_len)
    occlusion_effect[important_positions[important_positions < seq_len]] = 1

    im2 = ax2.imshow([occlusion_effect], cmap='Reds', aspect='auto')
    ax2.set_yticks([])
    ax2.set_title(
        f"Occluded Positions - New Pred: {occluded_pred_class} (Prob: {occluded_pred_prob:.2f}, Δ: {prob_diff:.2f})")

    # Set x-axis ticks and labels on the bottom subplot
    positions = list(range(seq_len))
    ax2.set_xticks(positions)
    ax2.set_xticklabels(list(original_seq), rotation=45 if seq_len > 20 else 0)

    # Add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Saliency')

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=0.9)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_occlusion_experiments(model, dataset, num_examples=5, threshold=0.7,
                                    save_dir=None, colormap='coolwarm', random_seed=None):
    """
    Visualize feature occlusion experiments for randomly selected sequences.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    dataset : Dataset
        The dataset containing sequences and labels
    num_examples : int, optional
        Number of examples to visualize
    threshold : float, optional
        Threshold for determining important positions (0.0-1.0)
    save_dir : str, optional
        Directory to save visualizations
    colormap : str, optional
        Matplotlib colormap to use
    random_seed : int, optional
        Seed for random number generator (for reproducibility)
    """
    model.eval()

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Randomly select indices from the dataset
    dataset_size = len(dataset)
    if num_examples >= dataset_size:
        selected_indices = list(range(dataset_size))
    else:
        selected_indices = random.sample(range(dataset_size), num_examples)

    for i, idx in enumerate(selected_indices):
        sequence, label = dataset[idx]

        # Perform feature occlusion
        original_output, occluded_output, saliency_map, occluded_sequence, important_positions = (
            perform_feature_occlusion(model, sequence, device, threshold=threshold)
        )

        # Create title with dataset index and original label
        title = f"Sample #{idx} - True Label: {label}"

        # Save path if requested
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/occlusion_sample_{idx}.png"

        # Plot occlusion comparison
        plot_occlusion_comparison(
            sequence, occluded_sequence, saliency_map,
            original_output, occluded_output, important_positions,
            title=title, save_path=save_path, colormap=colormap
        )


if __name__ == "__main__":
    # Create directory for saving visualizations
    import os

    save_dir = "saliency_maps"
    os.makedirs(save_dir, exist_ok=True)

    # Set parameters
    selected_colormap = 'coolwarm'
    random_seed = 42
    num_samples = 3

    # 1. First, visualize standard saliency maps
    print("Generating standard saliency maps...")
    visualize_saliency_for_sequences(
        model=model,
        dataset=val_data,
        num_examples=num_samples,
        save_dir=save_dir,
        colormap=selected_colormap,
        random_seed=random_seed
    )

    # 2. Then, perform occlusion experiments
    print("\nPerforming feature occlusion experiments...")
    # Set a threshold value for determining important positions
    occlusion_threshold = 0.5  # Positions with saliency > 0.7 will be masked

    # Create a separate directory for occlusion experiment results
    occlusion_dir = os.path.join(save_dir, "occlusion_experiments")
    os.makedirs(occlusion_dir, exist_ok=True)

    # Run occlusion experiments with the same random seed for consistency
    visualize_occlusion_experiments(
        model=model,
        dataset=val_data,
        num_examples=num_samples,
        threshold=occlusion_threshold,
        save_dir=occlusion_dir,
        colormap=selected_colormap,
        random_seed=random_seed  # Use same seed to get same samples as before
    )

    # 3. Additional experiment with different thresholds (optional)
    print("\nTesting different occlusion thresholds on a positive sample...")

    # Find positive samples
    positive_indices = [i for i in range(len(val_data)) if val_data[i][1] == 1]

    if positive_indices:
        # Select the first positive sample
        # You could also use random.choice(positive_indices) to select a random positive sample
        example_idx = positive_indices[0]
        sequence, label = val_data[example_idx]

        print(f"Selected positive sample #{example_idx} for threshold experiments")

        # Try different thresholds
        for threshold in [0.5, 0.7, 0.9]:
            original_output, occluded_output, saliency_map, occluded_sequence, important_positions = (
                perform_feature_occlusion(model, sequence, device, threshold=threshold)
            )

            # Get prediction probability
            with torch.no_grad():
                one_hot = get_onehot_features([sequence])
                one_hot_tensor = torch.tensor(one_hot, device=device, dtype=torch.float)
                output = model([sequence], one_hot_tensor)
                pred_class = output.argmax(dim=1).item()
                pred_prob = output[0, pred_class].item()

            title = f"Positive Sample #{example_idx} - Occlusion Threshold: {threshold} (Prob: {pred_prob:.2f})"
            save_path = f"{occlusion_dir}/positive_sample_{example_idx}_threshold_{threshold}.png"

            plot_occlusion_comparison(
                sequence, occluded_sequence, saliency_map,
                original_output, occluded_output, important_positions,
                title=title, save_path=save_path, colormap=selected_colormap
            )
    else:
        print("No positive samples found in the dataset.")