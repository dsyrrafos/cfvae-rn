import os
import warnings
import argparse
import numpy as np
import pickle as pkl
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import random_split

from helpers import train_rnvae_epoch, eval_rnvae_epoch, vae_beta_schedule, define_name_rnvae, define_results_name, set_seed
from utils import plot_loss_components_rnvae
from dataset import DatanetConcatDataset, DatanetDataset, get_dataloader_with_sampling
from autoencoder import LearntPriorVariationalRouteNet, FlowLearntPriorVariationalRouteNet

warnings.simplefilter(action='ignore', category=FutureWarning)

# Argument parser
parser = argparse.ArgumentParser(description='Train and Evaluate the Variational RouteNet Autoencoder (RN-VAE)')
parser.add_argument('--norm-flow', action='store_true', default=False, help='Use normalizing flows in the VAE')
parser.add_argument('--flow-type', type=str, default='planar', choices=['planar', 'affine_coupling'], help='Type of normalizing flow to use in the VAE')
parser.add_argument('--n-flows', type=int, default=4, help='Number of flow layers in the normalizing flow')
parser.add_argument('--lr-scheduler', type=str, default='plateau', choices=['step', 'plateau'], help='Type of learning rate scheduler to use')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau scheduler')
parser.add_argument('--patience-lr', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler')
parser.add_argument('--step-size', type=int, default=10, help='Step size for StepLR scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size of 1 due to variable size graphs')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--train-steps', type=int, default=1000, help='Number of training steps per epoch')
parser.add_argument('--eval-steps', type=int, default=500, help='Number of evaluation steps per epoch')
parser.add_argument('--reduction', type=str, default='mean', choices=['mean', 'sum'], help='Reduction method for loss calculation')
parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension of RouteNet encoder size')
parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension size for variational part')
parser.add_argument('--beta', type=float, default=1e-3, help='Initial beta value for VAE')
parser.add_argument('--beta-schedule', type=str, default='sigmoid', choices=['constant', 'linear', 'sigmoid', 'cyclical'], help='Beta schedule type for VAE')
parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio from training set')
parser.add_argument('--n-samples', type=int, default=10, help='Number of samples for prediction')
parser.add_argument('--variational', action='store_true', default=False, help='Use variational autoencoder')
parser.add_argument('--conditional', action='store_true', default=False, help='Use conditional autoencoder')
parser.add_argument('--prefix', type=str, default='RN', help='Prefix for model name')
parser.add_argument('--train', action='store_true', default=False, help='Train the model')
parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the model')
parser.add_argument('--dataset', nargs="+", choices=['nsfnetbw', 'gbnbw', 'geant2bw'], help='Datasets to use', required=True)
parser.add_argument('--normalize', nargs="+", choices=['nsfnetbw', 'gbnbw', 'geant2bw'], help='Datasets to use', required=True)
parser.add_argument('--log-transform', action='store_true', default=False, help='Apply log transform to target variables')
parser.add_argument('--standardize', action='store_true', default=False, help='Standardize target variables')
parser.add_argument('--targets', nargs="+", choices=['delay', 'jitter', 'loss'], default=['delay'], help='Target variables to predict')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
print('Working directory: ', os.getcwd())

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Torch version: ', torch.__version__)
print('CUDA version: ', torch.version.cuda)
print("Using device: "+str(device))

# Define project root dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set numpy random seed for reproducibility
set_seed(args.seed)
print('Random seed set to: ', args.seed)

# Define directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'v1')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
OUTPUTS_DIR = os.path.join(DATA_DIR, 'routenet')

# Create directories if they don't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Load dataset
train_dataset_list = []
test_dataset_list = []
for ds in args.dataset:
    dataset_prefix = ds
    if args.perturbation_feature is not None:
        dataset_prefix += f"_perturbed_{args.perturbation_feature}_nl{args.noise_level}"
    print(f"Using {dataset_prefix} dataset")
    train_path = os.path.join(DATA_DIR, 'routenet', 'padded_datasets', f'{dataset_prefix}_train_dataset.pkl')
    test_path = os.path.join(DATA_DIR, 'routenet', 'padded_datasets', f'{dataset_prefix}_test_dataset.pkl')
    with open(train_path, 'rb') as handle:
        train_dataset = pkl.load(handle)
    with open(test_path, 'rb') as handle:
        test_dataset = pkl.load(handle)

    print(f"Dataset loaded from {train_path} and {test_path}")
    print(f"Train set size: {len(train_dataset)}, type: {type(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}, type: {type(test_dataset)}")

    # Normalize features
    norm_prefix = '_'.join([ds for ds in args.normalize])
    stats_path = os.path.join(DATA_DIR, 'routenet', 'datasets', f'{norm_prefix}_dataset_stats.pkl')
    print(f"Normalizing {ds} dataset with stats from {stats_path}")

    with open(stats_path, 'rb') as handle:
        stats = pkl.load(handle)
    for feature in ['traffic', 'capacity']:
        mean, std = stats[f'{feature}_mean'], stats[f'{feature}_std']
        # Normalize the dataset
        train_dataset = train_dataset.standarize(feature, mean, std)
        test_dataset = test_dataset.standarize(feature, mean, std)
        print(f"Dataset '{feature}' normalized with mean: {mean}, std: {std}")

    # Keep only specified target variables
    if args.targets:
        train_dataset = train_dataset.keep_only_targets(args.targets)
        test_dataset = test_dataset.keep_only_targets(args.targets)

    # Apply log transform if specified
    if args.log_transform:
        for feature in args.targets:
            train_dataset = train_dataset.log_transform(feature)
            test_dataset = test_dataset.log_transform(feature)
            print(f"Applied log transform to feature: {feature}")
    elif args.standardize:
        for feature in args.targets:
            mean, std = stats[f'{feature}_mean'], stats[f'{feature}_std']
            train_dataset = train_dataset.standarize(feature, mean, std, target=True)
            test_dataset = test_dataset.standarize(feature, mean, std, target=True)
            print(f"Standardized feature: {feature} with mean: {mean}, std: {std}")
    else:
        # Transform lists to numpy arrays for consistency (labels are stored as lists)
        for feature in args.targets:
            train_dataset = train_dataset.list_to_numpy(feature)
            test_dataset = test_dataset.list_to_numpy(feature)
            print(f"Transformed lists to numpy arrays for feature: {feature}")
    
    train_dataset_list.append(train_dataset)
    test_dataset_list.append(test_dataset)

# Combine datasets if multiple are provided
if len(train_dataset_list) > 1:
    print("Combining multiple datasets...")
    train_dataset = DatanetConcatDataset(train_dataset_list)
    test_dataset = DatanetConcatDataset(test_dataset_list)
    print(f"Combined train set size: {len(train_dataset)}, type: {type(train_dataset)}")
    print(f"Combined test set size: {len(test_dataset)}, type: {type(test_dataset)}")
    dataset_prefix = '_'.join([ds for ds in args.dataset])
else:
    train_dataset = train_dataset_list[0]
    test_dataset = test_dataset_list[0]
    dataset_prefix = args.dataset[0]

print(f"Using dataset prefix: {dataset_prefix}")
print('Target variables:', args.targets)

# Train validation split
val_size = int(args.val_ratio * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

# Create DataLoader for train, validation and test datasets
train_loader = get_dataloader_with_sampling(train_dataset, 'routenet', args.train_steps, shuffle=True, batch_size=args.batch_size)
val_loader = get_dataloader_with_sampling(val_dataset, 'routenet', args.eval_steps, shuffle=True, batch_size=args.batch_size)
test_loader = get_dataloader_with_sampling(test_dataset, 'routenet', args.eval_steps, shuffle=False, batch_size=args.batch_size)

# Determine number of links and paths in the dataset
l_set = set([len(data[0]['path_to_link']) for data in train_dataset])
n_links = max(l_set)
print('Number of unique link counts in training set:', len(l_set))
print('Unique link counts in training set:', l_set)
print('Max number of links:', n_links)
p_set = set([len(data[0]['link_to_path']) for data in train_dataset])
n_paths = max(p_set)
print('Number of unique path counts in training set:', len(p_set))
print('Unique path counts in training set:', p_set)
print('Max number of paths:', n_paths)

# Define model name based on arguments
name = define_name_rnvae(args)

# Define model paths
model_dir = os.path.join(MODELS_DIR, 'autoencoder')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_name = os.path.join(model_dir, f'{name}.pth.tar')

if args.train:
    print("Saving model to", model_name)
    # Define loss curves directory and figure name
    loss_curves_dir = os.path.join(FIGURES_DIR, 'autoencoder')
    # Create autoencoder figures directory if it doesn't exist
    if not os.path.exists(loss_curves_dir):
        os.makedirs(loss_curves_dir)
    figure_name = os.path.join(loss_curves_dir, f'{name}.png')
    print("Plotting loss components to", figure_name)

if args.eval:
    print("Loading model from", model_name)
    # Define prediction paths
    PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, 'predictions')
    prediction_dir = os.path.join(PREDICTIONS_DIR, 'autoencoder')
    # Create predictions directory if it doesn't exist
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    LABELS_DIR = os.path.join(OUTPUTS_DIR, 'labels')
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)

    prediction_name = define_results_name(name, args)
    prediction_name = os.path.join(prediction_dir, f'{prediction_name}.npy')
    label_name = define_results_name(name, args, type='labels')
    label_name = os.path.join(LABELS_DIR, f'{label_name}.npy')
    print("Saving predictions to", prediction_name)
    print("Saving labels to", label_name)

# Define KLD path threshold
kld_threshold = args.latent_dim * 0.5 # Threshold for KLD path to consider the model trained
print("KLD path threshold for considering the model trained:", kld_threshold)
print(f'Maximum Beta value: {args.beta}')

# Load Routenet Predictor for conditional VAE
if args.conditional:
    from routenet import RouteNetPredictor
    print("Using RouteNet as conditional encoder")
    routenet = RouteNetPredictor(
        input_dim_path=1,
        input_dim_link=1,
        output_dim=len(args.targets),
        iterations=4,
        link_state_dim=args.hidden_dim,
        path_state_dim=args.hidden_dim,
        aggr_dropout=0.0,
        readout_dropout=0.0,
        device=device
    ).to(device)
else:
    routenet = None
    print("Not using conditional encoder")

# Define autoencoder model
if args.variational:
    print("Using Variational Autoencoder")
    if args.norm_flow:
        print(f"Using learned prior network with normalizing flows for the posterior ({args.flow_type}, {args.n_flows} layers)")
        autoencoder = FlowLearntPriorVariationalRouteNet(
            flow_type=args.flow_type,
            n_flows=args.n_flows,
            input_dim_path=1,
            input_dim_link=1,
            output_dim=len(args.targets),
            iterations=4,
            link_state_dim=args.hidden_dim,
            path_state_dim=args.hidden_dim,
            aggr_dropout=0.0,
            readout_dropout=0.0,
            latent_dim=args.latent_dim,
            conditional_encoder=routenet if args.conditional else None,
            device=device
        ).to(device)
    else:
        print("Using standard VAE with learned prior network")
        autoencoder = LearntPriorVariationalRouteNet(
            input_dim_path=1,
            input_dim_link=1,
            output_dim=len(args.targets),
            iterations=4,
            link_state_dim=args.hidden_dim,
            path_state_dim=args.hidden_dim,
            aggr_dropout=0.0,
            readout_dropout=0.0,
            latent_dim=args.latent_dim,
            conditional_encoder=routenet if args.conditional else None,
            device=device
        ).to(device)
else:
    raise ValueError("Autoencoder must be variational. Set --variational flag to use VAE.")

# Define optimizer and learning rate scheduler
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=args.lr)
if args.lr_scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience_lr)

# Train autoencoder
if args.train:
    # Print number of parameters
    total_params_autoenc = sum(p.numel() for p in autoencoder.parameters())
    print("Number of Autoencoder's total parameters: "+str(total_params_autoenc))
    # Print number of trainable parameters
    trainable_params_autoenc = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print("Number of Autoencoder's trainable parameters: "+str(trainable_params_autoenc))

    best_val_loss = np.inf
    best_epoch = -1
    train_loss_sum = []
    val_loss_sum = []
    train_loss_pred = []
    val_loss_pred = []
    train_loss_prior_pred = []
    val_loss_prior_pred = []
    train_loss_kld_path = []    
    val_loss_kld_path = []
    beta_list = []
    lr_list = []

    # Train loop
    print("Training autoencoder...")
    epochs_since_last_improvement = 0
    patience_countdown_started = False
    for epoch in range(1, args.epochs + 1):
        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Variational autoencoder training
        if args.variational:
            # Beta schedule
            beta_epoch=vae_beta_schedule(args.beta, epoch-1, args.epochs-1, type=args.beta_schedule)
            print(f'Epoch {epoch}/{args.epochs}, Beta: {beta_epoch:.8f}, Learning Rate: {scheduler.get_last_lr()[0]:.8f}')
            # Do one epoch of training
            train_loss_epoch, train_loss_pred_epoch, train_loss_prior_pred_epoch, train_loss_kld_path_epoch, train_count = train_rnvae_epoch(autoencoder, args.targets, train_loader, optimizer, args.variational, beta_epoch, reduction=args.reduction)
            # Do one epoch of validation
            val_loss_epoch, val_loss_pred_epoch, val_loss_prior_pred_epoch, val_loss_kld_path_epoch, val_count = eval_rnvae_epoch(autoencoder, args.targets, val_loader, args.variational, beta_epoch, reduction=args.reduction)
            print(f'Train N: {train_count}, Val N: {val_count}, Train TL: {train_loss_epoch:.8f}, Val TL: {val_loss_epoch:.8f}, Train RL: {train_loss_pred_epoch:.8f}, Val RL: {val_loss_pred_epoch:.8f}, Train PL: {train_loss_prior_pred_epoch:.8f}, Val PL: {val_loss_prior_pred_epoch:.8f}, Train KLD: {train_loss_kld_path_epoch:.8f}, Val KLD: {val_loss_kld_path_epoch:.8f}')

            # Append losses to lists
            train_loss_sum.append(train_loss_epoch)
            val_loss_sum.append(val_loss_epoch)
            train_loss_pred.append(train_loss_pred_epoch)
            val_loss_pred.append(val_loss_pred_epoch)
            train_loss_prior_pred.append(train_loss_prior_pred_epoch)
            val_loss_prior_pred.append(val_loss_prior_pred_epoch)
            train_loss_kld_path.append(train_loss_kld_path_epoch)
            val_loss_kld_path.append(val_loss_kld_path_epoch)
            beta_list.append(beta_epoch)
            lr_list.append(scheduler.get_last_lr()[0])

            # Save model if validation prediction loss improved
            if best_val_loss >= val_loss_prior_pred_epoch and beta_epoch > 0:
                if val_loss_kld_path_epoch < kld_threshold:
                    print(f"Validation prediction loss improved from {best_val_loss:.8f} to {val_loss_prior_pred_epoch:.8f}")
                    best_val_loss = val_loss_prior_pred_epoch
                    best_epoch = epoch
                    epochs_since_last_improvement = 0
                    torch.save({
                        'state_dict': autoencoder.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'train_loss_sum': train_loss_sum,
                        'val_loss_sum': val_loss_sum,
                        'train_loss_pred': train_loss_pred,
                        'val_loss_pred': val_loss_pred,
                        'train_loss_kld_path': train_loss_kld_path,
                        'val_loss_kld_path': val_loss_kld_path,
                        'train_loss_prior_pred': train_loss_prior_pred,
                        'val_loss_prior_pred': val_loss_prior_pred,
                        'beta_list': beta_list,
                        'lr_list': lr_list,
                    }, model_name)
                    print(f"Model saved at epoch {epoch} with validation prediction loss {val_loss_prior_pred_epoch:.8f}")

            print(f'Epochs since last improvement: {epochs_since_last_improvement}, best validation prediction loss: {best_val_loss:.8f} at epoch {best_epoch}')

            # Early stopping if no improvement in validation loss after 5 epochs
            # Beta should be greater than 0 to allow early stopping
            if beta_epoch > 0 and val_loss_kld_path_epoch < kld_threshold:
                patience_countdown_started = True
            if patience_countdown_started:
                epochs_since_last_improvement += 1
            if epochs_since_last_improvement > args.patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                break
        else:
            raise NotImplementedError("Standard Autoencoder training not implemented in this script.")

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(val_loss_prior_pred_epoch)
    
    print(epoch, "Training finished.")
    plot_loss_components_rnvae(
        args.variational,
        figure_name,
        epoch,
        best_epoch,
        beta_list,
        lr_list,
        train_loss_sum,
        val_loss_sum,
        train_loss_pred,
        val_loss_pred,
        train_loss_prior_pred,
        val_loss_prior_pred,
        train_loss_kld_path,
        val_loss_kld_path,
        show=True
    )
else:
    model_dict = {}
    # for name, model_name in [('overall', model_name_all), ('prediction', model_name_pred)]:
    checkpoint = torch.load(os.path.join(MODELS_DIR, model_name), map_location=device)
    autoencoder.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model_dict[name] = autoencoder
    print("Autoencoder model loaded successfully.")
    print("Model name:", model_name)

# Evaluate autoencoder on test set and save predictions
if args.eval:
    print("Evaluating on test set...")
    for name, autoencoder in model_dict.items():
        autoencoder.to(device)
        print(f"Evaluating model: {name}")
        print(f"Saving predictions to: {prediction_name}")
        autoencoder.eval()
        predictions = []
        labels_list = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                instance_predictions = []
                for i in range(args.n_samples):
                    inputs, labels = data
                    outputs = autoencoder(data)
                    instance_predictions.append(outputs)
                instance_predictions_tensor = torch.stack(instance_predictions, dim=1)
                predictions.append(instance_predictions_tensor)
                ground_truth = torch.stack([labels[t] for t in args.targets], dim=2).squeeze().float().to(device)
                labels_list.append(ground_truth)

        print("Saving predictions...")
        predictions = torch.cat(predictions).squeeze().cpu().numpy()
        labels_tensor = torch.cat(labels_list).squeeze().cpu().numpy()
        print("Predictions shape:", predictions.shape)
        print("Labels shape:", labels_tensor.shape)
        if args.log_transform:
            # Inverse log transform
            predictions = np.expm1(predictions)
            labels_tensor = np.expm1(labels_tensor)
        elif args.standardize:
            for i, feature in enumerate(args.targets):
                mean, std = stats[f'{feature}_mean'], stats[f'{feature}_std']
                predictions = predictions * std + mean
                labels_tensor = labels_tensor * std + mean
        else:
            pass

        # Save predictions and labels
        np.save(prediction_name, predictions)
        print("Predictions saved to: ", prediction_name)

        np.save(label_name, labels_tensor)
        print("Labels saved to: ", label_name)