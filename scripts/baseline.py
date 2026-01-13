import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from routenet import RouteNetPredictor
from dataset import DatanetDataset, DatanetConcatDataset, get_dataloader_with_sampling
from helpers import define_name_predictor, set_seed, define_results_name, train_predictor, evaluate_predictor
from utils import plot_loss_predictor

# Initiate parser and add arguments
parser = argparse.ArgumentParser(description='Train and evaluate RouteNet-Predictor model.')
parser.add_argument('--train', action='store_true', default=False, help='Flag to indicate training mode')
parser.add_argument('--eval', action='store_true', default=False, help='Flag to indicate evaluation mode')
parser.add_argument('--targets', nargs="+", choices=['delay', 'jitter', 'loss'], default=['delay'])
parser.add_argument('--dataset', nargs="+", choices=['nsfnetbw', 'gbnbw', 'geant2bw'], help='Datasets to use', required=True)
parser.add_argument('--normalize', nargs="+", choices=['nsfnetbw', 'gbnbw', 'geant2bw'], help='Datasets to use for normalization', required=True)
parser.add_argument('--log-transform', action='store_true', default=False, help='Apply log transform to target variables')
parser.add_argument('--standardize', action='store_true', default=False, help='Standardize target variables')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training and evaluation')
parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size for the model')
parser.add_argument('--prefix', type=str, default='ND3', help='Prefix for saving models and figures')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--train-steps', type=int, default=1000, help='Number of training steps per epoch')
parser.add_argument('--eval-steps', type=int, default=500, help='Number of evaluation steps per epoch')
parser.add_argument('--reduction', type=str, default='mean', choices=['mean', 'sum'], help='Reduction method for loss calculation')
parser.add_argument('--lr-scheduler', type=str, default='plateau', choices=['step', 'plateau'], help='Learning rate scheduler type')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau scheduler')
parser.add_argument('--patience-lr', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler')
parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate decay factor for StepLR scheduler')
parser.add_argument('--lr-step', type=int, default=10, help='Step size for StepLR scheduler')
parser.add_argument('--patience', type=int, default=25, help='Patience for early stopping')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    print('Working directory: ', os.getcwd())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Torch version: ', torch.__version__)
    print('CUDA version: ', torch.version.cuda)
    print('Using device: ', device)

    # Define project root dynamically
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Set numpy random seed for reproducibility
    set_seed(args.seed)
    print('Random seed set to: ', args.seed)

    # Define directories
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

    dataset_prefix = '_'.join(args.dataset)
    normalize_prefix = '_'.join(args.normalize)
    targets_prefix = '_'.join(args.targets)

    # Define model  name
    name = define_name_predictor(args)
    models_dir = os.path.join(PROJECT_ROOT, 'models', 'predictor')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f'Created models directory: {models_dir}')
    model_path = os.path.join(models_dir, f'{name}.pth')    

    if args.train:
        print(f'Model will be saved to: {model_path}')
        # Define loss curves directory and figure name
        loss_curves_dir = os.path.join(FIGURES_DIR, 'predictor')
        figure_name = os.path.join(loss_curves_dir, f'{name}.png')
        print(f'Loss curves will be saved to: {figure_name}')

    if args.eval:
        print(f'Model will be loaded from: {model_path}')
        # Define prediction and label paths
        prediction_name = define_results_name(name, args, type='predictions')
        label_name = define_results_name(name, args, type='labels')

        predictions_dir = f'./data/v1/routenet/predictions/predictor'
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
            print(f'Created predictions directory: {predictions_dir}')
        prediction_path = os.path.join(predictions_dir, f'{prediction_name}.npy')
        label_dir = f'./data/v1/routenet/labels'
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            print(f'Created labels directory: {label_dir}')
        label_path = os.path.join(label_dir, f'{label_name}.npy')
        print(f'Predictions will be saved to: {prediction_path}')
        print(f'Labels will be saved to: {label_path}')

    train_dataset_list = []
    test_dataset_list = []
    for dsn in args.dataset:
        print(f'Processing dataset: {dsn}')
        train_path = f'./data/v1/routenet/padded_datasets/{dsn}_{dataset_prefix}_train_dataset.pkl'
        test_path = f'./data/v1/routenet/padded_datasets/{dsn}_{dataset_prefix}_test_dataset.pkl'
        stats_path = f'./data/v1/routenet/datasets/{normalize_prefix}_dataset_stats.pkl'
        
        # Load the dataset from the pickle file
        with open(train_path, 'rb') as handle:
            train_dataset = pkl.load(handle)
        print("Train dataset loaded from: ", train_path)
        print(f'Train Dataset size: {len(train_dataset)}')
        with open(test_path, 'rb') as handle:
            test_dataset = pkl.load(handle)
        print("Test dataset loaded from: ", test_path)
        print(f'Test Dataset size: {len(test_dataset)}')

        # Standarize the dataset
        with open(stats_path, 'rb') as handle:
            stats = pkl.load(handle)
        for feature in ['traffic', 'capacity']:
            mean, std = stats[f'{feature}_mean'], stats[f'{feature}_std']
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

    # Train and validation split
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    print(f'Train Dataset size after split: {len(train_dataset)}')
    print(f'Val Dataset size after split: {len(val_dataset)}')

    print('Creating dataloaders...')
    train_loader = get_dataloader_with_sampling(train_dataset, 'routenet', args.train_steps, shuffle=True, batch_size=args.batch_size)
    val_loader = get_dataloader_with_sampling(val_dataset, 'routenet', args.eval_steps, shuffle=True, batch_size=args.batch_size)
    test_loader = get_dataloader_with_sampling(test_dataset, 'routenet', args.eval_steps, shuffle=False, batch_size=args.batch_size)
    print('Dataloaders created.')

    print('Target variable: ', targets_prefix)

    print('Creating the model...')
    if args.train:
        # Define the model
        model = RouteNetPredictor(output_dim=len(args.targets), link_state_dim=args.hidden_dim, path_state_dim=args.hidden_dim, probabilistic=args.prob_decoder, likelihood=args.likelihood, device=device, aggr_dropout=args.agg_dropout, readout_dropout=args.ro_dropout).to(device)
        # Define the loss criterion
        criterion = nn.MSELoss(reduction=args.reduction)
        # Define optimizer and scheduler
        print('Training all layers...')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
        elif args.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience_lr)

        # Print number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}, Trainable parameters: {trainable_params}')
        
        print("Training the model...")
        train_losses, val_losses, lr_list = train_predictor(model, args.targets, train_loader, val_loader, model_path, criterion, optimizer, scheduler, args.epochs, patience=args.patience)
        print("Plotting loss curves...")
        plot_loss_predictor(train_losses, val_losses, lr_list, figure_name, loss_name='MSE')

    if args.eval:
        # Evaluate the model
        print(f"Predictions will be saved to: {prediction_path}")
        # Single model evaluation - load the trained model
        model = RouteNetPredictor(output_dim=len(args.targets), link_state_dim=args.hidden_dim, path_state_dim=args.hidden_dim, probabilistic=args.prob_decoder, likelihood=args.likelihood, device=device, aggr_dropout=args.agg_dropout, readout_dropout=args.ro_dropout).to(device)
        # Define the loss criterion
        criterion = nn.MSELoss(reduction=args.reduction)
        # Load the model
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from: ", model_path)

        print("Evaluating the model...")
        combined_predictions, labels = evaluate_predictor(model, args.targets, test_loader, criterion)
        predictions = torch.cat(combined_predictions).squeeze().cpu().numpy()
        labels = torch.cat(labels).squeeze().cpu().numpy()
        
        # Inverse transformations
        if args.log_transform:
            # Inverse log transform
            predictions = np.expm1(predictions)
            labels = np.expm1(labels)
        elif args.standardize:
            # Inverse standardization
            for i, feature in enumerate(args.targets):
                mean, std = stats[f'{feature}_mean'], stats[f'{feature}_std']
                predictions = predictions * std + mean
                labels = labels * std + mean
        else:
            pass
    
        # Sanity check
        print("Predictions shape: ", predictions.shape)
        print("Labels shape: ", labels.shape)

        # Save predictions and labels
        print("Saving predictions")
        np.save(prediction_path, predictions)
        np.save(label_path, labels)
        print("Predictions saved to: ", prediction_path)
        print("Labels saved to: ", label_path)