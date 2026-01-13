import torch
import math
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm


def set_seed(seed):
    """
    Set random seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def define_name_rnvae(args):
    """
    Define a unique name for the model based on its configuration.
    Args:
        args: Argument parser object containing model configuration.
    Returns:
        str: The defined model name.
    """
    if len(args.prefix) > 0:
        prefix = args.prefix + '_'
    else:
        prefix = ''

    if args.conditional:
        prefix += 'C'

    if args.gmm:
        prefix += 'GMM'
        prefix += str(args.n_components)

    if args.norm_flow:
        if args.prior_flow:
            prefix += 'PR'
        elif args.vamp_prior:
            prefix += 'VP'
        elif args.full_flow:
            prefix += 'FF'
        if args.decoder_flow:
            prefix += 'DE'
        if args.flow_type == 'planar':
            prefix += 'PF'
        elif args.flow_type == 'affine_coupling':
            prefix += 'ACF'
        prefix += str(args.n_flows)

    if args.variational:
        prefix += 'VAE'
    else:
        prefix += 'AE'

    if args.pretrained:
        prefix += 'PT'
    if args.prior:
        prefix += 'PN'
    if args.log_transform:
        prefix += 'LOG'

    if args.reduction == 'mean':
        prefix += 'M'
    elif args.reduction == 'sum':
        prefix += 'S'
    prefix += '_'

    if 'delay' in args.targets:
        prefix += 'D'
    if 'jitter' in args.targets:
        prefix += 'J'
    if 'loss' in args.targets:
        prefix += 'L'

    return f'{prefix}_EP{args.epochs}_H{args.hidden_dim}_L{args.latent_dim}_LR{args.lr}_LRS{args.lr_scheduler}_B{args.beta}_BS{args.beta_schedule}'

def define_name_predictor(args):
    if len(args.prefix) > 0:
        prefix = args.prefix + '_'
    else:
        prefix = ''

    prefix += 'PRED'

    if args.deep_ensemble:
        prefix += f'DE{args.num_models}_'

    if args.mcdropout:
        prefix += 'MCD_AGD'
        prefix += str(args.agg_dropout).replace('.', '')
        prefix += '_ROD'
        prefix += str(args.ro_dropout).replace('.', '')
        prefix += '_'

    if args.prob_decoder:
        prefix += 'PD'
        if args.likelihood == 'gaussian':
            prefix += 'G_'
        elif args.likelihood == 'studentt':
            prefix += 'ST_'
        elif args.likelihood == 'mdn':
            prefix += 'MDN_'

    if args.pretrained:
        prefix += 'PT'

    if args.log_transform:
        prefix += 'LOG'

    if args.reduction == 'mean':
        prefix += 'M'
    elif args.reduction == 'sum':
        prefix += 'S'
    prefix += '_'

    if 'delay' in args.targets:
        prefix += 'D'
    if 'jitter' in args.targets:
        prefix += 'J'
    if 'loss' in args.targets:
        prefix += 'L'
    prefix += '_'

    prefix += f'EP{args.epochs}_H{args.hidden_dim}_LR{args.lr}_LRS{args.lr_scheduler}'

    return prefix

def define_results_name(model_name, args, type='predictions'):
    prefix = '_'.join(args.dataset)

    if args.perturbation_feature is not None:
        prefix += f"_perturbed_{args.perturbation_feature}_nl{args.noise_level}"

    if type == 'labels':
        return prefix
    else:
        try:
            if args.mcdropout:
                prefix += f'_{args.n_samples}S'
        except AttributeError:
            pass
        try:
            if args.variational:
                prefix += f'_{args.n_samples}S'
        except AttributeError:
            pass
        return f'{prefix}_{model_name}'

def train_rnvae_epoch(autoencoder, targets, train_loader, optimizer, is_variational, beta_epoch=1.0, reduction='mean'):
    autoencoder.train()
    train_count = 0
    train_loss_all = 0
    train_loss_all_pred = 0
    train_loss_prior_pred = 0

    if is_variational:
        train_loss_all_kld_path = 0

    for data in tqdm(train_loader):
        # if np.random.rand() < skip_prob:
        #     continue
        # data = data.to(device)
        optimizer.zero_grad()
        if is_variational:
            loss, pred_loss, kld_path  = autoencoder.loss_function(data, targets, beta=beta_epoch, reduction=reduction)
            train_loss_prior_pred += autoencoder.pred_loss_function(data, targets, reduction=reduction).item()
            train_loss_all += loss.item()
            train_loss_all_pred += pred_loss.item()
            train_loss_all_kld_path += kld_path.item()
        else:
            loss, pred_loss = autoencoder.loss_function(data, targets)
            train_loss_all += loss.item()
            train_loss_all_pred += pred_loss.item()
        
        # Backward pass
        loss.backward()
        train_count += 1
        optimizer.step()

    if is_variational:
        return (train_loss_all / train_count,
                train_loss_all_pred / train_count,
                train_loss_prior_pred / train_count,
                train_loss_all_kld_path / train_count,
                train_count)
    else:
        return (train_loss_all / train_count,
                train_loss_all_pred / train_count,
                train_count)

def eval_rnvae_epoch(autoencoder, targets, eval_loader, is_variational, beta_epoch=1.0, reduction='mean'):
    autoencoder.eval()
    eval_count = 0
    eval_loss_all = 0
    eval_loss_all_pred = 0
    eval_loss_all_prior_pred = 0

    if is_variational:
        eval_loss_all_kld_path = 0

    with torch.no_grad():
        for data in tqdm(eval_loader):
            # if np.random.rand() < skip_prob:
            #     continue
            # data = data.to(device)
            if is_variational:
                loss, pred_loss, kld_path = autoencoder.loss_function(data, targets, beta=beta_epoch, reduction=reduction)
                eval_loss_all_prior_pred += autoencoder.pred_loss_function(data, targets, reduction=reduction).item()
                eval_loss_all += loss.item()
                eval_loss_all_pred += pred_loss.item()
                eval_loss_all_kld_path += kld_path.item()
            else:
                loss, pred_loss = autoencoder.loss_function(data, targets)
                eval_loss_all += loss.item()
                eval_loss_all_pred += pred_loss.item()

            eval_count += 1

    if is_variational:
        return (eval_loss_all / eval_count,
                eval_loss_all_pred / eval_count,
                eval_loss_all_prior_pred / eval_count,
                eval_loss_all_kld_path / eval_count,
                eval_count)
    else:
        return (eval_loss_all / eval_count,
                eval_loss_all_pred / eval_count,
                eval_count)

def linear_beta_schedule_vae(epoch, max_beta, total_epochs, warmup_epochs=15):
    """
    Linear beta schedule for VAE.
    Args:
        epoch (int): Current epoch.
        max_beta (float): Maximum beta value.
        total_epochs (int): Total number of epochs.
        warmup_epochs (int): Number of warmup epochs before beta starts increasing.
    Returns:
        float: The beta value for the current epoch.
    """
    return max(0, max_beta * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)) if epoch >= warmup_epochs else 0

def sigmoid_beta_schedule_vae(epoch, max_beta, midpoint=15, steepness=10):
    """
    Sigmoid beta schedule for VAE.
    Args:
        epoch (int): Current epoch.
        max_beta (float): Maximum beta value.
        midpoint (int): The epoch at which the beta value is half of max_beta.
        steepness (float): Controls the steepness of the sigmoid curve.
    Returns:
        float: The beta value for the current epoch.
    """
    beta = max_beta / (1 + math.exp(-steepness * (epoch - midpoint) / midpoint))
    return beta

def cyclical_beta_schedule(epoch, max_beta=1.0, cycle_length=5):
    """
    Cyclical schedule for KL weight (beta).
    
    Args:
        epoch (int): current epoch (0-based).
        cycle_length (int): number of epochs in one cycle.
        beta_max (float): maximum beta value.
        
    Returns:
        float: beta value for the current epoch.
    """
    # position within the cycle [0,1)
    cycle_pos = (epoch % cycle_length) / cycle_length
    # linearly increase beta within the cycle
    beta = max_beta * cycle_pos
    return beta

def vae_beta_schedule(beta, epoch, total_epochs, type='sigmoid'):
    """
    Get the beta value for VAE based on the epoch and total epochs.
    Args:
        beta (float): The maximum beta value.
        epoch (int): The current epoch.
        total_epochs (int): Total number of epochs.
        type (str): Type of beta schedule, either 'linear' or 'sigmoid'.
        kwargs (dict, optional): Additional arguments for specific schedule types.
    Returns:
        float: The beta value for the current epoch.
    """
    if type == 'linear':
        return linear_beta_schedule_vae(epoch, beta, total_epochs)
    elif type == 'sigmoid':
        return sigmoid_beta_schedule_vae(epoch, beta, midpoint=int(total_epochs*0.9))
    elif type == 'cyclical':
        return cyclical_beta_schedule(epoch, beta)
    elif type == 'constant':
        return beta
    else:
        raise ValueError("Unknown beta schedule type: {}".format(type))
    
def train_predictor(model, targets, train_dataloader, val_dataloader, model_path, criterion, optimizer, scheduler, num_epochs, patience=10):
    best_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    lr_list = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples_train = 0
        print(f'Epoch {epoch+1}/{num_epochs}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        for inputs, labels in tqdm(train_dataloader):
            num_samples_train += 1
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            ground_truth = torch.stack([labels[t] for t in targets], dim=2).squeeze().float().to(model.device)
            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / num_samples_train
        train_losses.append(epoch_loss)
        lr_list.append(scheduler.get_last_lr()[0])

        model.eval()
        val_loss = 0.0
        num_samples_val = 0
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_dataloader):
                num_samples_val += 1
                val_outputs, _ = model(val_inputs)
                val_ground_truth = torch.stack([val_labels[t] for t in targets], dim=2).squeeze().float().to(model.device)
                val_loss += criterion(val_outputs, val_ground_truth).item()
        val_loss /= num_samples_val
        val_losses.append(val_loss)

        print(f'Num Samples Train: {num_samples_train}, Train Loss: {epoch_loss:.6f}, Num Samples Val: {num_samples_val}, Val Loss: {val_loss:.6f}')

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            print(f"Best model updated with val loss: {best_loss:.6f}. Saving model...")
            torch.save(best_model_state, model_path)
            print("Model saved to: ", model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs...")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, lr_list

def evaluate_predictor(model, targets, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        num_samples = 0
        for inputs, labels in tqdm(dataloader):
            num_samples += 1
            outputs, _  = model(inputs)
            ground_truth = torch.stack([labels[t] for t in targets], dim=2).squeeze().float().to(model.device)
            loss = criterion(outputs, ground_truth)
            total_loss += loss.item()
            predictions.append(outputs)
            ground_truths.append(ground_truth)

    print(f'Num samples: {num_samples}, Validation Loss: {total_loss / len(dataloader)}')
    return predictions, ground_truths