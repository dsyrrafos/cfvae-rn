import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_loss_components_rnvae(
    is_variational,
    name,
    epochs,
    best_epoch,
    beta_epoch,
    lr_epoch,
    train_total_losses,
    val_total_losses,
    train_prediction_losses,
    val_prediction_losses,
    train_loss_prior_pred,
    val_loss_prior_pred,
    train_kl_divergences_path,
    val_kl_divergences_path,
    show=False
):
    """
    Plots the loss function components over epochs for both training and validation.
    Adds a vertical line at the best epoch if provided.
    """

    def plot_loss(ax, train_loss, val_loss, logscale, title, ylabel, marker):
        best_val_loss = min(val_loss) if val_loss else None
        ax.plot(epoch_range, train_loss, label="Train", marker=marker)
        ax.plot(epoch_range, val_loss, label="Validation", marker=marker, linestyle="--")
        ax.set_title(f'{title} (Best Val Loss: {best_val_loss:.6f})' if best_val_loss is not None else title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        if logscale:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        if best_epoch is not None:
            ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1, label='Best Epoch')
            ax.legend()

    def plot_beta(ax, beta_values, title, marker):
        ax.plot(epoch_range, beta_values, label="Beta", color='purple', marker=marker)
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Beta Value")
        ax.legend()
        ax.grid(True)

    def plot_lr(ax, lr_values, title, marker):
        ax.plot(epoch_range, lr_values, label="Learning Rate", color='green', marker=marker)
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Learning Rate")
        ax.legend()
        ax.grid(True)

    if is_variational:
        epoch_range = range(1, epochs + 1)
        fig, axs = plt.subplots(2, 3, figsize=(14, 10))
        
        # Plot each component
        plot_loss(axs[0, 0], train_prediction_losses, val_prediction_losses, True, "Reconstruction Loss", "Loss Value", "o")
        plot_loss(axs[0, 1], train_loss_prior_pred, val_loss_prior_pred, True, "Prediction Loss", "Loss Value", "s")
        plot_loss(axs[0, 2], train_kl_divergences_path, val_kl_divergences_path, False, "Path KLD Loss", "Loss Value", "s")
        plot_loss(axs[1, 0], train_total_losses, val_total_losses, True, "Total Loss", "Loss Value", "x")
        plot_beta(axs[1, 1], beta_epoch, "Beta Annealing", "d")
        plot_lr(axs[1, 2], lr_epoch, "Learning Rate Schedule", "^")

    else:
        epoch_range = range(1, epochs + 1)
        fig, axs = plt.subplots(1, 3, figsize=(12, 10))

        # Plot each component
        plot_loss(axs[0, 0], train_prediction_losses, val_prediction_losses, "Reconstruction Loss", "Loss Value", "o")
        plot_loss(axs[0, 1], beta_epoch, "Beta Value", "s")
        plot_loss(axs[0, 2], train_total_losses, val_total_losses, "Total Loss", "Loss Value", "x")

    plt.tight_layout()
    plt.savefig(name)
    if show:
        plt.show()

def plot_loss_predictor(train_losses, val_losses, lr, figure_name, loss_name='MSE', show_plot=False):
    """
    Plots training and validation loss curves along with learning rate schedule.
    Highlights the epoch with the best validation loss.
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        lr (list): List of learning rates per epoch.
        figure_name (str): Path to save the figure.
        loss_name (str): Name of the loss function (for labeling).
        show_plot (bool): Whether to display the plot.
    """
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1
    plt.figure(figsize=(12, 5))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label='Best Epoch')
    plt.title(f'Loss Curves (Best Val Loss: {best_val_loss:.6f})')
    # Change y axis to logarithmic scale
    if loss_name == 'MSE':
        plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_name} Loss')
    plt.legend()
    plt.grid(True)

    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(lr, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(figure_name)
    if show_plot:
        plt.show()
    plt.close()
    print(f'Loss and Learning Rate curves saved to: {figure_name}')

def load_preds_labels(prediction_path, labels_path):
    """
    Loads predictions and labels from .npy files.
    Args:
        prediction_path (str): Path to the .npy file containing predictions.
        labels_path (str): Path to the .npy file containing labels.
    Returns:
        predictions (np.ndarray): Loaded predictions.
        labels (np.ndarray): Loaded labels.
    """
    with open(prediction_path, 'rb') as f:
        predictions = np.load(f)
    with open(labels_path, 'rb') as f:
        labels = np.load(f)
    return predictions, labels

def print_metrics(predictions, labels):
    """
    Computes and prints regression metrics between predictions and labels.
    Args:
        predictions (np.ndarray): Model predictions.
        labels (np.ndarray): Ground truth labels.
    """
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    rmspe = np.sqrt(np.mean(((predictions - labels) / (labels + 1e-8)) ** 2)) * 100  # Avoid division by zero
    mae = np.mean(np.abs(predictions - labels))
    mape = np.mean(np.abs((predictions - labels) / (labels + 1e-8))) * 100  # Avoid division by zero
    r_squared = 1 - (np.sum((labels - predictions) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
    print(f'RMSE: {rmse:.4f}')
    print(f'RMSPE: {rmspe:.4f}%')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.4f}%')
    print(f'R²: {r_squared:.4f}')

    p_low = 95
    p_high = 99.9
    # Filter only samples outside the 95-the percentile and inside fo the 99.9th percentile of the label distribution
    label_percentile_low = np.percentile(labels, p_low)
    label_percentile_high = np.percentile(labels, p_high)
    mask = (labels >= label_percentile_low) & (labels <= label_percentile_high)
    predictions = predictions[mask]
    labels = labels[mask]
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    rmspe = np.sqrt(np.mean(((predictions - labels) / (labels + 1e-8)) ** 2)) * 100  # Avoid division by zero
    mae = np.mean(np.abs(predictions - labels))
    mape = np.mean(np.abs((predictions - labels) / (labels + 1e-8))) * 100  # Avoid division by zero
    r_squared = 1 - (np.sum((labels - predictions) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
    print(f'--- Metrics for labels >= {p_low}th percentile ({label_percentile_low:.4f}) and <= {p_high}th percentile ({label_percentile_high:.4f}) ---')
    print(f'RMSE: {rmse:.4f}')
    print(f'RMSPE: {rmspe:.4f}%')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.4f}%')
    print(f'R²: {r_squared:.4f}')

def plot_vae_vs_deterministic(predictions_seen, labels_seen, predictions_unseen, labels_unseen, deterministic_preds_seen, deterministic_preds_unseen, metric_name='MAE', save_path=None):
    """
    Plots the comparison between cFVAE predictions and deterministic model predictions
    for both seen and unseen datasets using the specified metric (MAE or MAPE).
    Args:
        predictions_seen (np.ndarray): cFVAE predictions for seen data (shape: [num_samples, num_draws]).
        labels_seen (np.ndarray): Ground truth labels for seen data (shape: [num_samples,]).
        predictions_unseen (np.ndarray): cFVAE predictions for unseen data (shape: [num_samples, num_draws]).
        labels_unseen (np.ndarray): Ground truth labels for unseen data (shape: [num_samples,]).
        deterministic_preds_seen (np.ndarray): Deterministic model predictions for seen data (shape: [num_samples,]).
        deterministic_preds_unseen (np.ndarray): Deterministic model predictions for unseen data (shape: [num_samples,]).
        metric_name (str): Metric to use for comparison ('MAE' or 'MAPE').
        save_path (str or None): If provided, saves the plot to this path.
    """
    try:
        predictions_seen = predictions_seen.cpu().numpy()
        labels_seen = labels_seen.cpu().numpy()
        predictions_unseen = predictions_unseen.cpu().numpy()
        labels_unseen = labels_unseen.cpu().numpy()
        deterministic_preds_seen = deterministic_preds_seen.cpu().numpy()
        deterministic_preds_unseen = deterministic_preds_unseen.cpu().numpy()
    except:
        pass
    num_samples = predictions_seen.shape[1]
    maes_seen = []
    maes_unseen = []
    for i in range(1, num_samples + 1):
        if metric_name == 'MAE':
            preds_mean = np.mean(predictions_seen[:, :i], axis=1)
            mae = np.mean(np.abs(preds_mean - labels_seen))
            maes_seen.append(mae)
            preds_mean = np.mean(predictions_unseen[:, :i], axis=1)
            mae = np.mean(np.abs(preds_mean - labels_unseen))
            maes_unseen.append(mae)
        elif metric_name == 'MAPE':
            preds_mean = np.mean(predictions_seen[:, :i], axis=1)
            mape = np.mean(np.abs((labels_seen - preds_mean) / labels_seen)) * 100
            maes_seen.append(mape)
            preds_mean = np.mean(predictions_unseen[:, :i], axis=1)
            mape = np.mean(np.abs((labels_unseen - preds_mean) / labels_unseen)) * 100
            maes_unseen.append(mape)
    if metric_name == 'MAE':
        deterministic_mae_seen = np.mean(np.abs(deterministic_preds_seen - labels_seen))
        deterministic_mae_unseen = np.mean(np.abs(deterministic_preds_unseen - labels_unseen))
    elif metric_name == 'MAPE':
        deterministic_mae_seen = np.mean(np.abs((labels_seen - deterministic_preds_seen) / labels_seen)) * 100
        deterministic_mae_unseen = np.mean(np.abs((labels_unseen - deterministic_preds_unseen) / labels_unseen)) * 100
    plt.figure(figsize=(4, 3.5))
    plt.plot(range(1, num_samples + 1), maes_seen, label='cFVAE (Seen)', marker='o', markersize=3, color='tab:orange')
    plt.axhline(y=deterministic_mae_seen, color='tab:orange', linestyle='--', label='Deterministic (Seen)')
    plt.plot(range(1, num_samples + 1), maes_unseen, label='cFVAE (Unseen)', marker='s', markersize=3, color='tab:blue')
    plt.axhline(y=deterministic_mae_unseen, color='tab:blue', linestyle='--', label='Deterministic (Unseen)')
    plt.xlabel('Number of Samples Drawn')
    plt.ylabel(f'{metric_name}')
    plt.title(f'cFVAE vs Deterministic {metric_name}')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{save_path}.eps', format='eps', bbox_inches='tight')
    plt.show()

def plot_predictions_vs_labels(pred, label, title="", ax=None):
    """
    Scatter plot of predictions vs labels with a diagonal line indicating perfect predictions.
    Args:
        pred: array of shape (N,)
        label: array of shape (N,)
        title: title of the plot
        ax: matplotlib axis object (optional)
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Scatter plot
    ax.scatter(label, pred, alpha=0.6)
    
    # Diagonal line (perfect predictions)
    min_val = min(np.min(label), np.min(pred))
    max_val = max(np.max(label), np.max(pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="y = x")
    
    ax.set_title(title)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Predictions")
    ax.legend()

def interval_coverage_error(y_true, y_lower, y_upper, nominal=0.9):
    """
    Computes interval coverage error for a nominal alpha-level interval.
    
    Args:
        y_true:   array of shape (N,)
        y_lower:  array of shape (N,)
        y_upper:  array of shape (N,)
        nominal:  the nominal coverage, e.g. 0.9 for 90% interval
        
    Returns:
        empirical_coverage, interval_coverage_error
    """
    y_true = np.asarray(y_true)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)
    
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    empirical = inside.mean()
    ice = abs(empirical - nominal)
    return empirical, ice

def plot_interval_coverage_error(ec_list, ice, alpha_levels, save_path=None, show_score=False):
    """
    Plots interval coverage error (ICE) bar plot.
    Args:
        ec_list:       list or array of empirical coverages at different alpha levels
        ice:           interval coverage error (scalar)
        alpha_levels:  list or array of nominal alpha levels
        save_path:     if provided, saves plot as .eps and .pdf
        show_score:    if True, displays ICE score in the title
    """
    ec_list = np.array(ec_list)
    alpha_levels = np.array(alpha_levels)

    plt.figure(figsize=(5, 5))

    width = 1/(len(alpha_levels)+1)  # smaller width prevents bars touching each other

    # --- Main bars (empirical coverage) ---
    plt.bar(
        alpha_levels,
        ec_list,
        width=width,
        color="skyblue",
        edgecolor="black",
        linewidth=1.0,
        label="Empirical Coverage",
        zorder=2,
    )

    # --- Calibration gap ---
    calibration_gap = alpha_levels - ec_list

    plt.bar(
        alpha_levels,
        calibration_gap,
        bottom=ec_list,
        width=width,
        color="none",
        edgecolor="red",
        hatch="///",
        linewidth=1.0,
        alpha=0.5,
        label="Calibration gap",
        zorder=3,
    )

    # --- Ideal diagonal ---
    plt.plot(
        [0, 1], [0, 1],
        'k--',
        linewidth=2,
        label='Ideal Calibration',
        zorder=4
    )

    # --- Formatting ---
    plt.xlabel("Nominal Coverage Level", fontsize=12)
    plt.ylabel("Empirical Coverage", fontsize=12)
    if show_score:
        plt.title("Interval Coverage\nCalibration Error = {:.4f}".format(ice), fontsize=15)
    else:
        plt.title("Interval Coverage", fontsize=15)

    # important: align ticks EXACTLY with bar centers
    plt.xticks(alpha_levels, [f"{a:.02f}" for a in alpha_levels], rotation=45)
    plt.yticks(alpha_levels, [f"{a:.02f}" for a in alpha_levels])

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}.eps', format='eps')
        plt.savefig(f'{save_path}.pdf', format='pdf')
    plt.show()

def regression_ece(y_true, mu_pred, sigma_pred, n_bins=20, eps=1e-8):
    """
    Expected Calibration Error (ECE) for probabilistic regression.

    Args:
        y_true : array-like, shape (N,)
            True target values.
        mu_pred : array-like, shape (N,)
            Predicted means.
        sigma_pred : array-like, shape (N,)
            Predicted standard deviations.
        n_bins : int
            Number of bins (quantile levels).
        eps : float
            Small value to avoid zero or negative variance.
    Returns:
        ece : float
            Expected Calibration Error.
        pits : array-like, shape (N,)
            Probability Integral Transform values.
    """

    y_true = np.asarray(y_true)
    mu_pred = np.asarray(mu_pred)
    sigma_pred = np.asarray(sigma_pred)

    # Avoid zero or negative variance
    sigma_pred = np.maximum(sigma_pred, eps)

    # Probability Integral Transform
    pits = norm.cdf(y_true, loc=mu_pred, scale=sigma_pred)

    # Bin PIT values
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts, _ = np.histogram(pits, bins=bin_edges)

    N = len(pits)
    expected_frac = 1.0 / n_bins

    ece = 0.0
    for count in bin_counts:
        empirical_frac = count / N
        ece += empirical_frac * abs(empirical_frac - expected_frac)

    return ece, pits

def plot_pit_cdf_histogram(pits, ece, n_bins=20, save_path=None, show_score=False):
    """
    ICE-style histogram for PIT calibration using empirical CDF.

    Args:
        pits : array-like, shape (N,)
            Probability Integral Transform values.
        ece : float
            Expected Calibration Error.
        n_bins : int
            Number of bins (quantile levels).
        save_path : str or None
            If provided, saves plot as .eps and .pdf.
        show_score : bool
            If True, displays ECE score in the title.
    """

    pits = np.asarray(pits)

    # Nominal PIT levels (bin edges, excluding 0)
    # Nominal PIT levels EXCLUDING u = 1
    u_levels = np.linspace(0, 1, n_bins + 1)[1:-1]

    # Empirical CDF at each level
    empirical_cdf = np.array([
        np.mean(pits <= u) for u in u_levels
    ])

    # Calibration gap
    calibration_gap = empirical_cdf - u_levels

    width = 1 / (n_bins + 1)

    plt.figure(figsize=(5, 5))

    # Empirical CDF bars
    plt.bar(
        u_levels,
        empirical_cdf,
        width=width,
        color="skyblue",
        edgecolor="black",
        linewidth=1.0,
        label="Empirical PIT CDF",
        zorder=2
    )

    # Calibration gap (stacked, like ICE)
    plt.bar(
        u_levels,
        calibration_gap,
        bottom=u_levels,
        width=width,
        color="none",
        edgecolor="red",
        hatch="///",
        linewidth=1.0,
        alpha=0.6,
        label="Calibration gap",
        zorder=3
    )

    # Ideal diagonal
    plt.plot(
        [0, 1], [0, 1],
        "k--",
        linewidth=2,
        label="Ideal calibration",
        zorder=4
    )

    # Formatting
    plt.xlabel("Nominal PIT level", fontsize=12)
    plt.ylabel("Empirical PIT CDF", fontsize=12)
    if show_score:
        plt.title(f"Probability Integral Transform\nCalibration Error = {ece:.4f}", fontsize=15)
    else:
        plt.title("Probability Integral Transform", fontsize=15)
        
    plt.xticks(u_levels, [f"{u:.2f}" for u in u_levels], rotation=45)
    plt.yticks(u_levels, [f"{u:.2f}" for u in u_levels])

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}.eps", format="eps")
        plt.savefig(f"{save_path}.pdf", format="pdf")
    plt.show()
