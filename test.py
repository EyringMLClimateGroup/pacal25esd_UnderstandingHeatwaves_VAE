# %%
import os
import re
import math
import torch
import pickle
import numpy as np
import pandas as pd
import logging
from plotly.subplots import make_subplots

import joblib
from joblib import Parallel, delayed

import plotly.graph_objects as go
import plotly.express as px
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from openTSNE import TSNE as OpenTSNE

from tqdm import tqdm
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import r2_score

from models import vae_models
from utils.misc import is_datetime_string
from utils.data_loaders import load_heatwave_samples

import dash
from dash import dcc, html, Input, Output, ctx, State, dash_table

from plotly.colors import sample_colorscale
import random

# %%
SEED = 42

# Set seeds and single-threaded environment
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
random.seed(SEED)
np.random.seed(SEED)


# Define colormaps for different variables
var_cmaps = {
        't2m': 'bwr',
        'stream250': 'seismic',
        'msl': 'Spectral_r',
        'u10': 'PuOr',
        'v10': 'PuOr',
        'tcc': 'BrBG',
        'ssr': 'RdYlGn_r',
        'z500': 'Spectral_r',
        'r': 'RdYlBu_r',
    }   

# %%
logger = logging.getLogger("default_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# %%
def set_logger(_logger):
    global logger
    logger = _logger


# %%
def get_grid_dims(n_items, max_cols=4):
    """
    Given the number of items to plot, return (nrows, ncols) 
    with up to `max_cols` per row.
    """
    ncols = min(n_items, max_cols)
    nrows = math.ceil(n_items / ncols)
    return nrows, ncols


# %%
def convert_name(name: str) -> str:
    """
    Convert a short name to a long name or vice versa.
    """
    name_map = {
        'train': 'training',
        'val': 'validation',
        'test': 'testing',
        'combined': 'combined'
    }
    
    # Normalize input to lowercase
    name = name.lower()
    
    # Check if input is a short name
    if name in name_map:
        return name_map[name]
    
    # Check if input is a long name
    for short, long in name_map.items():
        if name == long:
            return short
    
    raise ValueError(f"Unknown name: '{name}'")


# %%
def reconstruct_data(model, data_loader, DEVICE, cfg):
    """
    Reconstruct the data using the model.
    
    Parameters:
        model (torch.nn.Module): 
            The trained model.
        data_loader (torch.utils.data.DataLoader): 
            DataLoader for the dataset.
        DEVICE (torch.device): 
            Device to run the model on.
        cfg (dict): 
            Configuration dictionary.
    """
    
    recon_x, z_vectors, date_labels = [], [], []
    
    num_used_samples = len(data_loader.dataset)    
    with torch.no_grad():
        # Loop through the data loader
        loop = tqdm(data_loader, desc="Reconstructing data", total=len(data_loader), leave=False)
        for i, (batch, dates) in enumerate(loop):
            batch = batch.float().to(DEVICE)
            recon_batch, mu, logvar, z = model(batch)
            recon_x.append(recon_batch.cpu().detach().numpy())
            z_vectors.append(z.cpu().detach().numpy())
            date_labels.append(dates)
    recon_x = np.concatenate(recon_x, axis=0)
    z_vectors = np.concatenate(z_vectors, axis=0)
    date_labels = np.concatenate(date_labels, axis=0)
    
    # Drop the extra samples from original dataset
    x = data_loader.dataset.samples
    x = x[:-(num_used_samples%cfg['batch_size'])]
    
    date_labels = date_labels[:num_used_samples]
    recon_x = recon_x[:num_used_samples]
    z = z_vectors[:num_used_samples]
    
    return x, recon_x, z_vectors, date_labels


# %%
def create_recon_dict(data_loaders, data_types, model, DEVICE, cfg, load_from_pkl_path=None):
    
    if load_from_pkl_path:
        logger.info("Loading recon_df from {}".format(load_from_pkl_path))
        # Load the recon_df from a pickle file
        with open(load_from_pkl_path, 'rb') as f:
            recon_df = pickle.load(f)
        logger.info("✅ Loaded recon_df from {}".format(load_from_pkl_path))

        cfg['output_dir'] = os.path.dirname(load_from_pkl_path)
        logger.info("Changed output_dir to {}".format(cfg['output_dir']))
        return recon_df
    
    reconstructed_datasets = {}
    for data_loader, data_type in zip(data_loaders, data_types):
        logger.info(f"Reconstructing {data_type} data...")
        x, recon_x, z, date_labels = reconstruct_data(model, data_loader, DEVICE, cfg)
        reconstructed_datasets[data_type] = {
            'x': x,
            'recon_x': recon_x,
            'z': z,
            'date_labels': date_labels
        }
        logger.info(f"Reconstructed {data_type} data shape: {recon_x.shape}")
        logger.info(f"Original {data_type} data shape: {x.shape}")
            
    return reconstructed_datasets


# %%
def create_dataloaders(cfg):
    """
    Create dataloaders.
    """
    train_loader = load_heatwave_samples.create_heatwave_dataloader(
        data_folder=cfg['data']['root_dir'],
        cluster_info_path=cfg['data']['cluster_info_csv'],
        variables=cfg['data']['variables'],
        model_name=cfg['data']['model_name'],
        cfg=cfg,
        years=(tuple(cfg['data']['train_years'])),
        batch_size=cfg['batch_size'],
        time_window=5,
        image_size=(64,192),
        shuffle=False,
        )
    
    val_loader = load_heatwave_samples.create_heatwave_dataloader(
        data_folder=cfg['data']['root_dir'],
        cluster_info_path=cfg['data']['cluster_info_csv'],
        variables=cfg['data']['variables'],
        model_name=cfg['data']['model_name'],
        cfg=cfg,
        years=(tuple(cfg['data']['val_years'])),
        batch_size=cfg['batch_size'],
        time_window=5,
        image_size=(64,192),
        shuffle=False,
    )
    
    test_loader = load_heatwave_samples.create_heatwave_dataloader(
        data_folder=cfg['data']['root_dir'],
        cluster_info_path=cfg['data']['cluster_info_csv'],
        variables=cfg['data']['variables'],
        model_name=cfg['data']['model_name'],
        cfg=cfg,
        years=(tuple(cfg['data']['test_years'])),
        batch_size=cfg['batch_size'],
        time_window=5,
        image_size=(64,192),
        shuffle=False,
    )
    
    # print train_loader length
    logger.info("train Loader Length: {}".format(len(train_loader)))
    for i, (batch, date_labels) in enumerate(train_loader):
        logger.info(f"{i+1}.Batch Shape: {batch.shape}")
        logger.info(f"{i+1}.Date Label Shape: {len(date_labels)}")
        break
    
    return train_loader, val_loader, test_loader


# %%
def load_model_file(model, model_path, DEVICE):
    """
    Load the model from the specified path.
    
    Args:
        model: The model to load.
        model_path: Path to the saved model.
    
    Returns:
        model: The loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not model_path.endswith('.pth'):
        raise ValueError(f"Model file should have .pth extension: {model_path}")

    state_dict = torch.load(model_path, map_location=DEVICE)

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


# %%
def load_trained_model(cfg, DEVICE):
    """
    Load the trained model from the specified path.
    
    Parameters:
        cfg (dict): Configuration dictionary.
        DEVICE (torch.device): Device to load the model on.
    Returns:
        model (torch.nn.Module): Loaded model.
    """

    ModelClass = vae_models.get(cfg['model']['name'])
    if ModelClass is None:
        logger.error("Model {} is not defined in vae_models. Available models: {}".format(cfg['model']['name'], list(vae_models.keys())))
        raise ValueError("Model {} is not defined in vae_models. Available models: {}".format(cfg['model']['name'], list(vae_models.keys())))
 
    model = ModelClass(
        input_dim=len(cfg['data']['variables']), 
        hidden_dim=cfg['model']['hidden_dim'], 
        latent_dim=cfg['model']['latent_dim'],
        input_shape=(cfg['data']['temporal_resolution'], 64, 192),
        kernel_size=(cfg['model']['kernel_size'],) * 3,
        stride=(cfg['model']['stride'],) * 3,
        padding=(cfg['model']['padding'],) * 3,
        apply_sigmoid=cfg['model']['apply_sigmoid']
        ).to(DEVICE)

    # find the model in the saved_models directory
    if cfg['test']['load_model'] == 'last':
        logger.info("Loading the last saved model...")
        model_dir = os.path.join(cfg['train']['checkpoint_dir'], max(os.listdir(cfg['train']['checkpoint_dir'])))
        # check if the model is following vae_lstm.pth naming convention otherwise choose the file from previous directory
        saved_models_listdir = sorted(os.listdir(cfg['train']['checkpoint_dir']))
        
        while not os.path.exists(os.path.join(model_dir, cfg['model']['name'] + '.pth')):
            model_dir = os.path.join(cfg['train']['checkpoint_dir'], saved_models_listdir[-1])
            saved_models_listdir.pop()
            
        # create a new directory in output_dir with the same name as the model directory
        cfg['output_dir'] = os.path.join(cfg['data']['output_dir'], os.path.basename(model_dir))
        os.makedirs(cfg['output_dir'], exist_ok=True)
        logger.info("Model directory: {}".format(model_dir))
        logger.info("Output directory: {}".format(cfg['output_dir']))
        
    elif is_datetime_string(cfg['test']['load_model']):
        logger.info("Loading the model from the specified date...")
        model_dir = os.path.join(cfg['train']['checkpoint_dir'], cfg['test']['load_model'])
        cfg['output_dir'] = os.path.join(cfg['data']['output_dir'], cfg['test']['load_model'])
        os.makedirs(cfg['output_dir'], exist_ok=True)
        logger.info("Model directory: {}".format(model_dir))
        logger.info("Output directory: {}".format(cfg['output_dir']))

    else:
        logger.error("Invalid model directory in config file. Expected 'last' or 'YYMMDD-HHMMSS' format.")
        # print the line number of the error in cfg
        raise ValueError("Invalid model directory in config file. Expected 'last' or 'YYMMDD-HHMMSS' format.")
        
    model = load_model_file(model=model, model_path=os.path.join(model_dir, cfg['model']['name'] + '.pth'), DEVICE=DEVICE)
    
    if torch.cuda.device_count() > 1:   
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE) 
    
    model.eval()
    return model


# %%
def load_hw_dataset(cfg):
    """
    Load the heatwave dataset from the specified CSV file.
    
    Parameters:
        cfg (dict): Configuration dictionary.
        
    Returns:
        pd.DataFrame: DataFrame containing the heatwave dataset.
    """
    
    heatwave_dataset = pd.read_csv(cfg['data']['cluster_info_csv'])
    heatwave_dataset['start_date'] = heatwave_dataset['start_date'].astype(str)
    heatwave_dataset['year'] = heatwave_dataset['start_date'].apply(lambda x: int(x.split('-')[0]))
    heatwave_dataset['month'] = heatwave_dataset['start_date'].apply(lambda x: int(x.split('-')[1]))
    # select unique start_date
    heatwave_unique = heatwave_dataset.drop_duplicates(subset=['start_date'])

    return heatwave_unique


# %%
def plot_hw_per_year(heatwave_df, cfg, file_suffix=None):
    """
    Plot the number of heatwaves per year as a line plot.
    
    Parameters:
        heatwave_df (pd.DataFrame): DataFrame containing heatwave data.
        cfg (dict): Configuration dictionary.
        
    Returns:
        None
    """
    
    yearly_counts = heatwave_df.groupby('year')['start_date'].count()
    years = yearly_counts.index
    counts = yearly_counts.values
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
    ax1.grid(True, linestyle=':', linewidth=0.5, zorder=0)  
    
    # Split year ranges
    split_year = 1990
    mask_train = (years <= split_year) & (years >= 1941)
    mask_rest = years >= split_year

    # Set reference year for centering
    x_ref = 1941

    # --- Train trendline (≤ 1990) ---
    x_train = years[mask_train]
    x_train_centered = x_train - x_ref
    coeffs1 = np.polyfit(x_train_centered, counts[mask_train], 1)
    logger.info(f"Train trendline coefficients: {coeffs1}")
    
    trend1 = np.poly1d(coeffs1)(x_train_centered)

    ax1.plot(
        x_train, trend1, linestyle=':', color='tab:gray',
        label=f'$≤1990: y = {coeffs1[0]:.3f}x {(coeffs1[0]*(-x_ref)) + coeffs1[1]:.2f}$',
        zorder=1
    )

    # --- Post-1990 trendline (> 1990) ---
    x_rest = years[mask_rest]
    x_rest_centered = x_rest - x_ref
    coeffs2 = np.polyfit(x_rest_centered, counts[mask_rest], 1)
    trend2 = np.poly1d(coeffs2)(x_rest_centered)
    ax1.plot(
        x_rest, trend2, linestyle='--', color='tab:gray',
        label=f'$>1990: y = {coeffs2[0]:.3f}x {(coeffs2[0]*(-x_ref)) + coeffs2[1]:.2f}$',
        zorder=1
    )

    
    ax1.plot(years, counts, linestyle='-', color='tab:red', alpha=0.3, zorder=2)
    for z, (x, y) in enumerate(zip(years, counts), start=3): 
        ax1.scatter(x, y, s=150, c='tab:red', zorder=z, edgecolors='k', linewidths=0.5)
        ax1.text(
            x, y-0.1, str(y),
            ha='center', va='center',
            color='white', fontsize=8, fontweight='bold',
            zorder=z  
        )    
    ax1.set_ylabel('Number of Heatwaves')
    ax1.set_xticks(np.arange(years.min(), years.max() + 1, 5))
    ax1.set_xlim(years.min() - 1, years.max() + 1)

    ax1.legend(loc='upper left', fontsize=10, labelspacing=0.3, handletextpad=0.5, markerscale=0.5)
    
    start_year, end_year = heatwave_df.year.min(), heatwave_df.year.max()
    heatwave_per_year_filename = os.path.join(cfg['output_dir'], f"lineplot_heatwaves_per_year_{start_year}_{end_year}")
    if file_suffix:
        heatwave_per_year_filename += f"_{file_suffix}"
        
    plt.savefig(heatwave_per_year_filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(heatwave_per_year_filename + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # plot the number of heatwaves per year
    heatwave_per_year = heatwave_df.groupby('year').count()['start_date']
    heatwave_per_year.to_csv(os.path.join(cfg['output_dir'], 'heatwaves_per_year.csv'))
    logger.info("Heatwaves per year saved to {}".format(os.path.join(cfg['output_dir'], 'heatwaves_per_year.csv')))


# %%
def calculate_tsne_and_gmm(recon_df, cfg, perplexity=[50, 100], tsne_components=2, gmm_components=(4,4), period='test', file_suffix=None):
    """
    Calculate t-SNE and GMM for the reconstructed data.
    """    
    combined_z_vectors = np.concatenate([recon_df[period]['z'] for period in recon_df if period != 'combined'], axis=0)
    combined_date_labels = np.concatenate([recon_df[period]['date_labels'] for period in recon_df if period != 'combined'], axis=0)
    logger.info(f"Combined z vectors shape: {combined_z_vectors.shape}")
    
    # Apply PCA to reduce dimensionality    
    logger.info("Applying PCA to reduce dimensionality before t-SNE...")
    pca = PCA(n_components=50,
              random_state=SEED)
    latents_pca = pca.fit_transform(combined_z_vectors)
    logger.info(f"PCA results shape: {latents_pca.shape}")
    
    loop = tqdm(perplexity, desc="Calculating t-SNE", total=len(perplexity), leave=False)
    for perp in loop:
        loop.set_description(f"Calculating t-SNE with perplexity {perp}")
        tsne = TSNE(n_components=tsne_components, 
                    perplexity=perp, 
                    random_state=SEED,
                    method='exact')
        combined_tsne = tsne.fit_transform(latents_pca)
        # Save t-SNE results to the reconstructed datasets
        separate_tsne_results = np.split(combined_tsne, np.cumsum([len(recon_df[period]['z']) for period in recon_df if period!='combined']), axis=0)      
        logger.info(f"Combined t-SNE shape: {combined_tsne.shape}")
        logger.info(f"Separate t-SNE shapes: {[tsne.shape for tsne in separate_tsne_results]}")
        for i, period in enumerate(recon_df):
            
            if period in ['test', 'val', 'train']:
                tsne_period = separate_tsne_results[i]
                logger.info(f"t-SNE shape for {period}: {tsne_period.shape}")
                n_components = np.arange(gmm_components[0], gmm_components[1] + 1)
                bics = {}
                logger.info(f"Fitting GMM for {period} with perplexity {perp} for {n_components} components")
                gmm_loop = tqdm(n_components, desc="Fitting GMM", total=len(n_components), leave=True)
                for n in gmm_loop:
                    gmm = GaussianMixture(
                        n_components=n, 
                        covariance_type='full', 
                        random_state=SEED)
                    gmm.fit(tsne_period)
                    bics[n] = gmm.bic(tsne_period)

                plot_bics(
                    bics=bics,
                    perplexity=perp,
                    period=period,
                    output_dir=cfg['output_dir'],
                    file_suffix=file_suffix
                )
                optimal_n = min(bics, key=bics.get)
                logger.info(f"Optimal number of GMM components for {period} with perplexity {perp}: {optimal_n}")
                gmm = GaussianMixture(n_components=optimal_n, covariance_type='full', random_state=SEED)
                labels = gmm.fit_predict(tsne_period)
                for c, mean in enumerate(gmm.means_):
                    logger.info(f"Cluster {c}: {len(np.where(labels == c)[0])} samples, mean: {mean}")
                    logger.info(f"Cluster {c}: {separate_tsne_results[i].shape} samples, mean: {mean}")
                recon_df[period][perp] = {
                    'tsne': separate_tsne_results[i],
                    'gmm': gmm,
                    'labels': labels,
                    'bics': bics,
                    'optimal_n': optimal_n,
                }
            else:
                logger.info(f"t-SNE shape for {period}: {separate_tsne_results[i].shape}")
                recon_df[period][perp] = {
                    'tsne': separate_tsne_results[i],
                }
                
        if 'combined' not in recon_df:
            recon_df['combined'] = {
                perp: {
                'tsne': combined_tsne,
                'kl_divergence': tsne.kl_divergence_
                }
            }
            recon_df['combined'][perp]['tsne'] = combined_tsne
            recon_df['combined']['date_labels'] = combined_date_labels
            recon_df['combined']['kl_divergence'] = tsne.kl_divergence_
        else:
            recon_df['combined'][perp]['tsne'] = combined_tsne
            recon_df['combined']['date_labels'] = combined_date_labels
            recon_df['combined'][perp]['kl_divergence'] = tsne.kl_divergence_
        loop.set_postfix({
            "t-SNE shape": combined_tsne.shape,
            "KL Divergence": tsne.kl_divergence_
        })

    # save the recon_df to a file
    if len(perplexity) > 1 or len(gmm_components) > 1:
        base_filename = f"recon_df_perplexity_{perplexity[0]}_{perplexity[-1]}_gmm_{gmm_components[0]}_{gmm_components[-1]}"
    else:
        base_filename = f"recon_df_perplexity_{perplexity[0]}_gmm_{gmm_components[0]}"
        
    recon_df_filename = f"{base_filename}.pkl"
    output_path = os.path.join(cfg['output_dir'], recon_df_filename)
        
    # check if the file already exists. If it does, append a number to the filename. Else, save it as is.
    i = 1
    while os.path.exists(output_path):
        logger.info(f"Saved recon_df file already exists: {output_path}.")
        recon_df_filename = f"{base_filename}_{i}.pkl"
        output_path = os.path.join(cfg['output_dir'], recon_df_filename)            
        i += 1
        
    with open(output_path, 'wb') as f:
        pickle.dump(recon_df, f)
    logger.info("Reconstructed data saved to {}".format(output_path))
    
    return recon_df


# %%
def plot_gmm_ellipses(ax, gmm, color='r'):
    """
    Plots Gaussian mixture model components as ellipses.
    """
    gmm_cluster_info = []

    centroids = gmm.means_
    covariances = gmm.covariances_

    for i, (mean, cov) in enumerate(zip(centroids, covariances)):
        vals, vecs = np.linalg.eigh(cov)
        angle = np.arctan2(vecs[0, 1], vecs[0, 0]) * 180 / np.pi
        width, height = 2 * np.sqrt(vals)
        ellipse = Ellipse(mean, 
                          width, 
                          height, 
                          angle=angle, 
                          edgecolor='red', 
                          facecolor='none', 
                          lw=2)
        ax.add_patch(ellipse)
        ax.plot(mean[0], mean[1], 'o', color=color, markersize=8)
        ax.text(mean[0], mean[1], str(i), fontsize=10, color='black', ha='center', va='center', weight='bold')
        
        gmm_cluster_info.append({
            'mean': mean,
            'covariance': cov,
            'ellipse': ellipse
        })
        
    return gmm_cluster_info


# %%
def select_best_perplexity(perplexities, silhouette_scores, kl_divergences, output_dir, method="combined", sil_threshold=0.5, file_suffix=None):
    """
    Select the best perplexity based on silhouette and KL divergence metrics.

    Args:
        perplexities (list): List of perplexity values.
        silhouette_scores (list): List of Silhouette Scores (higher is better).
        kl_divergences (list): List of KL Divergences from t-SNE (lower is better).
        method (str): Selection strategy: "combined" (default), "threshold", or "silhouette_only".
        sil_threshold (float): Used only if method == "threshold".

    Returns:
        dict: {
            "best_perplexity": int,
            "index": int,
            "silhouette": float,
            "kl_divergence": float,
            "ranking": list of (perplexity, silhouette, kl, combined_score)
        }
    """
    sil = np.array(silhouette_scores)
    kl = np.array(kl_divergences)
    perp = np.array(perplexities)

    if method == "combined":
        # Normalize
        sil_norm = (sil - sil.min()) / (sil.max() - sil.min() + 1e-8)
        kl_norm = (kl.max() - kl) / (kl.max() - kl.min() + 1e-8)
        combined_score = sil_norm + kl_norm

        best_idx = np.argmax(combined_score)

    elif method == "threshold":
        valid_idxs = [i for i in range(len(sil)) if sil[i] > sil_threshold]
        if valid_idxs:
            best_idx = min(valid_idxs, key=lambda i: kl[i])
        else:
            raise ValueError("No perplexity met the silhouette threshold.")

    elif method == "silhouette_only":
        best_idx = np.argmax(sil)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'combined', 'threshold', or 'silhouette_only'.")

    ranking = []
    for i in range(len(perp)):
        score = None
        if method == "combined":
            score = sil_norm[i] + kl_norm[i]
        ranking.append((perp[i], sil[i], kl[i], score))

    # plot combined scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(perp, [x[3] for x in ranking], marker='o', color='purple')
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Combined Score")
    # ax.set_title("Combined Score vs. Perplexity")
    plt.grid()
    plt.tight_layout()
    figure_name = os.path.join(output_dir, f"perplexity_scores_{method}")
    if file_suffix is not None:
        figure_name += f"_{file_suffix}"
    plt.savefig(figure_name + '.png', dpi=300)
    plt.savefig(figure_name + '.pdf', dpi=300)
    plt.close(fig)
    
    
    return {
        "best_perplexity": int(perp[best_idx]),
        "index": int(best_idx),
        "silhouette": float(sil[best_idx]),
        "kl_divergence": float(kl[best_idx]),
        "ranking": sorted(ranking, key=lambda x: -(x[3] if x[3] is not None else 0))  # highest score first
    }
    
    
# %%
def plot_bics(bics, perplexity, period, output_dir, file_suffix=None):
    """
    Plot BIC values for different GMM components.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on.
        bics (list): List of BIC values.
        perplexity (int): Perplexity value used for t-SNE.
    """
    # Plot BIC values for different GMM components bics={n_components: bic}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bics.keys(), bics.values(), marker='o', color='purple')
    ax.set_xlabel("Number of GMM Components")
    ax.set_ylabel("BIC")
    ax.set_title(f"BIC vs. Number of GMM Components (Perplexity: {perplexity})")
    plt.grid()
    plt.tight_layout()
    figure_name = os.path.join(output_dir, f"gmm_bics_p{perplexity}_{period}")
    if file_suffix is not None:
        figure_name += f"_{file_suffix}"
    plt.savefig(figure_name + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(figure_name + '.pdf', dpi=300, bbox_inches='tight')
    logger.info(f"Saved BIC plot to {figure_name}.png and {figure_name}.pdf")  
    plt.close(fig)
    
    
# %%
def get_gmm_centroids_and_closest_points(gmm, data):
    """
    Get GMM centroids and the closest points to each centroid.

    Args:
        gmm (GaussianMixture): Fitted GMM model.
        data (np.ndarray): Data points.

    Returns:
        tuple: Centroids and closest points to each centroid.
    """
    # Get GMM centroids
    centroids = gmm.means_

    # Calculate distances from each point to each centroid
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

    # Find the closest points to each centroid
    closest_points_indices = np.argmin(distances, axis=0)
    closest_points = data[closest_points_indices]

    return centroids, closest_points

# %%
def compute_cluster_composites(input_x, tsne_data, centroids, n_samples):
    # Compute distances: shape (N, K)
    distances = np.linalg.norm(tsne_data[:, np.newaxis] - centroids, axis=2)

    # Get indices of top_k closest samples to each centroid: shape (top_k, K)
    logger.info(f"Distances shape: {distances.shape}")
    
    if isinstance(n_samples, int) and distances.shape[0] < n_samples:
        logger.warning(f"Not enough samples for n_samples={n_samples}. Using all available samples.")
        n_samples = distances.shape[0]
        logger.info(f"Using all available samples: {n_samples}")
    elif n_samples == "all":
        n_samples = distances.shape[0]
    
    # Get the indices of the closest samples for each centroid
    closest_indices = np.argsort(distances, axis=0)[:n_samples]

    # Allocate array for the composite: shape (K, 9, 11, 64, 192)
    num_clusters = centroids.shape[0]
    composite_array = np.zeros((num_clusters, *input_x.shape[1:]), dtype=np.float32)

    for cluster_idx in range(num_clusters):
        indices = closest_indices[:, cluster_idx]  # top_k indices for this cluster
        cluster_samples = input_x[indices]         # shape: (top_k, 9, 11, 64, 192)
        composite = np.mean(cluster_samples, axis=0)
        composite_array[cluster_idx] = composite
    logger.info(f"Composite shape: {composite_array.shape}")
    return composite_array, closest_indices


# %%
def compute_composite_closest_to_sample(input_x, tsne_data, date_labels, target_date, n_samples=10):
    """
    Compute composite of samples closest to a given sample (by date) in t-SNE space.

    Args:
        input_x (np.ndarray): Input data of shape (N, C, T, H, W)
        tsne_data (np.ndarray): t-SNE embedding of shape (N, 2)
        date_labels (list or np.ndarray): List of date strings (format 'YYYY-MM-DD')
        target_date (str): The date to locate the center sample
        n_samples (int): Number of closest samples to include in composite

    Returns:
        np.ndarray: Composite of shape (C, T, H, W)
        np.ndarray: Indices of the closest samples used
        int: Index of the center sample (closest match to the date)
    """
    date_labels = np.array(date_labels)
    label_dates = np.array([np.datetime64(d) for d in date_labels])
    target_date = np.datetime64(target_date)
    logger.info(f"Target date: {target_date}")
    # Find the index of the sample closest in date
    date_diffs = np.abs(label_dates - target_date)
    center_idx = np.argmin(date_diffs)
    logger.info(f"Center index: {center_idx}, Date: {date_labels[center_idx]}")
    # Compute distances in t-SNE space to that sample
    distances = np.linalg.norm(tsne_data - tsne_data[center_idx], axis=1)

    # Get indices of the closest samples
    closest_indices = np.argsort(distances)[:n_samples]

    # Compute composite over these closest samples
    composite = np.mean(input_x[closest_indices], axis=0)

    return composite, closest_indices, center_idx, date_labels[center_idx]


# %%
def plot_cluster_variable_panels_vertical(
    reconstructed_data,
    cfg,
    period='test',
    perplexity=50,
    n_samples=100,
    mode='composite',
    dpi=300,
    date=None,
    file_suffix=None,
):
    assert mode in ['composite', 'closest', 'date'], "mode must be 'composite', 'closest', or 'date'"

    input_x = reconstructed_data[period]['x']
    date_labels = reconstructed_data[period]['date_labels']
    tsne = reconstructed_data[period][perplexity]['tsne']
    gmm = reconstructed_data[period][perplexity]['gmm']
    n_clusters = gmm.n_components
    centroids = gmm.means_

    var_labels = cfg['data']['variables']
    var_labels = [var.replace('stream', 'stream250').replace('z', 'z500') for var in var_labels]
    n_vars = input_x.shape[1]
    n_steps = input_x.shape[2]
    timestep_labels = [f"t={offset}" if offset == 0 else f"t{offset:+}" for offset in range(-5, 6)]

    min_lon, max_lon = -74.38596, 59.64912
    min_lat, max_lat = 30.23438, 74.53125

    if mode == 'closest':
        distances = np.linalg.norm(tsne[:, np.newaxis] - centroids, axis=2)
        closest_idx = np.argmin(distances, axis=0)
        sample_set = input_x[closest_idx]
        sample_dates = date_labels[closest_idx]

    elif mode == 'composite':
        sample_set, _ = compute_cluster_composites(input_x, tsne, centroids, n_samples=n_samples)
        sample_dates = None

    elif mode == 'date':
        assert date is not None, "date must be provided for 'date' mode"
        sample_set, closest_indices, center_idx, closest_date = compute_composite_closest_to_sample(
            input_x, tsne, date_labels, target_date=date, n_samples=n_samples
        )
        sample_set = sample_set[np.newaxis, ...]
        sample_dates = [date_labels[i] for i in closest_indices]
        n_clusters = 1
    else:
        raise ValueError("Invalid mode.")

    for v, var_name in enumerate(var_labels):
        cmap = var_cmaps.get(var_name, 'bwr')
        fig = plt.figure(figsize=(n_clusters * 2.2, n_steps * 0.7))
        spec = gridspec.GridSpec(n_steps, n_clusters + 1, 
                                 height_ratios=[1] * n_steps,
                                 width_ratios=[1] * n_clusters + [0.1],
                                 hspace=0.01,
                                 wspace=0.1)

        v_data = sample_set[:, v, :, :, :].ravel()
        vmin, vmax = np.percentile(v_data, [1, 99])
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        imgs = []
        for t in range(n_steps):
            for i in range(n_clusters):
                ax = fig.add_subplot(spec[t, i], projection=ccrs.PlateCarree())

                lons = np.linspace(min_lon, max_lon, sample_set.shape[-1])
                lats = np.linspace(min_lat, max_lat, sample_set.shape[-2])
                lon_grid, lat_grid = np.meshgrid(lons, lats)

                img = ax.contourf(
                    lon_grid, lat_grid,
                    sample_set[i, v, t],
                    levels=10,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax,
                    norm=divnorm
                )
                imgs.append(img)

                ax.coastlines(linewidth=0.3, color='black')
                ax.set_xticks([])
                ax.set_yticks([])

                if i == 0:
                    ax.set_ylabel(timestep_labels[t], fontsize=13, labelpad=15, rotation=0,
                                  verticalalignment='center')
                if t == 5:
                    rect = Rectangle(
                        (min_lon, min_lat),
                        max_lon - min_lon,
                        max_lat - min_lat,
                        linewidth=4,
                        edgecolor='black',
                        facecolor='none',
                        transform=ccrs.PlateCarree(),
                    )
                    ax.add_patch(rect)
                if t == 0:
                    label = f"#{i+1}"
                    if mode == 'closest' and sample_dates is not None:
                        label += f"\n{sample_dates[i]}"
                    ax.set_title(label, fontsize=16)

        # Colorbar on the bottom
        cbar_width = 0.5  # width as fraction of figure
        cbar_height = 0.025  # height as fraction of figure
        cbar_left = 0.25  # center it (0.3 + 0.4 = 0.7)
        cbar_bottom = 0.07  # distance from bottom

        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(imgs[-1], cax=cbar_ax, orientation='horizontal')
        cbar.set_label(var_labels[v], fontsize=12, labelpad=4)
        cbar.ax.tick_params(labelsize=8)

        # plt.tight_layout(rect=[0.1, 0.3, 1, 1]) # left, bottom, right, top
        if mode == 'closest':
            mode_str = "closest"
        elif mode == 'composite':
            mode_str = f"{mode}_{n_samples}"
        elif mode == 'date':
            mode_str = f"{mode}_target{date}_closest{closest_date}_n{n_samples}"
        else:
            raise ValueError("Invalid mode.")

        fname = f"{var_labels[v]}_{mode_str}_{period}_p{perplexity}_c{n_clusters}_VERTICAL"
        if file_suffix:
            fname += f"_{file_suffix}"
        out_path = os.path.join(cfg['output_dir'], fname)
        plt.savefig(out_path + ".png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✅ Saved vertical layout: {out_path}")
        

# %%
def plot_time_composites_by_var_and_date(
    reconstructed_data,
    cfg,
    period='test',
    perplexity=50,
    target_dates=['2003-07-13', '2010-07-15'],
    var_list=['t2m', 'z500'],
    n_samples=10,
    dpi=300,
    file_suffix=None,
):
    """
    Plot composites for multiple dates and variables.

    For each variable in var_list, creates a figure with rows = dates and cols = timesteps.

    Args:
        reconstructed_data (dict): Dict containing t-SNE and data.
        cfg (dict): Configuration dictionary.
        period (str): Dataset period (e.g., 'train', 'val', 'test').
        perplexity (int): t-SNE perplexity.
        target_dates (list): List of date strings (YYYY-MM-DD).
        var_list (list): List of variables to plot.
        n_samples (int): Number of closest samples to average.
        dpi (int): DPI for the saved figure.
    """
    # 
    input_x = reconstructed_data[period]['x']
    date_labels = reconstructed_data[period]['date_labels']
    tsne_data = reconstructed_data[period][perplexity]['tsne']

    n_vars = input_x.shape[1]
    n_steps = len(target_dates)
    nrows, ncols = get_grid_dims(n_steps, 4)
    var_labels_full = [v.replace('stream', 'stream250').replace('z', 'z500') for v in cfg['data']['variables']]
    timestep_labels = [f"t={offset}" if offset == 0 else f"t{offset:+}" for offset in range(-5, 6)]
    var_to_idx = {v: i for i, v in enumerate(cfg['data']['variables'])}

    min_lon, max_lon = -74.38596, 59.64912
    min_lat, max_lat = 30.23438, 74.53125

    
    for var in var_list:
        if var not in var_to_idx:
            logger.warning(f"Variable {var} not found in data, skipping.")
            continue
        var_idx = var_to_idx[var]
        var_label = var_labels_full[var_idx]
        
        fig, axes = plt.subplots(
            nrows=nrows, 
            ncols=ncols,
            figsize=(ncols*2, nrows * 4),
            subplot_kw={'projection': ccrs.PlateCarree()},
            gridspec_kw={'hspace': 0.02, 'wspace': 0.05},
            dpi=dpi
        )
        axes = np.atleast_2d(axes)
        
        vmin, vmax = np.percentile(input_x[:, var_idx, :, :], [0.1, 99.9])
        logger.info(f"Variable {var_label} vmin: {vmin}, vmax: {vmax}")
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
        imgs = []
        for col_idx, target_date in enumerate(target_dates):
            composite, closest_indices, center_idx, closest_date = compute_composite_closest_to_sample(
                input_x, tsne_data, date_labels, target_date, n_samples=n_samples
            )

            logger.info(f"Composite for var {var}, date {target_date}: center={closest_date}, samples={n_samples}")

            v_data = composite[var_idx]  # shape: (n_steps, lat, lon)
            v_data_time_composite = np.mean(v_data, axis=0)  # shape: (lat, lon)
            cmap = var_cmaps.get(var_label, 'bwr')
                        

            ax = axes[0, col_idx]
            lons = np.linspace(min_lon, max_lon, composite.shape[-1])
            lats = np.linspace(min_lat, max_lat, composite.shape[-2])
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            img = ax.contourf(
                lon_grid, lat_grid,
                v_data_time_composite,
                levels=10,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                vmin=vmin, vmax=vmax,
                norm=divnorm
            )
            ax.coastlines(linewidth=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(1.5)
            ax.set_extent([-20, 45, 32, 72])
            imgs.append(img)
            
            title_str = (
                f"{'Target:'.ljust(8)}{target_date.replace('-', '.').rjust(11)}\n"
                f"{'Closest:'.ljust(8)}{closest_date.replace('-', '.').rjust(11)}"
            )
            ax.set_title(title_str, 
                         fontsize=8, 
                         fontfamily='DejaVu Sans Mono',
                         ha='center')
            # if col_idx == 0:
            #     ax.set_ylabel(var_label, 
            #                   fontsize=10, 
            #                   rotation=0, 
            #                   labelpad=0, 
            #                   ha='right', 
            #                   va='center')
            
                
        # 
        cbar_ax = fig.add_axes([0.92, 0.33, 0.015, 0.3])  # [left, bottom, width, height]
        cbar = fig.colorbar(imgs[-1], cax=cbar_ax, orientation='vertical')
        cbar.set_label(var_label, fontsize=8, labelpad=-2)
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 1]) # left, bottom, right, top
        plt.subplots_adjust(left=0.20, right=0.91)
        fname = f"composites_{var}_n{n_samples}_{period}_p{perplexity}"
        if file_suffix:
            fname += f"_{file_suffix}"
        out_path = os.path.join(cfg['output_dir'], fname)
        plt.savefig(out_path + ".png", dpi=dpi, bbox_inches='tight')
        plt.savefig(out_path + ".pdf", dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✅ Saved composite plot: {out_path}")

    
# %%
def plot_bics_all_periods(recon_df, cfg, perplexity=50, periods=('train', 'val', 'test'), file_suffix=None, dpi=300):
    """
    Plot BIC scores for each period (train, val, test) as subplots in a single figure.

    Args:
        recon_df (dict): Dictionary containing t-SNE and GMM results including BICs.
        cfg (dict): Configuration dictionary with 'output_dir'.
        perplexity (int): t-SNE perplexity value used for GMM fitting.
        periods (tuple): List or tuple of periods to include in the figure.
        file_suffix (str): Optional suffix to add to filename.
        dpi (int): Resolution of the saved figure.
    """
    n_periods = len(periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(5 * n_periods, 4))

    if n_periods == 1:
        axes = [axes]

    for i, period in enumerate(periods):
        bics = recon_df[period][perplexity]['bics']
        x = list(bics.keys())
        y = list(bics.values())
        axes[i].plot(x, y, marker='o')
        axes[i].set_title(f'{period.capitalize()}')
        if i == 1:
            axes[i].set_xlabel('Number of GMM Components')
        if i == 0:
            axes[i].set_ylabel('BIC Score')
        axes[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"bic_scores_all_periods_perp{perplexity}"
    if file_suffix:
        filename += f"_{file_suffix}"
    filename += ".png"
    filepath = os.path.join(cfg['output_dir'], filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"BIC summary plot saved to {filepath}")


# %%
def calculate_r2_score(originals, reconstructions, cfg, file_suffix=None):
    """
    Calculate R² score between original data and reconstructions.
    
    Args:
        originals: tensor of shape (batch, channels, time, lon, lat)
        reconstructions: tensor of same shape as originals
        
    Returns:
        float: overall R² score
        dict: R² scores per channel
    """
    # Ensure inputs are numpy arrays
    if not isinstance(originals, np.ndarray):
        originals = originals.cpu().detach().numpy()
    if not isinstance(reconstructions, np.ndarray):
        reconstructions = reconstructions.cpu().detach().numpy()
    
    logger.info(f"Originals shape: {originals.shape}")
    logger.info(f"Reconstructions shape: {reconstructions.shape}")
    
    # calculate overall R² score
    orig_flat = originals.reshape(originals.shape[0], -1)
    recon_flat = reconstructions.reshape(reconstructions.shape[0], -1)
    
    overall_r2 = r2_score(orig_flat, recon_flat)
    
    # Calculate R² score per channel
    channel_r2 = {}
    for c, cname in enumerate(cfg['data']['variables']):
        orig_c = originals[:, c, ...].reshape(originals.shape[0], -1)
        recon_c = reconstructions[:, c, ...].reshape(reconstructions.shape[0], -1)
        logger.info(f"Channel {cname} shape: {orig_c.shape}")
        channel_r2[f'channel_{cname}'] = r2_score(orig_c, recon_c)
    
    r2_per_sample = []
    for i in range(originals.shape[0]):  # over samples
        originals_i = originals[i].reshape(-1)
        reconstructions_i = reconstructions[i].reshape(-1)
        r2_i = r2_score(originals_i, reconstructions_i)
        r2_per_sample.append(r2_i)
        
    # print the sample with the highest R² score
    max_r2_idx = np.argmax(r2_per_sample)
    logger.info(f"Sample with highest R² score: {max_r2_idx} with R² score: {r2_per_sample[max_r2_idx]:.4f}")
    
    # save the overall, per-channel, and per-sample R² scores to CSV
    r2_df = pd.DataFrame({
        'overall_r2': [overall_r2],
        'per_channel_r2': [channel_r2],
        'per_sample_r2': [r2_per_sample]
    })
    file_name = f'r2_scores'
    if file_suffix is not None:
        file_name += f'_{file_suffix}'
    file_name += '.csv'
    r2_df.to_csv(os.path.join(cfg['output_dir'], file_name), index=False)
    
    logger.info(f"Overall R² score: {overall_r2:.4f}")
    logger.info("R² score per channel:")
    for channel, score in channel_r2.items():
        logger.info(f"  {channel}: {score:.4f}")
        
    return overall_r2, channel_r2

# %%
def plot_combined_tsne_figure(recon_dict, cfg, perplexity=50, dpi=300, file_suffix=None):
    """
    Combines t-SNE plots for train, val, test, combined, and seasonal GMM clustering into a single figure.
    Layout:
        ABEE
        CDEE
    """
    # Setup
    periods = ['train', 'val', 'test']
    markers = ['o', 's', '^']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    season_colors = ['cornflowerblue', 'mediumseagreen', 'orange', 'indianred']
    season_list = ['DJF', 'MAM', 'JJA', 'SON']
    season_to_int = {season: idx for idx, season in enumerate(season_list)}
    subplot_labels = {
        'train': 'A',
        'val': 'B',
        'test': 'C',
        'combined': 'D',
        'distribution': 'E',
        'seasons': 'F',
        
    }
    label_fontsize = 15
    title_fontsize = 20
    ticks_fontsize = 12
    legend_fontsize = 12
    fig = plt.figure(figsize=(16, 8))
    
    
    
    mosaic = [
        ['train', 'train', 'val', 'val', 'distribution', 'distribution', 'distribution'],
        ['train', 'train', 'val', 'val', 'seasons', 'seasons', 'seasons'], 
        ['test', 'test', 'combined', 'combined', 'seasons', 'seasons', 'seasons'], 
        ['test', 'test', 'combined', 'combined', 'seasons', 'seasons', 'seasons']
]
  
    ax_dict = fig.subplot_mosaic(mosaic)
    
    x_y_ref_ax = ax_dict['train']
    # Individual t-SNE panels ('train', 'val', 'test')
    for i, period in enumerate(periods):
        ax_period = ax_dict[period]
        if period != 'train':
            ax_period.sharex(x_y_ref_ax)
            ax_period.sharey(x_y_ref_ax)
        label = subplot_labels[period]
        ax_period.text(0.05, 0.95, label, transform=ax_period.transAxes,
                 fontsize=title_fontsize, va='top', ha='left')
        tsne_data = recon_dict[period][perplexity]['tsne']
        ax_period.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=0.6, c=colors[i], 
                   marker=markers[i], label=period, s=10)
        start_year = cfg['data'][f'{period}_years'][0]
        end_year = cfg['data'][f'{period}_years'][-1]
        # ax.set_title(f"{period} ({start_year}-{end_year})", fontsize=title_fontsize)
        if period == 'train':
            ax_period.set_ylabel("t-SNE 2", fontsize=label_fontsize)
        if period == 'test':
            ax_period.set_xlabel("t-SNE 1", fontsize=label_fontsize)
            ax_period.set_ylabel("t-SNE 2", fontsize=label_fontsize)
        # ax.legend(fontsize=legend_fontsize)


    # Combined panel
    ax_d = ax_dict['combined']
    for i, period in enumerate(periods):
        tsne_data = recon_dict[period][perplexity]['tsne']
        ax_d.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                     alpha=0.6, c=colors[i], marker=markers[i], s=5,
                     label=period)
        ax_d.sharex(x_y_ref_ax)
        ax_d.sharey(x_y_ref_ax)
    start_year = cfg['data']['train_years'][0]
    end_year = cfg['data']['test_years'][-1]
    ax_d.set_xlabel("t-SNE 1", fontsize=label_fontsize)
    ax_d.legend(fontsize=legend_fontsize, markerscale=2, framealpha=0.5, labelspacing=0.2, handletextpad=0.1)
    label = subplot_labels['combined']
    ax_d.text(0.05, 0.95, label, transform=ax_d.transAxes,
             fontsize=title_fontsize, va='top', ha='left')
    
    
    # Seasonal panel
    tsne_data = recon_dict['test'][perplexity]['tsne']
    date_labels = recon_dict['test']['date_labels']
    gmm = recon_dict['test'][perplexity]['gmm']
    centroids = gmm.means_
    covariances = gmm.covariances_

    month_nums = np.array([int(d[5:7]) for d in date_labels])
    seasons = np.array([season_list[(m % 12) // 3] for m in month_nums])
    season_indices = np.array([season_to_int[s] for s in seasons])

    ax_seasons = ax_dict['seasons']
    label = subplot_labels['seasons']
    ax_seasons.text(0.02, 0.97, label, transform=ax_seasons.transAxes,
                  fontsize=title_fontsize, va='top', ha='left')
    ax_seasons.scatter(tsne_data[:, 0], tsne_data[:, 1], c=season_indices,
                cmap=mcolors.ListedColormap(season_colors), s=15, alpha=0.7)
    ax_seasons.sharex(x_y_ref_ax)
    ax_seasons.sharey(x_y_ref_ax)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=season,
                   markerfacecolor=color, markersize=6)
        for season, color in zip(season_list, season_colors)
    ]
    ax_seasons.legend(handles=handles, loc='best', fontsize=legend_fontsize, markerscale=1.5,framealpha=0.5,labelspacing=0.2, handletextpad=0.1)
    for i, (mean, cov) in enumerate(zip(centroids, covariances)):
        vals, vecs = np.linalg.eigh(cov)
        angle = np.arctan2(vecs[0, 1], vecs[0, 0]) * 180 / np.pi
        width, height = 2 * np.sqrt(vals)
        ellipse = Ellipse(mean, width, height, angle, edgecolor='black', facecolor='none', lw=2)
        ax_seasons.add_patch(ellipse)
        ax_seasons.text(mean[0], mean[1], str(i + 1), fontsize=15, ha='center', va='center', weight='bold')
    ax_seasons.set_xlabel("t-SNE 1", fontsize=label_fontsize)


    # Distribution panel
    ax_dist = ax_dict['distribution']
    gmm_labels = recon_dict['test'][perplexity]['labels']
    n_clusters = np.max(gmm_labels) + 1

    # Assign seasons to each date
    month_nums = np.array([int(d[5:7]) for d in date_labels])
    seasons = np.array([season_list[(m % 12) // 3] for m in month_nums]) 

    # Count and normalize
    counts = np.zeros((n_clusters, len(season_list)))
    for cluster_id in range(n_clusters):
        cluster_seasons = seasons[gmm_labels == cluster_id]
        
        for i, season in enumerate(season_list):
            counts[cluster_id, i] = np.sum(cluster_seasons == season)
    percentages = counts / counts.sum(axis=1, keepdims=True) * 100
    logger.info(f"Cluster percentages: {percentages}")
    
     # Plot grouped bars
    bar_width = 0.2
    x = np.arange(n_clusters)
    for i, season in enumerate(season_list):
        bars = ax_dist.bar(x + i * bar_width, percentages[:, i], width=bar_width,
                    color=season_colors[i], label=season)
        # Add text inside the bar at the bottom
        for j, bar in enumerate(bars):
            bar_x = bar.get_x() + bar.get_width() / 2
            bar_height = bar.get_height()
            label = f"{percentages[j, i]:.1f}"
            ax_dist.text(bar_x, 1, label, ha='center', va='bottom', fontsize=10, color='black', rotation=90)      

    ax_dist.set_xticks(x + 1.5 * bar_width)
    ax_dist.set_xticklabels([f"#{i+1}" for i in range(n_clusters)])
    ax_dist.set_ylabel("%", rotation=0, fontsize=label_fontsize, labelpad=10)
    ax_dist.legend(ncols=2, framealpha=0.5,labelspacing=0.2, handletextpad=0.1)
    label = subplot_labels['distribution']
    ax_dist.text(0.02, 0.97, label, transform=ax_dist.transAxes,
             fontsize=title_fontsize, va='top', ha='left')


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(cfg['output_dir'], f"combined_tsne_gmm_p{perplexity}")
    if file_suffix:
        save_path += f"_{file_suffix}"
    plt.savefig(save_path + '.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined t-SNE plot: {save_path}")


def modify_cfg_output_dir(cfg):
    """
    Modify the output directory in the configuration.
    
    Args:
        cfg (dict): Configuration dictionary.
        
    Returns:
        dict: Modified configuration dictionary.
    """
    current_time = datetime.now().strftime("%y%m%d-%H%M%S")
    if cfg['output_dir'].split('/')[-2] == 'results':
        cfg['org_output_dir'] = cfg['output_dir']
        cfg['output_dir'] = os.path.join(cfg['output_dir'], current_time)
        os.makedirs(cfg['output_dir'], exist_ok=True)
        logger.info(f"Output directory: {cfg['output_dir']}")
    elif bool(re.fullmatch(r'\d+_\d+', cfg['output_dir'].split('/')[-2])):
        cfg['org_output_dir'] = '/'.join(cfg['output_dir'].split('/')[:-1])
        cfg['output_dir'] = os.path.join('/'.join(cfg['output_dir'].split('/')[:-1]), current_time)
        os.makedirs(cfg['output_dir'], exist_ok=True)
        logger.info(f"Output directory: {cfg['output_dir']}")
    else:
        logger.info("Unrecognized output directory format. Using original output directory.")
        logger.info(f"Output directory: {cfg['output_dir']}")
    
    return cfg, current_time


# %% 
def main(cfg, DEVICE):

    # %%
    logger.info(f"*** test {cfg['model']['name']} ***")
    logger.info("📅 Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # %%
    logger.info("🤖 Loading model...")
    model = load_trained_model(cfg, DEVICE)
    
    # %%
    logger.info("📦 Modify configuration...")
    cfg, current_time = modify_cfg_output_dir(cfg)

    # %%
    logger.info("📦 Check configuration...")
    logger.info(f"Variables: {cfg['data']['variables']}")
    logger.info(f"Saved model directory: {cfg['train']['checkpoint_dir']}")
    logger.info(f"Output directory: {cfg['output_dir']}")
    
    # %%
    logger.info("🔄 Reconstructing data..")
    recon_data_dict = create_recon_dict(
        [train_loader, val_loader, test_loader], 
        ['train', 'val', 'test'], 
        model, 
        DEVICE, 
        cfg
        )
    
    # %%
    overall_r2, per_channel_r2 = calculate_r2_score(
        recon_data_dict['test']['x'], 
        recon_data_dict['test']['recon_x'], 
        cfg=cfg,
        file_suffix=None
        ) 

    # %%
    perp_list = list(range(50, 51, 50)) # Fixed after testing
    gmm_components = (4, 4)  # Fixed after testing
    
    # %%    
    logger.info("Calculating t-SNE...")
    recon_data_dict = calculate_tsne_and_gmm(
        recon_df=recon_data_dict,
        cfg=cfg,
        perplexity=perp_list, 
        tsne_components=2, 
        gmm_components=gmm_components, 
        period='test',
        file_suffix=None
        )  
    
    # %%
    logger.info("Loading heatwave dataset...")
    heatwave_df = load_hw_dataset(cfg)
    
    # %%
    logger.info("Plotting heatwaves per year...")
    plot_hw_per_year(heatwave_df, cfg, file_suffix=None)
    
    # %% 
    plot_bics_all_periods(recon_data_dict, cfg, perplexity=50, file_suffix=None)

    # %%
    plot_combined_tsne_figure(recon_data_dict, cfg, perplexity=50, file_suffix="withdensity")
    
    # %%
    plot_time_composites_by_var_and_date(
        reconstructed_data=recon_data_dict,
        cfg=cfg,
        period='test',
        perplexity=50,
        target_dates=['2010-07-08', '2003-07-13','2021-06-19', '2018-07-11'],
        var_list=['z'],
        n_samples=10,
        file_suffix=None
    )
    
    # %%
    plot_cluster_variable_panels_vertical(
        reconstructed_data=recon_data_dict,
        cfg=cfg,
        period='test',
        perplexity=50,
        n_samples=100,
        mode='composite',
        dpi=300,
    )
    