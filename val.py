import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import vae_models
import utils
import logging

def set_logger(_logger):
    global logger
    logger = _logger
    utils.logger = logger

# load trained model from .pth file
def load_model(model, model_path):
    state_dict = torch.load(model_path)

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def validate_model(trained_model, device, val_loader):
    
    trained_model.eval()
    writer = SummaryWriter()
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.float().to(device)
            recon_batch, mu, logvar = trained_model(data)
            plot_reconstruction_loss_per_frame(recon_batch, data, writer)
            logger.info("Reconstructed Batch Shape: ", recon_batch.shape)
            logger.info("Mu Shape: ", mu.shape)
            logger.info("Logvar Shape: ", logvar.shape)
            
    return recon_batch, mu, logvar

def plot_reconstruction_loss_per_frame(recon_batch, data, writer,):
    loss = nn.MSELoss(reduction='sum')
    loss_per_frame = loss(recon_batch, data)
    writer.add_scalar('Reconstruction Loss per Frame', loss_per_frame)
    return loss_per_frame


def main(cfg, DEVICE):
    
    logger.info("*** Validating VAE LSTM model ***")
    val_loader = utils.create_dataloaders_from_npy(
        root_dir=cfg['data']['root_dir'], 
        variables=cfg['data']['variables'], 
        file_pattern=cfg['data']['file_pattern'], 
        batch_size=cfg['batch_size'], 
        years=tuple(cfg['data']['validation_years']),
        logger=logger,
        cfg=cfg
        )
    
    # print val_loader length
    logger.info("Validation Loader Length: {}".format(len(val_loader)))
    for i, data in enumerate(val_loader):
        logger.info(i, data.shape)
        break
    
    ModelClass = vae_models.get(cfg['model']['name'])
    if ModelClass is None:
        raise ValueError("Model {} is not defined in vae_models. Available models: {}".format(cfg['model']['name'], list(vae_models.keys())))
    model = ModelClass(
        input_dim=len(cfg['data']['variables']),
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        input_shape=(
            cfg['data']['temporal_resolution'], 
            cfg['data']['spatial_size'], 
            cfg['data']['spatial_size']
            ),
        kernel_size=cfg['model']['kernel_size'],
        stride=cfg['model']['stride'],
        padding=cfg['model']['padding']
    ).to(DEVICE)
    
    if torch.cuda.device_count() > 1:   
        model = nn.DataParallel(model)
    
        # find the model in the saved_models directory
    if cfg['testing']['load_model'] == 'last':
        model_dir = os.path.join(cfg['training']['checkpoint_dir'], max(os.listdir(cfg['training']['checkpoint_dir'])))
        # check if the model is following vae_lstm.pth naming convention otherwise choose the file from previous directory
        saved_models_listdir = sorted(os.listdir(cfg['training']['checkpoint_dir']))
        
        while not os.path.exists(os.path.join(model_dir, 'vae_lstm.pth')):
            model_dir = os.path.join(cfg['training']['checkpoint_dir'], saved_models_listdir[-1])
            saved_models_listdir.pop()
            
        # create a new directory in output_dir with the same name as the model directory
        cfg['testing']['output_dir'] = os.path.join(cfg['testing']['output_dir'], os.path.basename(model_dir))
        os.makedirs(cfg['testing']['output_dir'], exist_ok=True)
        
    elif utils.is_datetime_string(cfg['testing']['load_model']):
        model_dir = os.path.join(cfg['training']['checkpoint_dir'], cfg['testing']['load_model'])
        cfg['testing']['output_dir'] = os.path.join(cfg['testing']['output_dir'], cfg['testing']['load_model'])
        os.makedirs(cfg['testing']['output_dir'], exist_ok=True)
        
    else:
        logger.error("Invalid model directory in config file. Expected 'last' or 'YYYYMMDD-HHMMSS' format.")
        # print the line number of the error in the config file
        sys.exit(0)
    
    
    # load the trained model from the checkpoint
    checkpoint_files = [f for f in os.listdir(cfg['checkpoint_dir']) if f.startswith('vae_lstm')]
    if checkpoint_files:
        checkpoint_files.sort()
        checkpoint = torch.load(os.path.join(cfg['checkpoint_dir'], checkpoint_files[-1]))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded checkpoint from epoch: ", checkpoint['epoch'])
    else:
        logger.info("No checkpoints found to load")
    
    # validate the model
    recon_batch, mu, logvar = validate_model(model, DEVICE, val_loader)
    
    return recon_batch, mu, logvar

    
    