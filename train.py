import os
import re
import tqdm
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tabulate import tabulate
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from utils.data_loaders import load_heatwave_samples
from models import vae_models

logger = None

def set_logger(_logger):
    global logger
    logger = _logger
    load_heatwave_samples.set_logger(logger)


def pretty_print_stats(data, recon_data=None, name=""):
    stats = [
        ["Min", data.min().item(), recon_data.min().item() if recon_data is not None else None], 
        ["Max", data.max().item(), recon_data.max().item() if recon_data is not None else None],
        ["Mean", data.mean().item(), recon_data.mean().item() if recon_data is not None else None],
        ["Std", data.std().item(), recon_data.std().item() if recon_data is not None else None],
        ["Shape", data.shape, recon_data.shape if recon_data is not None else None],
        ["Type", data.dtype, recon_data.dtype if recon_data is not None else None]
    ] 
    table = tabulate(stats, headers=["Statistic", "Value", "Value"], tablefmt="fancy_grid")
    logger.info(f"{name} Stats:\n{table}")


def pretty_print_model_summary(model):
    logger.info("Model Summary:")
    logger.info(model)
    logger.info("Model Parameters:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.shape}")


def add_model_graph_to_tensorboard(model, device, train_loader, writer):
    """
    Adds the model graph to TensorBoard.
    """
    
    sample_data = next(iter(train_loader))
    if isinstance(sample_data, (list, tuple)):
        sample_data = sample_data[0]
    sample_data = sample_data[:1, :, :, :, :].to(device)
    try:
        if isinstance(model, torch.nn.DataParallel):
            writer.add_graph(model.module, sample_data)
        else:
            writer.add_graph(model, sample_data)
        logger.info("Added model graph to TensorBoard")
    except Exception as e:
        logger.info(f"Failed to add graph to TensorBoard: {str(e)}")


def save_best_checkpoint(checkpoint_dir, cfg, model, optimizer, total_loss, best_val_loss, epoch, avg_train_loss_after_epoch, lr, device):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"{cfg['model']['name']}_best.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        'best_val_loss': best_val_loss
    }, checkpoint_path)
    
    with open(os.path.join(checkpoint_dir, 'best_model_summary.txt'), 'w') as file:
        file.write(str(model))
    
    with open(os.path.join(checkpoint_dir, 'best_train_summary.txt'), 'w') as file:
        file.write(f"Epoch: {epoch}\n")
        file.write(f"Learning Rate: {lr}\n")
        file.write(f"Batch Size: {cfg['batch_size']}\n")
        file.write(f"Loss Function: VAE\n")
        file.write(f"Optimizer: {cfg['optimizer']['type']}\n")
        file.write(f"Scheduler: {cfg['scheduler']['type']}\n")
        file.write(f"Scheduler Params: {cfg['scheduler']['params']}\n")
        file.write(f"Device: {device}\n")
        file.write(f"Checkpoint Directory: {checkpoint_dir}\n")
        file.write(f"Best val Loss: {best_val_loss}\n")
        file.write(f"Avg train Loss: {avg_train_loss_after_epoch}\n")


def save_periodic_checkpoint(checkpoint_dir, cfg, model, optimizer, epoch, avg_val_loss):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    periodic_checkpoint_path = os.path.join(checkpoint_dir, f"{cfg['model']['name']}_epoch_{epoch+1}.pth")
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_val_loss': avg_val_loss
    }, periodic_checkpoint_path)
    print(f"Saved periodic checkpoint: {periodic_checkpoint_path}")


def save_final_model_and_summary(checkpoint_dir, cfg, model, epochs, lr, save_checkpoint, load_checkpoint, device):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, cfg['model']['name'] + '.pth'))
    
    # get the date_tiime from checkpoint_dir

    cfg['test']['load_model'] = current_time
    
    with open(os.path.join(checkpoint_dir, 'main.yaml'), 'w') as file:
        yaml.dump(cfg, file)
    
    with open(os.path.join(checkpoint_dir, 'final_model_summary.txt'), 'w') as file:
        file.write(str(model))
    
    with open(os.path.join(checkpoint_dir, 'final_train_summary.txt'), 'w') as file:
        file.write(f"Epochs: {epochs}\n")
        file.write(f"Learning Rate: {lr}\n")
        file.write(f"Batch Size: {cfg['batch_size']}\n")
        file.write(f"Loss Function: VAE\n")
        file.write(f"Optimizer: {cfg['optimizer']['type']}\n")
        file.write(f"Scheduler: {cfg['scheduler']['type']}\n")
        file.write(f"Scheduler Params: {cfg['scheduler']['params']}\n")
        file.write(f"Device: {device}\n")
        file.write(f"Checkpoint Directory: {checkpoint_dir}\n")
        file.write(f"Save Checkpoint: {save_checkpoint}\n")
        file.write(f"Load Checkpoint: {load_checkpoint}\n")


def loss_function(recon_x, x, mu, logvar, beta=1.0, use_L1=True, reduction='sum') -> tuple:
    """
    VAE loss function with KL divergence scaling and stability adjustments.
    """
    # Reconstruction Loss (MSE)
    if use_L1:
        loss_fn = nn.L1Loss(reduction='sum')
    else:
        loss_fn = nn.MSELoss(reduction='sum')  
        
    recon_loss = loss_fn(recon_x, x)
    
    # Stabilize logvar
    logvar = torch.clamp(logvar, min=-10, max=10)
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
    # Total Loss with Beta Scaling
    total_loss = recon_loss + beta * KLD
    return total_loss, recon_loss, KLD


def select_optimizer(optimizer_type, model, lr, **kwargs) -> torch.optim.Optimizer:
    try:
        optimizer_class = getattr(optim, optimizer_type)
        optimizer = optimizer_class(model.parameters(), lr=lr, **kwargs)
    except AttributeError:
        logger.error(f"Invalid optimizer type '{optimizer_type}'. Check PyTorch documentation for available optimizers.")
        raise ValueError(f"Invalid optimizer type '{optimizer_type}'. Check PyTorch documentation for available optimizers.")
    
    return optimizer


def define_lr_scheduler(optimizer, scheduler_type, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    try:
        scheduler_class = getattr(lr_scheduler, scheduler_type)
        scheduler = scheduler_class(optimizer, **kwargs)
    except AttributeError:
        logger.error(f"Invalid scheduler type '{scheduler_type}'. Check PyTorch documentation for available schedulers.")
        raise ValueError(f"Invalid scheduler type '{scheduler_type}'. Check PyTorch documentation for available schedulers.")
    
    return scheduler



def validate_step(model, device, val_loader, loss_function, writer, epoch, cfg):
    """
    Validates the model on the val set and logs metrics.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    val_mse_loss = 0
    val_kld_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (data, date_labels) in enumerate(val_loader):
            data = data.float().to(device)
            recon_batch, mu, logvar, z = model(data)
            total_samples += data.size(0)
            
            # Compute val loss
            loss, mse_loss, kld_loss = loss_function(recon_batch, data, mu, logvar)
            
            val_loss += loss.item()
            val_mse_loss += mse_loss.item()
            val_kld_loss += kld_loss.item()
            
    # Average metrics across the val set
    avg_val_loss = val_loss / total_samples
    avg_val_mse_loss = val_mse_loss / total_samples
    avg_val_kld_loss = val_kld_loss / total_samples
    
    # Log val metrics
    writer.add_scalar('val Loss', avg_val_loss, epoch)
    writer.add_scalar('val MSE Loss', avg_val_mse_loss, epoch)
    writer.add_scalar('val KLD', avg_val_kld_loss, epoch)
    
    logger.info(
        f"val ===> Epoch: {epoch + 1}: "
        f"Loss: {avg_val_loss:.4f}, MSE Loss: {avg_val_mse_loss:.4f}, KLD: {avg_val_kld_loss:.4f}"
    )
    
    return avg_val_loss


def train_step(train_loader, model, loss_function, optimizer, device, writer, epoch, epochs, cfg):
    """
    Trains the model for one epoch and logs metrics.
    """
    loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
    total_loss, total_samples = 0, 0
    
    model.train()  # Set model to train mode
    for batch_idx, (data, date_labels) in loop:  
        data = data.float().to(device) 
        recon_batch, mu, logvar, z = model(data)
        print(f"Data Shape: {data.shape}")
        print(f"Recon Batch Shape: {recon_batch.shape}")
        print(f"Data Min: {data.min().item()}, Max: {data.max().item()}")
        print(f"Recon Batch Min: {recon_batch.min().item()}, Max: {recon_batch.max().item()}")
        
        # pretty_print_stats(data, recon_batch, name="Batch")
        batch_size = data.size(0)
        
        beta = min(1.0, epoch / epochs)
        print(f"Beta: {beta}")
        loss, MSE, KLD = loss_function(recon_batch, data, mu, logvar, beta, use_L1=True, reduction='sum')
        
        # Backpropagation
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()             
        optimizer.zero_grad()
        loss_value = loss.item()
        mse_loss = MSE.item()
        kld_loss = KLD.item()
        total_loss += loss_value
        total_samples += batch_size
        global_step = epoch * len(train_loader) + batch_idx
        loop.set_postfix(loss=loss_value, MSE=mse_loss, KLd=kld_loss)
        
        # Log scalar metrics
        writer.add_scalar('Loss', loss_value, global_step)
        writer.add_scalar('MSE Loss', mse_loss, global_step)
        writer.add_scalar('KLD', kld_loss, global_step)
        
    loop.close()
    
    # Log input and output videos every 5 epochs
    if (epoch+1) % (20 if cfg['mode'] == 'grid_search' else 10) == 0:
        # Log model weights
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        for i, var in enumerate(cfg['data']['variables']):
            input_video = data.permute(0,2,1,3,4)[:, :, i, :, :].unsqueeze(2).repeat(1,1,3,1,1)
            reconstruct_video = recon_batch.permute(0,2,1,3,4)[:, :, i, :, :].unsqueeze(2).repeat(1,1,3,1,1)
            writer.add_video(f'{var}_in', input_video, fps=2, global_step=epoch)
            writer.add_video(f'{var}_out',  reconstruct_video, fps=2, global_step=epoch)
        
    avg_train_loss_after_epoch = total_loss / total_samples
    writer.add_scalar('Average loss per sample after epoch', avg_train_loss_after_epoch, epoch)
    logger.info(f'train ===> Epoch: {epoch + 1} Average loss after epoch: {avg_train_loss_after_epoch:.4f}')
    
    return avg_train_loss_after_epoch, total_loss


def train_model(
    model, device, train_loader, val_loader, cfg, epochs, lr, checkpoint_dir, save_checkpoint, load_checkpoint, checkpoint_interval=5
    ) -> torch.nn.Module:
        
    optimizer = select_optimizer(cfg['optimizer']['type'], model, lr)
    scheduler = (define_lr_scheduler(optimizer, cfg['scheduler']['type'], **cfg['scheduler']['params']) 
                if cfg['scheduler']['use_scheduler'] 
                else None)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    start_epoch = 0
    best_val_loss = float('inf')    
    patience = cfg['search_patience']
    patience_counter = 0
    
    # load from last epoch if load_checkpoint is True. Load the last epoch
    if load_checkpoint and os.path.exists(checkpoint_dir):
        checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith(cfg['model']['name'])],
            key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)) if re.search(r'epoch_(\d+)', x) else -1
        )
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            model.train()
            logger.info(f"➡️ Resumed from checkpoint: {latest_checkpoint} at epoch {start_epoch}")        
        else:
            logger.info("❗ No checkpoints found to load")
    
    # add_model_graph_to_tensorboard(model, device, train_loader, writer)   

    for epoch in range(start_epoch, epochs):
        
        # train step
        avg_train_loss_after_epoch, total_loss = train_step(train_loader, model, loss_function, optimizer, device, writer, epoch, epochs, cfg)
        
        # validate step
        avg_val_loss = validate_step(model, device, val_loader, loss_function, writer, epoch, cfg)
        
        # if the mode is grid search, check for early stopping. If the val loss is not improving for 'patience' epochs, stop train
        if cfg['mode'] == 'grid_search':
            if avg_val_loss >= best_val_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"val loss has not improved for {patience} epochs. Stopping train.")
                    # save hyperparameters to a file
                    file_name = f"bs{cfg['batch_size']}_lr{lr}_hd{cfg['model']['hidden_dim']}_ld{cfg['model']['latent_dim']}_killed"
                    with open(os.path.join(checkpoint_dir, f'{file_name}.txt'), 'w') as file:
                        file.write(f"val Loss: {best_val_loss}\n")
                        file.write(f"Epochs: {epoch + 1}\n")
                        file.write(f"Learning Rate: {lr}\n")
                        file.write(f"Batch Size: {cfg['batch_size']}\n")
                        file.write(f"Loss Function: VAE\n")
                        file.write(f"Optimizer: {cfg['optimizer']['type']}\n")
                        file.write(f"Scheduler: {cfg['scheduler']['type']}\n")
                        file.write(f"Scheduler Params: {cfg['scheduler']['params']}\n")
                        file.write(f"Checkpoint Directory: {checkpoint_dir}\n")
                        file.write(f"Best val Loss: {best_val_loss}\n")
                        file.write(f"Avg train Loss: {avg_train_loss_after_epoch}\n")
                        file.write(f"Hidden dim: {cfg['model']['hidden_dim']}\n")
                        file.write(f"Latent dim: {cfg['model']['latent_dim']}\n")
                    
                    break 
            else:
                patience_counter = 0
                
                
        # save model if val loss is the best
        if save_checkpoint and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_best_checkpoint(checkpoint_dir, cfg, model, optimizer, total_loss, best_val_loss, epoch, avg_train_loss_after_epoch, lr, device)
            
        # Step the scheduler
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Learning Rate after epoch {epoch + 1}: {current_lr}')
            writer.add_scalar('learning_rate', current_lr, epoch + 1)
            
        # Save periodic checkpoints
        if save_checkpoint and (epoch + 1) % checkpoint_interval == 0:
            save_periodic_checkpoint(checkpoint_dir, cfg, model, optimizer, epoch, avg_val_loss)
            

          
    if writer:
        writer.close()
    
    if save_checkpoint:
        save_final_model_and_summary(checkpoint_dir, cfg, model, epochs, lr, save_checkpoint, load_checkpoint, device)
        
    if cfg['mode'] == 'grid_search':
        # save the hyperparameters to a file
        file_name = f"bs{cfg['batch_size']}_lr{lr}_hd{cfg['model']['hidden_dim']}_ld{cfg['model']['latent_dim']}"
        with open(os.path.join(checkpoint_dir, f'{file_name}.txt'), 'w') as file:
            file.write(f"Best val Loss: {best_val_loss}\n")
            file.write(f"Best train Loss: {avg_train_loss_after_epoch}\n")
            file.write(f"Learning Rate: {lr}\n")
            file.write(f"Batch Size: {cfg['batch_size']}\n")
            file.write(f"Hidden dim: {cfg['model']['hidden_dim']}\n")
            file.write(f"Latent dim: {cfg['model']['latent_dim']}\n")
            
    return model


def create_run_name(cfg, logger):
    '''
    Create a run name based on the configuration file and the current time.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    logger : logging.Logger
        Logger object.
        
    Returns
    -------
    current_time : str
        Current time in the format 'yymmdd_HHMMSS'.
    log_dir : str
        Log directory path.
    '''    
    # get current time from logger file name
    if isinstance(logger.handlers[1], logging.FileHandler):
        base_filename = logger.handlers[1].baseFilename
        # Extract current time from logger file name
        current_time_match = re.search(r'(\d{6}_\d{6})', base_filename)
        if current_time_match:
            current_time = current_time_match.group(1)
        else:
            logger.error("⚠️ Failed to extract current time from the logger file name.")
            raise ValueError("Failed to extract current time from the logger file name.")
    else:
        logger.error("⚠️ Logger handler is not a FileHandler or does not support 'baseFilename'.")
        raise TypeError("Logger handler is not a FileHandler or does not support 'baseFilename'.")

    # create run name from cfg  
    if cfg['mode'] == 'grid_search':
        run_name = (
            f"{cfg['model']['name']}_c{len(cfg['data']['variables'])}"
            f"_tr{cfg['data']['temporal_resolution']}"
            f"_ss{cfg['data']['spatial_size']}"
            f"_ld{cfg['model']['latent_dim']}"
            f"_hd{cfg['model']['hidden_dim']}"
            f"_lr{cfg['optimizer']['learning_rate']}"
            f"_bs{cfg['batch_size']}"
            f"_e{cfg['train']['num_epochs']}"
            f"_grid_search"
            )  
    else:
        run_name = (
            f"{cfg['model']['name']}_c{len(cfg['data']['variables'])}"
            f"_tr{cfg['data']['temporal_resolution']}"
            f"_ss{cfg['data']['spatial_size']}"
            f"_ld{cfg['model']['latent_dim']}"
            f"_hd{cfg['model']['hidden_dim']}"
            f"_lr{cfg['optimizer']['learning_rate']}"
            f"_bs{cfg['batch_size']}"
            f"_e{cfg['train']['num_epochs']}"
            )    
    log_dir = os.path.join("runs", current_time + "_" + run_name)
    logger.info(f"🔢 TensorBoard log directory: {log_dir}")
    
    return current_time, log_dir


def main(cfg, DEVICE):
    
    logger.info(f"*** train {cfg['model']['name']} model ***")

    # Get current time and create log directory
    global current_time, log_dir
    if cfg["train"]["load_checkpoint"]:
        current_time = cfg["train"]["checkpoint_dir"].split("/")[-2]
        _, log_dir = create_run_name(cfg, logger)
    else:
        current_time, log_dir = create_run_name(cfg, logger)
    
    ModelClass = vae_models.get(cfg['model']['name'])
    if ModelClass is None:
        logger.error("Model {} is not defined in vae_models. Available models: {}".format(cfg['model']['name'], list(vae_models.keys())))
        raise ValueError("Model {} is not defined in vae_models. Available models: {}".format(cfg['model']['name'], list(vae_models.keys())))
    
    # create model  
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

    # if multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    
    train_loader = load_heatwave_samples.create_heatwave_dataloader(
        data_folder=cfg['data']['root_dir'],
        cluster_info_path=cfg['data']['cluster_info_csv'],
        variables=cfg['data']['variables'],
        years=tuple(cfg['data']['train_years']),
        cfg=cfg,
        model_name=cfg['data']['model_name'],
        batch_size=cfg['batch_size'],
        time_window=5,
        image_size=(64, 192),
        shuffle=False,
        )

    val_loader = load_heatwave_samples.create_heatwave_dataloader(
        data_folder=cfg['data']['root_dir'],
        cluster_info_path=cfg['data']['cluster_info_csv'],
        variables=cfg['data']['variables'],
        years=tuple(cfg['data']['val_years']),
        cfg=cfg,
        model_name=cfg['data']['model_name'],
        batch_size=cfg['batch_size'],
        time_window=5,
        image_size=(64, 192),
        shuffle=False,
        )
        
    # print train_loader length
    logger.info("Train Loader Length: {}".format(len(train_loader)))
    for i, (batch, date_labels) in enumerate(train_loader):
        logger.info(f"{i+1}.Batch Shape: {batch.shape}")
        logger.info(f"{i+1}.Date Label Shape: {len(date_labels)}")
        break
    
    # train model
    if cfg['train']['load_checkpoint']:
        checkpoint_dir = cfg['train']['checkpoint_dir']        
    else:
        checkpoint_dir = os.path.join(cfg['train']['checkpoint_dir'], current_time)
    logger.info(f"💾 Checkpoint Directory: {checkpoint_dir}")
    
    trained_model = train_model(model=model,
                                device=DEVICE, 
                                train_loader=train_loader, 
                                val_loader=val_loader,
                                cfg = cfg,
                                epochs=cfg['train']['num_epochs'], 
                                lr=cfg['optimizer']['learning_rate'], 
                                checkpoint_dir=checkpoint_dir,
                                save_checkpoint=cfg['train']['save_checkpoint'], 
                                load_checkpoint=cfg['train']['load_checkpoint'])
    
    return trained_model
