import os
import torch
import numpy as np
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader


def tabulate_dataloader(dataset, dataloader, drop_last=True):
    """
    Print a tabulated representation of the data loader.
    """
    column_titles = ["Item", "Value"]
    data = [[r"# Samples", len(dataset)], 
            ["Sample Shape (C, D, H, W)",   dataset[0].shape],
            ["Batch Size", dataloader.batch_size],
            [r"# Batches", len(dataloader)],
            ["Last Batch {}".format("Dropped" if drop_last else "Included"), len(dataloader.dataset) % dataloader.batch_size]]
             
    print(tabulate(data, headers=column_titles, tablefmt="fancy_grid"))    
    

def calculate_height_width(height, width, spatial_size) -> tuple:
    """ 
    Crop a centered spatial region of size spatial_size x spatial_size from the input height x width image.
    """
    if spatial_size > height or spatial_size > width:
        raise ValueError("Spatial size {} should be smaller than the input image size (height: {}, width: {})".format(spatial_size, height, width))

    else:
        start_height = (height - spatial_size) // 2
        start_width = (width - spatial_size) // 2
    
    return start_height, start_width


def create_time_patches(data, temporal_resolution, method='fixed', padding_value=np.nan) -> tuple:
    """
    Create time patches from the data using fixed steps or sliding window without losing data.
    
    Parameters
    ----------
    data : np.ndarray
        The data to create time patches from [days, lat, lon].
    temporal_resolution : int
        The temporal resolution of the time patches.
    method : str, optional
        The method to create patches, either 'fixed' (default) or 'sliding'.
        - 'fixed': Creates patches with non-overlapping windows of temporal_resolution.
        - 'sliding': Creates patches using a sliding window with step size of 1.
    padding_value : float or np.nan, optional
        Value to use for padding if the data length is not divisible by temporal_resolution.
    
    Returns
    -------
    np.ndarray
        The time patches.
    np.ndarray
        Array of starting day indices for each patch (for mapping back to original data).
    """
    if method not in ['fixed', 'sliding']:
        raise ValueError("Invalid method. Choose 'fixed' or 'sliding'.")
    
    # Calculate padding if needed
    excess_days = len(data) % temporal_resolution
    if excess_days > 0:
        padding_needed = temporal_resolution - excess_days
        padding_shape = (padding_needed,) + data.shape[1:]
        padding_array = np.full(padding_shape, padding_value)
        data = np.concatenate((data, padding_array), axis=0)
    
    # Create time patches and track starting day indices
    patches = []
    start_day_indices = []

    if method == 'fixed':
        for i in range(0, data.shape[0], temporal_resolution):
            patches.append(data[i:i + temporal_resolution])
            start_day_indices.append(i)  # Record the start index of each patch
    elif method == 'sliding':
        for i in range(0, data.shape[0] - temporal_resolution + 1):
            patches.append(data[i:i + temporal_resolution])
            start_day_indices.append(i)
    
    return np.array(patches), np.array(start_day_indices)


class NPYVariableDataset(Dataset):
    def __init__(self, root_dir, variables, years, logger, cfg, file_pattern="{variable}_{start_year}_{end_year}.npy"):
        self.data = []
        self.start_year, self.end_year = years
        self.spatial_size = cfg['data']['spatial_size']
        
                
        for var in variables:
            if logger is not None:
                logger.info(f"Loading data for variable: {var}")
            file_path = os.path.join(
                root_dir, 
                file_pattern.format(
                    variable=var, 
                    start_year=self.start_year,
                    end_year=self.end_year,
                    resample_hour=cfg['data']['resample_hour'],
                    resample_method=cfg['data']['resample_method'][var],
                    scaler=cfg['data']['scaler']
                    )
                )
            
            if os.path.exists(file_path):
                var_data = np.load(file_path).squeeze() # shapes = (timestep, lon, lat)
                var_data, start_day_indices = create_time_patches(var_data, cfg['data']['temporal_resolution'], method=cfg['data']['temporal_method'])
                
                # Crop a centered spatial region of size spatial_size x spatial_size
                start_height, start_width = calculate_height_width(var_data.shape[2], var_data.shape[3], self.spatial_size)
                var_data = var_data[:, :, start_height:start_height+self.spatial_size, start_width:start_width+self.spatial_size]                
            
                if len(self.data) == 0:
                    if logger is not None:
                        logger.info(f"First variable added: {var}")
 
                    self.data = np.expand_dims(var_data, axis=1)
                else:
                    if logger is not None:
                        logger.info(f"Concatenating variable: {var}")
                    self.data = np.concatenate([self.data, np.expand_dims(var_data, axis=1)], axis=1)
                    
            else:
                logger.error(f"File not found: {file_path}")
                raise ValueError(f"File not found: {file_path}")
        
        self.data = np.array(self.data)  # Ensure self.data is a NumPy array
        self.num_years = self.data.shape[0]
        

    def __len__(self):
        return self.num_years


    def __getitem__(self, idx):
        year_data = self.data[idx]
        return torch.tensor(year_data, dtype=torch.float32)





def create_dataloaders_from_npy(root_dir, variables, file_pattern, batch_size, years, logger, cfg, drop_last=True):
    dataset = NPYVariableDataset(
        root_dir=root_dir, 
        variables=variables, 
        years=years, 
        logger=logger, 
        cfg=cfg, 
        file_pattern=file_pattern,
        )
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last if len(dataset) > batch_size else False)

    if logger is not None:
        tabulate_dataloader(dataset, data_loader, drop_last=drop_last if len(dataset) > batch_size else False)

    return data_loader