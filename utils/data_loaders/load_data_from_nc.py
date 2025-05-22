import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr


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


class NCVariableDataset(Dataset):
    def __init__(self, root_dir, variables, years, logger, cfg):
               
        self.data = []
        self.spatial_size = cfg['data']['spatial_size']
        
    
        for var in variables:
            var_dir = os.path.join(root_dir, var)
            main_file = os.path.join(var_dir, f"{var}_{years[0]}-{years[1]}_standardized.nc")
            
            if logger is not None:
                logger.info(f"Extracting data for variable: {var}, years: {years}")
            
            if os.path.exists(main_file):
                ds = xr.open_dataset(main_file)
                if var.upper() in ds.variables:
                    era5var = var.upper()
                else:
                    for varname in ds.variables:
                        if var.upper() in varname.upper(): 
                            era5var = varname
                        
                var_data = ds[era5var].values.squeeze()
                ds.close()
                
                var_data, start_day_indices = create_time_patches(var_data, cfg['data']['temporal_resolution'], method=cfg['data']['temporal_method'])
                
                start_height, start_width = calculate_height_width(var_data.shape[2], var_data.shape[3], self.spatial_size)
                var_data = var_data[:, :, start_height:start_height+self.spatial_size, start_width:start_width+self.spatial_size]
                
                if len(self.data) == 0:
                    self.data = np.expand_dims(var_data, axis=1)
                else:
                    self.data = np.concatenate([self.data, np.expand_dims(var_data, axis=1)], axis=1)
            else:
                logger.error(f"File not found: {main_file}")
                raise ValueError(f"File not found: {main_file}")
        
        self.num_years = self.data.shape[0]
        
    def __len__(self):
        return self.num_years

    def __getitem__(self, idx):
        year_data = self.data[idx]
        return torch.tensor(year_data, dtype=torch.float32)


def create_dataloaders_from_nc(root_dir, variables, file_pattern, batch_size, years, logger, cfg, drop_last=True):
    dataset = NCVariableDataset(
        root_dir=root_dir, 
        variables=variables, 
        years=years, 
        logger=logger, 
        cfg=cfg
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last if len(dataset) > batch_size else False)
    return data_loader