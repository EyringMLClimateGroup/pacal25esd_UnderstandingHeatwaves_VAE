import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from skimage.transform import resize
from tqdm import tqdm


logger = None  # default logger is None

def set_logger(ext_logger):
    global logger
    logger = ext_logger

    
def extract_sample(date, time_values, variables, datasets, time_window, image_size):
    if date not in time_values:
        return None
    idx = np.where(time_values == date)[0][0]

    if idx - time_window < 0 or idx + time_window >= len(time_values):
        return None  # not enough context

    start, end = idx - time_window, idx + time_window + 1
    var_samples = []

    for var in variables:
        try:
            data = datasets[var].isel(time=slice(start, end)).values.squeeze()

            if data.ndim == 2:
                data = np.expand_dims(data, axis=0)

            if data.shape[0] != 2 * time_window + 1:
                return None

            if data.shape[1:] != image_size:
                data = np.array([resize(img, image_size, mode="reflect") for img in data])

            var_samples.append(data)

        except Exception as e:
            if logger is not None:
                logger.info(f"Error loading {var} on {date}: {e}") # type: ignore
            else:
                print(f"Error loading {var} on {date}: {e}")
            return None

    try:
        return np.stack(var_samples, axis=0)  # (time, var, H, W)
    except Exception as e:
        if logger is None:
            print(f"Stacking error on {date}: {e}")
        else:
            logger.info(f"Stacking error on {date}: {e}") # type: ignore
        return None


class HeatwaveDataset(Dataset):
    def __init__(self, heatwave_dates, model_name, time_values, variables, datasets, time_window, image_size):
        super(HeatwaveDataset, self).__init__()
        self.samples = []
        self.labels = []
        self.model_name = model_name
        self.variables = variables
        self.time_window = time_window
        self.image_size = image_size
        self.time_values = time_values
        self.datasets = datasets
        self.heatwave_dates = heatwave_dates

        valid_count = 0
        skipped_count = 0

        tqdm_loop = tqdm(heatwave_dates, desc=f"Loading {model_name} samples")
        for d in tqdm_loop:
            s = extract_sample(d, self.time_values, self.variables, self.datasets, self.time_window, self.image_size)
            if s is not None:
                self.samples.append(s)
                self.labels.append(d)
                valid_count += 1
            else:
                skipped_count += 1

            tqdm_loop.set_postfix({
                "Valid": valid_count,
                "Skipped": skipped_count,
                "Date": str(d)
            })

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        if logger is not None:
            logger.info(f"✅ Loaded {len(self.samples)} samples for model {model_name}") # type: ignore
        else:
            print(f"✅ Loaded {len(self.samples)} samples for model {model_name}")
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), str(self.labels[idx])
    
    
def create_heatwave_dataloader(
    data_folder,
    cluster_info_path,
    variables,
    years,
    cfg,
    model_name,
    batch_size,
    time_window,
    image_size,
    shuffle,
    ):
    """
    Create a PyTorch DataLoader with heatwave samples (-N to +N days) from standardized NetCDF files.
    Works with both ERA5 and CMIP6 datasets.

    Parameters:
    ----------
    data_folder : str
        Root folder containing standardized NetCDF files.
    cluster_info_path : str
        CSV path with 'start_date' column listing heatwave start dates.
    variables : list of str
        List of variable names to load.
    years : tuple of (int, int)
        Year range (inclusive) to filter heatwave dates.
    model_name : str
        "ERA5" or any CMIP6 model name.
    batch_size : int
        Number of samples per batch.
    time_window : int
        Days before and after the event (total window = 2*time_window + 1).
    image_size : tuple
        Desired spatial resolution (H, W).
    shuffle : bool
        Whether to shuffle the dataset.

    Returns:
    --------
    torch.utils.data.DataLoader
        PyTorch DataLoader of shape (batch, time, variable, H, W)
    """

    # Load and filter heatwave start dates
    df = pd.read_csv(cluster_info_path)
    heatwave_dates = pd.to_datetime(df["start_date"]).values.astype("datetime64[D]")
    start_year, end_year = years
    heatwave_dates = heatwave_dates[
        (heatwave_dates >= np.datetime64(f"{start_year}-01-01")) &
        (heatwave_dates <= np.datetime64(f"{end_year}-12-31"))
    ]

    # Load variables from NetCDF
    datasets = {}
    for var in variables:
        file_name = cfg['data']['file_pattern'].format(model_name=model_name, var=var)
        file_path = os.path.join(data_folder, var, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        ds = xr.open_dataset(file_path)
        matched_var = next((v for v in ds.variables if isinstance(v, str) and var.lower() in v.lower()), None)
        if matched_var is None:
            raise ValueError(f"Could not find variable {var} in {file_path}")
        datasets[var] = ds[matched_var]

    time_values = datasets[variables[0]]["time"].values.astype("datetime64[D]")
    
    # Create dataloader
    dataset = HeatwaveDataset(
        heatwave_dates,
        model_name,
        time_values,
        variables,
        datasets,
        time_window,
        image_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return loader
