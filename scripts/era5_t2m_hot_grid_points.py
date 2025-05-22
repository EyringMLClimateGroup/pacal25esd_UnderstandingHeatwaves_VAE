import os
from cdo import Cdo
import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from collections import deque

from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import Parallel, delayed


import numpy as np
from collections import deque
from joblib import Parallel, delayed

def compute_cluster_extent(cluster, labels, latitudes, longitudes):
    t_indices, lon_indices, lat_indices = np.where(labels == cluster)
    heatwave_length = np.max(t_indices) - np.min(t_indices) + 1

    if heatwave_length < 3:
        return None  # Ignore short events

    min_lat_idx = np.clip(np.min(lat_indices), 0, len(latitudes) - 1)
    max_lat_idx = np.clip(np.max(lat_indices), 0, len(latitudes) - 1)
    min_lon_idx = np.clip(np.min(lon_indices), 0, len(longitudes) - 1)
    max_lon_idx = np.clip(np.max(lon_indices), 0, len(longitudes) - 1)

    extent = {
        "time_range": (int(np.min(t_indices)), int(np.max(t_indices))),
        "lat_range": (float(latitudes[min_lat_idx]), float(latitudes[max_lat_idx])),
        "lon_range": (float(longitudes[min_lon_idx]), float(longitudes[max_lon_idx])),
        "heatwave_length": int(heatwave_length)
    }

    return cluster, {"extent": extent}


class GDBSCAN:
    def __init__(self, grid, latitudes, longitudes, min_card=21, connectivity=8, time_window=1):
        self.grid = grid
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.min_card = min_card
        self.connectivity = connectivity
        self.time_window = time_window
        self.NOISE = -1
        self.UNCLASSIFIED = 0
        self.cluster_id = 1
        self.labels = np.zeros_like(grid, dtype=int)
    
    def get_neighbors(self, t, lon, lat):
        time_steps, lons, lats = self.grid.shape
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        if self.connectivity == 8:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dt in range(-self.time_window, self.time_window + 1):
            for dlon, dlat in directions:
                nt, nlon, nlat = t + dt, lon + dlon, lat + dlat
                if (
                    0 <= nt < time_steps and
                    0 <= nlon < lons and
                    0 <= nlat < lats and
                    self.grid[nt, nlon, nlat] == 1
                ):
                    neighbors.append((nt, nlon, nlat))
        
        return neighbors
    
    def latitude_weighted_min_card(self, lat_idx, total_lat_points):
        lat_index = int((lat_idx / total_lat_points) * (len(self.latitudes) - 1))
        return np.round(self.min_card * np.cos(np.radians(self.latitudes[lat_index])))
    
    def expand_cluster(self, t, lon, lat):
        neighbors = self.get_neighbors(t, lon, lat)
        local_min_card = self.latitude_weighted_min_card(lat, self.grid.shape[2])
        
        if len(neighbors) + 1 < local_min_card:
            self.labels[t, lon, lat] = self.NOISE
            return False
        
        cluster_id = self.cluster_id
        queue = deque([(t, lon, lat)])
        self.labels[t, lon, lat] = cluster_id
        
        while queue:
            ct, clon, clat = queue.popleft()
            for nt, nlon, nlat in self.get_neighbors(ct, clon, clat):
                if self.labels[nt, nlon, nlat] == self.UNCLASSIFIED:
                    self.labels[nt, nlon, nlat] = cluster_id
                    queue.append((nt, nlon, nlat))
        
        return True

    def get_cluster_info(self, n_jobs=4):
        unique_clusters = np.unique(self.labels[self.labels > 0])

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_cluster_extent)(cluster, self.labels, self.latitudes, self.longitudes)
            for cluster in unique_clusters
        )

        cluster_info = {cid: info for res in results if res is not None for cid, info in [res]}
        return cluster_info

    def get_cluster_info_single(self):
        # Optional non-parallel version (e.g., for testing)
        cluster_info = {}
        unique_clusters = np.unique(self.labels[self.labels > 0])

        for cluster in unique_clusters:
            t_indices, lon_indices, lat_indices = np.where(self.labels == cluster)
            heatwave_length = np.max(t_indices) - np.min(t_indices) + 1

            if heatwave_length < 3:
                continue

            min_lat_idx = np.clip(np.min(lat_indices), 0, len(self.latitudes) - 1)
            max_lat_idx = np.clip(np.max(lat_indices), 0, len(self.latitudes) - 1)
            min_lon_idx = np.clip(np.min(lon_indices), 0, len(self.longitudes) - 1)
            max_lon_idx = np.clip(np.max(lon_indices), 0, len(self.longitudes) - 1)

            extent = {
                "time_range": (np.min(t_indices), np.max(t_indices)),
                "lat_range": (self.latitudes[min_lat_idx], self.latitudes[max_lat_idx]),
                "lon_range": (self.longitudes[min_lon_idx], self.longitudes[max_lon_idx]),
                "heatwave_length": heatwave_length
            }
            cluster_info[cluster] = {"extent": extent}

        return cluster_info

    def run(self, use_parallel_info=True, n_jobs=100):
        time_steps, lons, lats = self.grid.shape

        for t in range(time_steps):
            for lon in range(lons):
                for lat in range(lats):
                    if self.grid[t, lon, lat] == 1 and self.labels[t, lon, lat] == self.UNCLASSIFIED:
                        if self.expand_cluster(t, lon, lat):
                            self.cluster_id += 1

        if use_parallel_info:
            return self.labels, self.get_cluster_info(n_jobs=n_jobs)
        else:
            return self.labels, self.get_cluster_info_single()



def plot_yearly_heatwaves(unique_clusters, output_dir):
    # Group heatwave counts by year
    heatwave_counts = unique_clusters.groupby("year").size()

    # Fit a linear trendline
    years = heatwave_counts.index.astype(int)  # Convert years to integers if needed
    counts = heatwave_counts.values
    trend = np.polyfit(years, counts, 1)  # Fit a 1st-degree polynomial (linear)
    trendline = np.poly1d(trend)

    # Plot yearly heatwave count
    plt.figure(figsize=(12, 6))
    plt.plot(years, counts, marker="o", linestyle="-", label="Heatwave Count")

    # Add trendline
    plt.plot(years, trendline(years), linestyle="--", color="red", label="Trendline")

    # put ticks at every 5 years
    plt.xticks(np.arange(years.min(), years.max() + 1, 5))
    plt.grid(True)
    # Labels and title
    plt.xlabel("Year")
    plt.ylabel("Number of Heatwaves")
    plt.title("Yearly Heatwave Count with Trendline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "yearly_heatwave_count.png"))

def calculate_spatial_extent(lon_range, lat_range):
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    
    lat_extent = (lat_max - lat_min) * 111.32  # Latitude is mostly uniform
    lat_mean = (lat_max + lat_min) / 2  # Midpoint latitude for longitude correction
    
    # Adjust longitude using cos(latitude)
    lon_extent = (lon_max - lon_min) * 111.32 * np.cos(np.radians(lat_mean))
    
    return lon_extent * lat_extent 


def calculate_heatwave_clusters(file_path, output_dir):
    ds = xr.open_dataset(file_path)

    hot_points = ds["T2M"].values
    latitudes = ds["lat"].values
    longitudes = ds["lon"].values
    time_step = ds["time"].values
    
    print("✅ Running GDBSCAN...")
    gdbscan = GDBSCAN(hot_points, latitudes, longitudes, min_card=21, connectivity=8, time_window=1)
    labels, cluster_info = gdbscan.run()
    
    # find dates for the clusters
    cluster_dates = {}
    for cluster_id, cluster in cluster_info.items():
        time_range = cluster["extent"]["time_range"]
        start_date = time_step[time_range[0]].strftime("%Y-%m-%d")
        end_date = time_step[time_range[1]].strftime("%Y-%m-%d")
        cluster_id = int(cluster_id)
        cluster_dates[cluster_id] = (start_date, end_date)
        
    # Organize the cluster information into a DataFrame
    cluster_info_df = pd.DataFrame(cluster_info).T
    cluster_info_df["cluster_id"] = cluster_info_df.index
    cluster_info_df["start_date"] = cluster_info_df["cluster_id"].apply(lambda x: cluster_dates[x][0])
    cluster_info_df["end_date"] = cluster_info_df["cluster_id"].apply(lambda x: cluster_dates[x][1])
    cluster_info_df["heatwave_start"] = cluster_info_df["extent"].apply(lambda x: x["time_range"][0])
    cluster_info_df["heatwave_end"] = cluster_info_df["extent"].apply(lambda x: x["time_range"][1])
    cluster_info_df["time_range"] = cluster_info_df["extent"].apply(lambda x: x["time_range"])
    cluster_info_df["lat_range"] = cluster_info_df["extent"].apply(lambda x: x["lat_range"])
    cluster_info_df["lon_range"] = cluster_info_df["extent"].apply(lambda x: x["lon_range"])
    cluster_info_df["heatwave_length"] = cluster_info_df["extent"].apply(lambda x: x["heatwave_length"])
    cluster_info_df.to_csv(os.path.join(output_dir, "hot_grid_clusters.csv"), index=False)

        
    # select clusters with unique start date
    unique_clusters = cluster_info_df.drop_duplicates(subset="start_date")
    unique_clusters = unique_clusters.sort_values(by="start_date").reset_index(drop=True)
    unique_clusters['month'] = pd.to_datetime(unique_clusters['start_date']).dt.month
    unique_clusters['year'] = pd.to_datetime(unique_clusters['start_date']).dt.year
    unique_clusters["spatial_extent"] = unique_clusters["lon_range"].combine(
        unique_clusters["lat_range"], calculate_spatial_extent).round(0)
    unique_clusters.to_csv(os.path.join(output_dir, "unique_hot_grid_clusters.csv"), index=False)
    
    big_clusters = unique_clusters[unique_clusters["spatial_extent"] > 40000].reset_index(drop=True)
    big_clusters.to_csv(os.path.join(output_dir, "big_hot_grid_clusters.csv"), index=False)

    # Plot yearly heatwaves
    plot_yearly_heatwaves(unique_clusters, output_dir)


def main():
    # === CONFIG ===
    INPUT_FILE = "/work/bd1083/b309178/HW_detection_VAE/data/hourly_era5_NA/t2m/t2m_1940-2022.nc"
    OUTPUT_DIR = "/work/bd1083/b309178/HW_detection_VAE/data/hourly_era5_NA/t2m/hot_grids/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    YDRUN_MIN_FILE = os.path.join(OUTPUT_DIR, "t2m_1940-2022_ydrunmin.nc")
    YDRUN_MAX_FILE = os.path.join(OUTPUT_DIR, "t2m_1940-2022_ydrunmax.nc")
    T2M_YDRUNPCTL_15DAY_90 = os.path.join(OUTPUT_DIR, "t2m_1940-2022_ydrunpctl90_15day.nc")
    MERGED_HOTGRIDS = os.path.join(OUTPUT_DIR, "t2m_1940-2022_hot_grids.nc")
    SUMMER_HOTGRIDS = os.path.join(OUTPUT_DIR, "t2m_1940-2022_hot_grids_jja.nc")
    ERA5_LAND = "/work/bd1083/b309178/HW_detection_VAE/data/era5_hw_vae/lsm/era5_lsm_NA.nc"
    CROPPED_WE = os.path.join(OUTPUT_DIR, "t2m_1940-2022_hot_grids_land_WE.nc")
    
    
    cdo = Cdo()

    # print("✅ Computing 15-day running min...")
    # cdo.ydrunmin("15", input=INPUT_FILE, output=YDRUN_MIN_FILE)

    # print("✅ Computing 15-day running max...")
    # cdo.ydrunmax("15", input=INPUT_FILE, output=YDRUN_MAX_FILE)

    # print("✅ Computing 90th percentile with 15-day window...")
    # cdo.ydrunpctl("90,15", input=f"{INPUT_FILE} {YDRUN_MIN_FILE} {YDRUN_MAX_FILE}", output=T2M_YDRUNPCTL_15DAY_90)

    # print("✅ Identifying hot grid cells (T2M > 90th percentile)...")
    # cdo.gtc("0", input=f"-ydaysub {INPUT_FILE} {T2M_YDRUNPCTL_15DAY_90}", output=MERGED_HOTGRIDS)

    # print("✅ Extracting summer months (JJA)...")
    # cdo.selmon("6/8", input=MERGED_HOTGRIDS, output=SUMMER_HOTGRIDS)
    
    # print("✅ Masking out land points...")
    # cdo.div(input=f"{MERGED_HOTGRIDS} {ERA5_LAND}", output=MERGED_HOTGRIDS.replace(".nc", "_land.nc"))
    # cdo.div(input=f"{SUMMER_HOTGRIDS} {ERA5_LAND}", output=SUMMER_HOTGRIDS.replace(".nc", "_land.nc"))
    
    print("✅ Cropping to ERA5 western Europe grid...")    
    cdo.sellonlatbox("-10,15,35,55", input=MERGED_HOTGRIDS.replace(".nc", "_land.nc"), output=CROPPED_WE)
    
    print("✅ Calculating heatwave clusters...")
    calculate_heatwave_clusters(CROPPED_WE, OUTPUT_DIR)

if __name__ == "__main__":
    main()