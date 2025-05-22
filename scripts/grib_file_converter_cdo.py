import os
import re
import sys
import numpy as np
import pandas as pd
from cdo import Cdo
import intake
import glob

sys.path.append('/work/bd1083/b309178/2paper_repo/')
from utils import read_config, get_args, start_logger
from tqdm.notebook import tqdm
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
import time
import gc
from ast import literal_eval


def read_dkrz_catalog():
    """
    Read the dkrz catalog to get the metadata of the ERA5 dataset
    
    Returns:
    --------
    col: intake_esm.collection
        The collection of the ERA5 dataset
    """
    
    era5 = intake.open_esm_datastore("/work/ik1017/Catalogs/dkrz_era5_disk.json")
    era5_df = era5.df
    return era5_df


def dynamic_filter_era5_data(df, cfg):
    df.columns = df.columns.str.lower()
    variable_config = cfg.get("VARIABLE_ID", {})
    era_id = cfg.get("ERA_ID", None)

    filter_list = []
    for var_info in variable_config.values():
        condition = (
            (df["short_name"] == var_info.get("short_name")) &
            (df["frequency"] == var_info.get("frequency")) &
            (df["steptype"] == var_info.get("stepType")) &
            (df["datatype"] == var_info.get("dataType")) &
            (df["level_type"] == var_info.get("level_type"))
        )
        if era_id and "era_id" in df.columns:
            condition &= (df["era_id"] == era_id)
        filter_list.append(condition)

    if filter_list:
        combined_filter = filter_list[0]
        for cond in filter_list[1:]:
            combined_filter |= cond
        filtered_df = df[combined_filter].copy()
    else:
        filtered_df = pd.DataFrame()

    return filtered_df


def convert_to_monthly(var_name, year, month, file_paths, output_dir, cfg):
    from cdo import Cdo

    cdo = Cdo()

    # Get files matching year and month
    monthly_files = [
        f for f in file_paths
        if re.search(rf"{year:04d}-{month:02d}-\d{{2}}", f)
    ]

    if not monthly_files:
        logger.info(f"⚠️ No files found for {var_name} {year}-{month:02d}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Output path
    output_file = os.path.join(output_dir, f"{var_name}_{year:04d}-{month:02d}_daily.nc")
    if os.path.exists(output_file):
        logger.info(f"✅ Monthly file already exists: {output_file}")
        return output_file

    # Determine daily reduction method (mean or max)
    method = cfg["VARIABLE_ID"][var_name].get("cdo_method", "daymean")
    cdo_method = getattr(cdo, method)

    # Land-sea mask for remapping
    remap_mask = cfg["ERA5_LSM_NA"]

    # Step: remap → cat → daily agg (mean or max) → save
    try:
        level_type = cfg["VARIABLE_ID"][var_name].get("level_type", "surface")
        if level_type == "surface":
            input_string = f"-remapbil,{remap_mask} -cat {' '.join(sorted(monthly_files))}"
        else:
            var_level = cfg["VARIABLE_ID"][var_name].get("level_height", "50000")
            input_string = f"-remapbil,{remap_mask} -sellevel,{var_level} -cat {' '.join(sorted(monthly_files))}"
        
        cdo_command = cdo_method(
            input=input_string,
            output=output_file,
            options="-O -f nc -R -t ecmwf"
        )
        logger.info(f"✅ Created {output_file}")
    except Exception as e:
        logger.info(f"❌ Failed to create monthly file for {var_name} {year}-{month:02d}: {e}")
        return None

    return output_file


def calculate_streamfunction(var_name, year, month, file_paths, output_dir, cfg):
    




def merge_monthly_to_chunk(var_name, chunk_years, monthly_dir, output_dir):
    import os
    import glob
    from cdo import Cdo

    cdo = Cdo()
    os.makedirs(output_dir, exist_ok=True)

    # Match only monthly daily files (e.g., tas_2010-06_daily.nc)
    pattern = os.path.join(monthly_dir, f"{var_name}_????-??_daily.nc")
    all_files = sorted(glob.glob(pattern))

    # Filter files that match the years in the chunk
    chunk_files = [
        f for f in all_files
        if any(f"{y:04d}-" in os.path.basename(f) for y in chunk_years)
    ]

    output_file = os.path.join(output_dir, f"{var_name}_{chunk_years[0]}-{chunk_years[-1]}_chunk.nc")

    if not os.path.exists(output_file):
        cdo.cat(input=" ".join(chunk_files), output=output_file, options="-f nc -R -t ecmwf")
        logger.info(f"✅ Created chunk: {output_file}")
    else:
        logger.info(f"✅ Chunk already exists: {output_file}")

    return output_file


def standardize_deseason(var_name, input_file, output_dir, window=15):
    from cdo import Cdo
    cdo = Cdo()

    # Smooth seasonal cycle (mean and std)
    climatology_start = cfg["data"]["climatology_start"]
    climatology_end = cfg["data"]["climatology_end"]
    ydrunmean_file = os.path.join(output_dir, f"{var_name}_{cfg["data"]["climatology_start"]}-{cfg["data"]["climatology_end"]}_15ydrunmean.nc")
    ydrunstd_file = os.path.join(output_dir, f"{var_name}_{cfg["data"]["climatology_start"]}-{cfg["data"]["climatology_end"]}_15ydrunstd.nc")
    deseasonalized_file = os.path.join(output_dir, f"{var_name}_{cfg["data"]["start_year"]}-{cfg["data"]["end_year"]}_deseasonalized.nc")
    standardized_file = os.path.join(output_dir, f"{var_name}_{cfg["data"]["start_year"]}-{cfg["data"]["end_year"]}_standardized.nc")

    # Daily mean (remove inter-annual variation)
    cdo.ydrunmean(window, input=f"-del29feb -selyear,{cfg["data"]["climatology_start"]}/{cfg["data"]["climatology_end"]} " + input_file, output=ydrunmean_file)
    cdo.ydrunstd(window, input=f"-del29feb -selyear,{cfg["data"]["climatology_start"]}/{cfg["data"]["climatology_end"]} " + input_file, output=ydrunstd_file)
    
    # Deseasonalize
    cdo.sub(input=f"{input_file} {ydrunmean_file}", output=deseasonalized_file)

    # Standardize
    cdo.div(input=f"{deseasonalized_file} {ydrunstd_file}", output=standardized_file)

    logger.info(f"✅ Standardized data saved to {standardized_file}")


def process_variable(var_name, var_df):
    from cdo import Cdo
    cdo = Cdo()

    file_paths = var_df["path"].tolist()
    monthly_dir = os.path.join(cfg["OUTPUT_PATH"], var_name, "monthly")
    chunk_dir = os.path.join(cfg["OUTPUT_PATH"], var_name, "chunks")
    final_dir = os.path.join(cfg["OUTPUT_PATH"], var_name)

    os.makedirs(monthly_dir, exist_ok=True)
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # 1. Extract years
    years = sorted(set(int(re.search(r"\d{4}", os.path.basename(f)).group()) for f in file_paths))
    years = [y for y in years if cfg["data"]["start_year"] <= y <= cfg["data"]["end_year"]]
    logger.info(f"Processing {var_name} ({len(years)} years)")
    
    # 2. Convert to monthly (with daily timesteps)
    logger.info(f"📆 Converting {var_name} hourly → monthly (daily)")
    Parallel(n_jobs=cfg["number_jobs"], verbose=10)(
        delayed(convert_to_monthly)(var_name, y, m, file_paths, monthly_dir, cfg)
        for y in years for m in range(1, 13)
    )
    
    # 3. Merge into 5-year chunks
    logger.info(f"🔄 Merging {var_name} monthly files into 5-year chunks")
    chunks = [years[i:i + 5] for i in range(0, len(years), 5)]
    Parallel(n_jobs=len(chunks), verbose=10)(
        delayed(merge_monthly_to_chunk)(var_name, chunk, monthly_dir, chunk_dir)
        for chunk in chunks
    )
    
    # 4. Merge all chunks
    full_file = os.path.join(final_dir, f"{var_name}_{cfg['data']['start_year']}-{cfg['data']['end_year']}.nc")
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, f"{var_name}_*_chunk.nc")))
    if not os.path.exists(full_file):
        logger.info(f"🧩 Merging all chunks into full period for {var_name}")
        cdo.del29feb(input="-cat " + " ".join(sorted(chunk_files)), output=full_file)
    else:
        logger.info(f"✅ Full merged file already exists: {full_file}")

    # 5. Deseasonalize and standardize
    logger.info(f"📉 Deseasonalizing and standardizing {var_name}")
    standardize_deseason(var_name, full_file, final_dir)

    logger.info(f"✅ Processed {var_name}")
    
    # remove chunk files directory and its contents
    for f in chunk_files:
        os.remove(f)       
    
    gc.collect()


def main():
    global logger, cfg
    args = get_args()
    cfg = read_config(args.config)
    logger = start_logger("era5_merge", f"log_era5_merge_{time.strftime('%Y%m%d-%H%M%S')}.log")

    start_time = time.time()

    metadata_path = os.path.join(cfg['OUTPUT_PATH'], cfg['metadata_filename'])

    if os.path.exists(metadata_path) and not cfg['force_create_metadata']:
        logger.info(f"📃 Using existing metadata file at {metadata_path}")
        era5_df = pd.read_csv(metadata_path)
    else:
        logger.info("📃 Creating metadata file...")
        era5_df = dynamic_filter_era5_data(read_dkrz_catalog(), cfg)
        era5_df.to_csv(metadata_path, index=False)
           
    with parallel_backend("loky", n_jobs=len(cfg["VARIABLE_ID"])):
        Parallel(verbose=10)(
            delayed(process_variable)(var, era5_df[era5_df.short_name == cfg["VARIABLE_ID"][var]["short_name"]])
            for var in cfg["VARIABLE_ID"]
        )


if __name__ == "__main__":
    main()
