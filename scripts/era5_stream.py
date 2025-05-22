import os
import glob
from joblib import Parallel, delayed
from cdo import Cdo
from tqdm import tqdm
import argparse 


cdo = Cdo()


def get_args():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_JOBS', type=int, default=100, help='Number of parallel jobs to run.')
    parser.add_argument('--START_YEAR', type=int, default=1940, help='Start year of the data.')
    parser.add_argument('--END_YEAR', type=int, default=2022, help='End year of the data.')
    parser.add_argument('--CHUNK_SIZE', type=int, default=5, help='Number of years to process.')
    
    args = parser.parse_args()
    return args


def process_month(var_name: str , input_dir: str, year: int, month: int) -> str:
    """
    Processes a single month of data for a given variable.
    
    Parameters
    ----------
    var_name : str
        Variable name (e.g., "vo" or "d").
    input_dir : str
        Path to the input directory.
    year : int
        Year of the data.
    month : int
        Month of the data.

    Returns
    -------
    str
        Path to the processed NetCDF file.
    """
        
    from cdo import Cdo
    cdo = Cdo()
    
    input_dir = os.path.join(input_dir, f"{VAR_DICT[var_name]}")
    files = sorted(glob.glob(os.path.join(input_dir, f"*{year}-{month:02d}-*.grb")))

    if not files:
        print(f"No files found for {var_name} in {year}-{month:02d}")
        return None

    output_file = os.path.join(OUTPUT_DIR, f"{var_name}_{year}-{month:02d}.nc")
    input_str = f"-sellevel,25000 -cat " + " ".join(files)
    options_str = "-O -f nc -t ecmwf -n ecmwf -b F32 -R"
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping...")
        return output_file
    
    else:
        cdo.daymean(
            input=input_str,
            output=output_file,
            options=options_str
            )
        return output_file


def merge_months_multiyear(var_name, start_year, end_year):
    """
    Merges all monthly files for a multi-year chunk.
    
    Parameters
    ----------
    var_name : str
        Variable name (e.g., "vo" or "d").
    start_year : int
        Start year of the chunk.
    end_year : int
        End year of the chunk.
        
    Returns
    -------
    str
        Path to the merged NetCDF file.
    """
    
    from cdo import Cdo
    cdo = Cdo()
    
    print(f"Merging {var_name} files from {start_year} to {end_year}...")
    
    file_list = []
    for year in range(start_year, end_year + 1):
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{var_name}_{year}-??.nc")))
        file_list.extend(files)
        
    if not file_list:
        print(f"No files found for {var_name} from {start_year} to {end_year}")
        return None
    
    output_file = os.path.join(OUTPUT_DIR, f"{var_name}_{start_year}-{end_year}.nc")
    
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping...")
        return output_file
    else:
        cdo.cat(input=" ".join(file_list), output=output_file, options="-O -f nc -R -t ecmwf")
        # remove intermediate files
        print(f"Removing intermediate files for {var_name} from {start_year} to {end_year}")
        for file in file_list:
            os.remove(file)
    
    return output_file


def merge_all_years(var_name, output_file):
    """
    Merges all processed files for a given variable.
    
    Parameters
    ----------
    var_name : str
        Variable name (e.g., "vo" or "d").
    output_file : str
        Path to the merged output file.

    Returns
    -------
    str
        Path to the merged NetCDF file.
    """
    
    from cdo import Cdo
    cdo = Cdo()
    print(f"Processing {var_name} files...")
    
    chunk_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{var_name}_????-????.nc")))
    if not chunk_files:
        print(f"No processed files found for merging {var_name}.")
        return None

    print(f"Merging {var_name} files...")
    
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping...")
        return output_file
    
    else:
        cdo.cat(input=" ".join(chunk_files), output=output_file)
        
        # remove intermediate files
        for file in chunk_files:
            os.remove(file)

        return output_file


def merge_var_files(*files, output_file):
    """
    Merges multiple files into a single file.
    
    Parameters
    ----------
    files : list of str
        List of input files to merge.
    output_file : str
        Path to the merged output file.
    
    Returns
    -------
    str
        Path to the merged NetCDF file.
    """
        
    from cdo import Cdo
    cdo = Cdo()
    
    print(f"Merging files into, changing variable names, converting to spectral space...")
    cdo.gp2sp(
        input="-chname,VO,svo,D,sd -merge " + " ".join(files), 
        output=output_file)
    
    # # remove intermediate files
    # for file in files:
    #     os.remove(file)
    
    return output_file


def compute_streamfunction(input_file, output_file):
    """
    Computes the streamfunction from the input file.
    
    Parameters
    ----------
    input_file : str
        Path to the input NetCDF file.
    output_file : str
        Path to the output NetCDF file.

    Returns
    -------
    str
        Path to the output NetCDF file.
    """
    
    from cdo import Cdo
    cdo = Cdo()
    
    print("Computing streamfunction...")
    cdo.dv2ps(
        input=input_file,
        output=output_file,
    )

    return output_file


def crop_streamfunction(input_file, output_file):
    """
    Crops the streamfunction to the desired region.
    
    Parameters
    ----------
    input_file : str
        Path to the input NetCDF file.
    output_file : str
        Path to the output NetCDF file.
        
    Returns
    -------
    str
        Path to the output NetCDF file.
    """
    
    from cdo import Cdo
    cdo = Cdo()
    
    print("Cropping streamfunction...")
    input_str = f"-remapbil,{TARGET_GRID} -sp2gp -select,name=stream "
    cdo.sellonlatbox(
        LONLATBOX,
        input=input_str + input_file,
        output=output_file)
    
    return output_file


def main():
    """
    Main function to process ERA5 data.
    """    
    
    # read args
    global VAR_DICT, OUTPUT_DIR, GRIB_DIR, LONLATBOX, TARGET_GRID, START_YEAR, END_YEAR, CHUNK_SIZE
    
    args = get_args()
    N_JOBS = args.N_JOBS
    if N_JOBS == 1:
        print("Running in serial mode.")
    else:
        print(f"Running with {N_JOBS} parallel jobs.")
        
    START_YEAR = args.START_YEAR
    END_YEAR = args.END_YEAR
    CHUNK_SIZE = args.CHUNK_SIZE
        
    VAR_DICT = {
    "vo": 138,
    "d": 155,
    }
        
    GRIB_DIR = "/work/bk1099/data/E5/pl/an/1H"
    OUTPUT_DIR = "/work/bd1083/b309178/HW_detection_VAE/data/deseasonalized/stream250"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Output files
    DAILY_VO = os.path.join(OUTPUT_DIR, "era5_{}-{}_vo_daily.nc".format(START_YEAR, END_YEAR))
    DAILY_D = os.path.join(OUTPUT_DIR, "era5_{}-{}_d_daily.nc".format(START_YEAR, END_YEAR))
    MERGED_SVO_SD = os.path.join(OUTPUT_DIR, "era5_{}-{}_svo_sd.nc".format(START_YEAR, END_YEAR))
    STREAMFUNCTION = os.path.join(OUTPUT_DIR, "era5_{}-{}_STREAM250.nc".format(START_YEAR, END_YEAR))
    CROP_STREAM250 = os.path.join(OUTPUT_DIR, "era5_{}-{}_STREAM250_cropped.nc".format(START_YEAR, END_YEAR))

    # Define region and grid
    LONLATBOX = "-9.84375,39.9375,30.21076,72.3652875"
    TARGET_GRID = "/work/bd1083/b309178/HW_detection_VAE/data/deseasonalized/era5_EU_grid"
    
    # print general information
    print(f"Processing ERA5 data from {START_YEAR} to {END_YEAR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"GRIB directory: {GRIB_DIR}")
    print(f"Target grid: {TARGET_GRID}")
    print(f"Region: {LONLATBOX}")
    print("Variables to process:", VAR_DICT.keys())
    print()
    
    print("Processing monthly files...")
    month_files = Parallel(n_jobs=N_JOBS)(
        delayed(process_month)(var, GRIB_DIR, year, month)
        for var in VAR_DICT
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    )
    
    print("Merging monthly files...")
    merge_months = Parallel(n_jobs=N_JOBS)(
        delayed(merge_months_multiyear)(var, year, year + CHUNK_SIZE - 1)
        for var in VAR_DICT
        for year in range(START_YEAR, END_YEAR + 1, CHUNK_SIZE)
    )

    
    # Merge all processed files
    vo_merged = merge_all_years("vo", DAILY_VO)
    d_merged = merge_all_years("d", DAILY_D)
    
    merge_vars = merge_var_files(DAILY_VO, DAILY_D, output_file=MERGED_SVO_SD)
    
    stream_function = compute_streamfunction(MERGED_SVO_SD, STREAMFUNCTION)
    
    cropped_streeam250 = crop_streamfunction(STREAMFUNCTION, CROP_STREAM250)

    print("Done!")
    
if __name__ == "__main__":
    main()
