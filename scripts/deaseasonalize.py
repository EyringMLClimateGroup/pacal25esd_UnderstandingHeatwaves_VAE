import os
import glob
from joblib import Parallel, delayed
from cdo import Cdo
from tqdm import tqdm
import argparse


def apply_land_mask(var_name, output_folder, mask_file):
    """ Apply land-sea mask to all files in var folder."""
    from cdo import Cdo
    cdo = Cdo()
    
    var_folder = os.path.join(output_folder, var_name)
    files = sorted(glob.glob(os.path.join(var_folder, "*.nc")))
    output_folder = os.path.join(output_folder, var_name)
    os.makedirs(output_folder, exist_ok=True)
    
    for file in files:
        # append "_land" to the file name
        output_file = os.path.join(output_folder, os.path.basename(file).replace(".nc", "_land.nc"))
        if os.path.exists(output_file):
            print(f"{output_file} already exists. Skipping...")
        else:
            print(f"Applying land mask to {file}...")
            cdo.div(input=f"{file} {mask_file}", output=output_file)
    
    
    


def deseasonalize(var_name, output_folder, merged_file, climatology_file, start_year, end_year):
    """Remove the seasonal cycle by subtracting the climatology."""
    from cdo import Cdo
    cdo = Cdo()
    
    # get years from the merged file
    var_folder = os.path.join(output_folder, var_name)

    deseasonalized_file = os.path.join(var_folder, f"{var_name}_{start_year}-{end_year}_deseasonalized.nc")
    standardize_file = os.path.join(var_folder, f"{var_name}_{start_year}-{end_year}_standardized.nc")
    
    if os.path.exists(deseasonalized_file) and os.path.exists(standardize_file):
        print(f"{deseasonalized_file} and {standardize_file} already exist. Skipping...")
        return {var_name: {"deseasonalized": deseasonalized_file, "standardized": standardize_file}}
    else:
        cdo.ydaysub(input=f"-selyear,{start_year}/{end_year} {merged_file} {climatology_file["mean"]}", output=deseasonalized_file)
        cdo.ydaydiv(input=f"{deseasonalized_file} {climatology_file["std"]}", output=standardize_file)
        return deseasonalized_file


def compute_climatology(var_name, merged_file, output_folder, start_year, end_year):
    from cdo import Cdo
    cdo = Cdo()
    
    var_folder = os.path.join(output_folder, var_name)

    mean_output_file = os.path.join(var_folder, f"{var_name}_{start_year}-{end_year}_ydrunmean15.nc")
    std_output_file = os.path.join(var_folder, f"{var_name}_{start_year}-{end_year}_ydrunstd15.nc")
    
    if os.path.exists(mean_output_file) and os.path.exists(std_output_file):
        print(f"{mean_output_file} and {std_output_file} already exist. Skipping...")
        return {var_name: {"mean": mean_output_file, "std": std_output_file}}
    else:
        print(f"Computing climatology for {var_name} from {start_year} to {end_year}...")
        cdo.ydrunmean(
            "15",
            input=f"-selyear,{start_year}/{end_year} {merged_file}",
            output=mean_output_file
        )
        cdo.ydrunstd(
            "15",
            input=f"-selyear,{start_year}/{end_year} {merged_file}",
            output=std_output_file
        )
        return {var_name: {"mean": mean_output_file, "std": std_output_file}}    


def merge_variable(var_name, output_folder, start_year, end_year):
    from cdo import Cdo
    cdo = Cdo()
    
    var_folder = os.path.join(output_folder, var_name)
    files = sorted(glob.glob(os.path.join(var_folder, "????.nc")))
    output_file = os.path.join(var_folder, f"{var_name}_{start_year}-{end_year}.nc")
    
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping...")
        return {var_name: output_file}
    else:  
        print(f"Merging {var_name} from {start_year} to {end_year}...")
        cdo.cat(
            input=" ".join(files),
            output=output_file
        )        
        for i in files:
            os.remove(i)
        return {var_name: output_file}
    
    
def process_year(var_name, input_folder, output_folder, year):
    
    from cdo import Cdo
    cdo = Cdo()

    var_folder = os.path.join(input_folder, var_name)
    files = sorted(glob.glob(os.path.join(var_folder, f"{year}*.nc")))
    output_folder = os.path.join(output_folder, var_name)
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, f"{year}.nc")
    try:  
        if os.path.exists(output_file):
            print(f"{output_file} already exists. Skipping...")
            return output_file
        else:
            print(f"Processing {var_name} for {year}...")
            input_string = f"-{VARIABLES[var_name]} -cat " + " ".join(files)
            cdo.del29feb(
                input=input_string,
                output=output_file
            )        
            return output_file
    except Exception as e:
        print(f"Error processing {var_name} for {year}")
        return None


def main():
    """Main function to process all variables in parallel."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_JOBS', type=int, default=200, help='Number of parallel jobs.')
    args = parser.parse_args()
    
    global INPUT_FOLDER, OUTPUT_FOLDER, VARIABLES, CLIMATOLOGY_START, CLIMATOLOGY_END, DATA_START, DATA_END
    # Define input and output directories
    INPUT_FOLDER = "/work/bd1083/b309178/HW_detection_VAE/data/hourly_era5_EU"
    OUTPUT_FOLDER = "/work/bd1083/b309178/HW_detection_VAE/data/deseasonalized"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    VARIABLES = {
        "t2m": "daymax",
        "d2m": "daymax",
        "q": "daymean",
        "r": "daymean",
        "sp": "daymean",
        "tcc": "daymean",
        "u10": "daymean",
        "v10": "daymean",
        "ssr": "daysum",
        "tp": "daysum"
    }
    
    # Climatology years
    CLIMATOLOGY_START = 1941
    CLIMATOLOGY_END = 1980

    DATA_START = 1940
    DATA_END = 2022
    
    MASK_FILE = "/work/bd1083/b309178/HW_detection_VAE/data/deseasonalized/era5_land.nc" 
    
    
    print(f"Processing {VARIABLES} from {DATA_START} to {DATA_END}...")
    year_files = Parallel(n_jobs=args.N_JOBS)(
        delayed(process_year)(var_name, INPUT_FOLDER, OUTPUT_FOLDER, start_year) 
        for var_name in VARIABLES.keys()
        for start_year in range(DATA_START, DATA_END+1)
    )
    
    print("Merging files...")
    merge_variable_year = Parallel(n_jobs=len(VARIABLES))(
        delayed(merge_variable)(var_name, OUTPUT_FOLDER, DATA_START, DATA_END)
        for var_name in VARIABLES.keys()
    )
    merge_variable_year = {k: v for d in merge_variable_year for k, v in d.items()}
    
    print("Computing climatology...")
    var_climatology = Parallel(n_jobs=len(VARIABLES))(
        delayed(compute_climatology)(var_name, merge_variable_year[var_name], OUTPUT_FOLDER, CLIMATOLOGY_START, CLIMATOLOGY_END)
        for var_name in VARIABLES.keys()
    )
    var_climatology = {k: v for d in var_climatology for k, v in d.items()}
    
    print("Deseasonalizing...")
    deseasonalized_files_all = Parallel(n_jobs=len(VARIABLES), 
                                        temp_folder="/scratch/b/b309178/tmp/",
                                        backend="loky"
                                        )(
        delayed(deseasonalize)(var_name, OUTPUT_FOLDER, merge_variable_year[var_name], var_climatology[var_name], DATA_START, DATA_END)
        for var_name in VARIABLES.keys()
    )    
    
    print("Deseasonalizing 1941-1980...")
    deseasonalized_files_1941_1980 = Parallel(n_jobs=len(VARIABLES), 
                                        temp_folder="/scratch/b/b309178/tmp/",
                                        backend="loky")(
        delayed(deseasonalize)(var_name, OUTPUT_FOLDER, merge_variable_year[var_name], var_climatology[var_name], 1941, 1980)
        for var_name in VARIABLES.keys()
    )
    
    print("Deseasonalizing 1981-1999...")
    deseasonalized_files_1981_1999 = Parallel(n_jobs=len(VARIABLES))(
        delayed(deseasonalize)(var_name, OUTPUT_FOLDER, merge_variable_year[var_name], var_climatology[var_name], 1981, 1999)
        for var_name in VARIABLES.keys()
    )
    
    print("Deseasonalizing 2000-2022...")
    deseasonalized_files_2000_2022 = Parallel(n_jobs=len(VARIABLES))(
        delayed(deseasonalize)(var_name, OUTPUT_FOLDER, merge_variable_year[var_name], var_climatology[var_name], 2000, 2022)
        for var_name in VARIABLES.keys()
    )
    
    # apply land mask
    masking_land = Parallel(n_jobs=1)(
        delayed(apply_land_mask)(var_name, OUTPUT_FOLDER, MASK_FILE)
        for var_name in VARIABLES.keys()
    )
        
        
    print("Processing complete!")


if __name__ == "__main__":
    main()
