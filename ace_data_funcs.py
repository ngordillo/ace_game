# ============================
# Imports
# ============================
import netCDF4 as nc
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import pickle
import os
import shutil

# ============================
# Fix longitudes in NetCDF file
# ============================
def fix_lons(nc_fp):
    # Open the NetCDF file in read-write mode
    dataset = nc.Dataset(nc_fp, 'r+')

    # Define the corrected longitude values (0.5 to 359.5)
    new_longitudes = [0.5 + i for i in range(360)]

    # Overwrite longitude values in the file
    dataset.variables['lon'][:] = new_longitudes

    # Close the file to save changes
    dataset.close()

# ============================
# Create a configuration YAML file for ACE experiment
# ============================
def create_yaml(run_num, fp, ace_fp, episode): 
    EXP_NAME = "falco_test"
    
    # Template for YAML config
    file_template = """
  experiment_dir: """ + fp +  """ace_output/
  n_forward_steps: 16
  forward_steps_in_memory: 4
  checkpoint_path: """ + fp + """ace_data/ckpt/ace_ckpt.tar
  logging:
    log_to_screen: true
    log_to_wandb: false
    log_to_file: true
    project: ace
    # entity: your_wandb_entity
  initial_condition:
    path: """ + fp + """ace_output/run""" + str(run_num) + """_restart.nc
    # start_indices:
    #     first: 0
    #     n_initial_conditions: 1
  forcing_loader:
    dataset:
      data_path: """ + fp + """ace_data/fv3/validation/ic_0011
    num_data_workers: 4
  data_writer:
    save_prediction_files: true
    save_monthly_files: false
    names: ["surface_temperature"]
  """.strip()

    # Write the config to appropriate path
    for i in range(1,100001):
        file_contents = file_template.format(i)
        with open(ace_fp + f"configs/" + EXP_NAME + "/episode" + str(episode) + "/" + str(run_num) + "_test_config.yaml", "w") as f:
            f.write(file_contents)

# ============================
# Get surface temperature at specific lat/lon from ACE run
# ============================
def get_loc_temp(u_lat, u_lon, run_num, fp):
    dataset = nc.Dataset(fp + 'Data/ace_output/lat_lons.nc', 'r+')
    lats = np.asarray(dataset.variables['lat'][:])
    lons = np.asarray(dataset.variables['lon'][:])

    # Find nearest indices to requested lat/lon
    lat_ind = lats[(np.abs(lats - u_lat)).argmin()]
    lon_ind = lons[(np.abs(lons - u_lon)).argmin()]

    # Access temp from initial run or prediction depending on run_num
    if (run_num == 0):
        dataset = nc.Dataset(fp + 'ace_data/initialization/initialization_data.nc', 'r+')
        temp = dataset['surface_temperature'][0, lat_ind, lon_ind]
    else:
        dataset = nc.Dataset(fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions.nc', 'r+')
        temp = dataset['surface_temperature'][0, 10, lat_ind, lon_ind]

    dataset.close()
    return float(temp)

# ============================
# Plot surface temperature field for given ACE run
# ============================
def update_plot(run_num, fp):
    # Load dataset path based on run_num
    if (run_num == 0):
        filename = fp + 'Data/ace_data/initialization/initialization_data.nc'
    else:
        filename = fp + 'Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc'

    ds = xr.open_dataset(filename)
    ds["lon"] = np.arange(0, 360, 1)  # fix longitude values for plotting

    # Parameters for plot
    var = "surface_temperature"
    member = 0
    leadtime = 10  # ~2.5 days if 6-hour steps

    # Load custom colormap from pickle
    fp = open('../../Data/ace_data/sfc_temp_cmap.pkl', 'rb')
    sfc_temp_cmap = pickle.load(fp)
    fp.close()

    # Create 500 hPa anomaly color map (not used in this function)
    colors1 = plt.cm.YlOrRd(np.linspace(0, 1, 36))
    colors2 = plt.cm.BuPu(np.linspace(0.5, 0.75, 8))
    colors_500avo = plt.cm.colors.ListedColormap(np.vstack((colors2, (1, 1, 1, 1), colors1)))

    # Create the plot
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if (run_num == 0):
        img = ds[var][0,:,:].plot(ax=ax, transform=ccrs.PlateCarree(), cmap=sfc_temp_cmap)
    else:
        img = ds[var][member, leadtime, :, :].plot(ax=ax, transform=ccrs.PlateCarree(), cmap=sfc_temp_cmap)

    if var == "surface_temperature":
        img.set_clim(260, 310)
    else:
        img.set_cmap("Blues")
        img.set_clim(0, 0.0005)

    ax.coastlines()

    ds.close()

    # Save plot
    plt.tight_layout()
    plt.savefig("../../Data/ace_output/figures/run" + str(run_num) + "_temps.png", dpi=300, bbox_inches='tight')
    # plt.show()

# ============================
# Modify input surface temperature at specified location
# ============================
def modify_ace_input(u_lat, u_lon, temp_change, run_num, fp):
    # First restore original files to avoid cumulative edits
    copy_files(run_num, fp)

    # Open prediction and restart files for editing
    dataset = nc.Dataset(fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions.nc', 'r+')
    dataset_restart = nc.Dataset(fp + 'ace_output/run' + str(run_num) + '_restart.nc', 'r+')

    lats = np.asarray(dataset.variables['lat'][:])
    lons = np.asarray(dataset.variables['lon'][:])

    # Get closest grid cell to target coordinates
    lat_ind = (np.abs(lats - u_lat)).argmin()
    lon_ind = (np.abs(lons - u_lon)).argmin()

    print(lat_ind)
    print(lon_ind)

    # Apply temperature perturbation to a rectangular patch
    dataset['surface_temperature'][0,10,lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_change)
    dataset_restart['surface_temperature'][0,lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_change)

    dataset.close()
    dataset_restart.close()

# ============================
# Restore original ACE output files to undo edits
# ============================
def copy_files(run_num, fp):
    # Autoregressive prediction file
    if not os.path.exists(fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions_original.nc'):
        original_file = fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions.nc'
        copy_file = fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions_original.nc'
        shutil.copyfile(original_file, copy_file)
    else:
        original_file  = fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions_original.nc'
        copy_file = fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions.nc'
        shutil.copyfile(original_file, copy_file)

    # Restart file
    if not os.path.exists(fp + 'ace_output/run' + str(run_num) + '_restart_original.nc'):
        original_file = fp + 'ace_output/run' + str(run_num) + '_restart.nc'
        copy_file = fp + 'ace_output/run' + str(run_num) + '_restart_original.nc'
        shutil.copyfile(original_file, copy_file)
    else:
        original_file  = fp + 'ace_output/run' + str(run_num) + '_restart_original.nc'
        copy_file = fp + 'ace_output/run' + str(run_num) + '_restart.nc'
        shutil.copyfile(original_file, copy_file)

    # fix_lons('../../Data/ace_output/run' + str(1) + '_autoregressive_predictions.nc')
    # fix_lons('../../Data/ace_output/run' + str(1) + '_restart.nc')
