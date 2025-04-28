import netCDF4 as nc
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import pickle

import os
import shutil

def fix_lons(nc_fp): # Open the NetCDF file
    dataset = nc.Dataset(nc_fp, 'r+')

    # Specify the new latitudes array
    new_longitudes = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 
        12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 
        24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 
        36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 
        48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 
        60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 
        72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 
        84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 
        96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 
        107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 
        117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5, 126.5, 
        127.5, 128.5, 129.5, 130.5, 131.5, 132.5, 133.5, 134.5, 135.5, 136.5, 
        137.5, 138.5, 139.5, 140.5, 141.5, 142.5, 143.5, 144.5, 145.5, 146.5, 
        147.5, 148.5, 149.5, 150.5, 151.5, 152.5, 153.5, 154.5, 155.5, 156.5, 
        157.5, 158.5, 159.5, 160.5, 161.5, 162.5, 163.5, 164.5, 165.5, 166.5, 
        167.5, 168.5, 169.5, 170.5, 171.5, 172.5, 173.5, 174.5, 175.5, 176.5, 
        177.5, 178.5, 179.5, 180.5, 181.5, 182.5, 183.5, 184.5, 185.5, 186.5, 
        187.5, 188.5, 189.5, 190.5, 191.5, 192.5, 193.5, 194.5, 195.5, 196.5, 
        197.5, 198.5, 199.5, 200.5, 201.5, 202.5, 203.5, 204.5, 205.5, 206.5, 
        207.5, 208.5, 209.5, 210.5, 211.5, 212.5, 213.5, 214.5, 215.5, 216.5, 
        217.5, 218.5, 219.5, 220.5, 221.5, 222.5, 223.5, 224.5, 225.5, 226.5, 
        227.5, 228.5, 229.5, 230.5, 231.5, 232.5, 233.5, 234.5, 235.5, 236.5, 
        237.5, 238.5, 239.5, 240.5, 241.5, 242.5, 243.5, 244.5, 245.5, 246.5, 
        247.5, 248.5, 249.5, 250.5, 251.5, 252.5, 253.5, 254.5, 255.5, 256.5, 
        257.5, 258.5, 259.5, 260.5, 261.5, 262.5, 263.5, 264.5, 265.5, 266.5, 
        267.5, 268.5, 269.5, 270.5, 271.5, 272.5, 273.5, 274.5, 275.5, 276.5, 
        277.5, 278.5, 279.5, 280.5, 281.5, 282.5, 283.5, 284.5, 285.5, 286.5, 
        287.5, 288.5, 289.5, 290.5, 291.5, 292.5, 293.5, 294.5, 295.5, 296.5, 
        297.5, 298.5, 299.5, 300.5, 301.5, 302.5, 303.5, 304.5, 305.5, 306.5, 
        307.5, 308.5, 309.5, 310.5, 311.5, 312.5, 313.5, 314.5, 315.5, 316.5, 
        317.5, 318.5, 319.5, 320.5, 321.5, 322.5, 323.5, 324.5, 325.5, 326.5, 
        327.5, 328.5, 329.5, 330.5, 331.5, 332.5, 333.5, 334.5, 335.5, 336.5, 
        337.5, 338.5, 339.5, 340.5, 341.5, 342.5, 343.5, 344.5, 345.5, 346.5, 
        347.5, 348.5, 349.5, 350.5, 351.5, 352.5, 353.5, 354.5, 355.5, 356.5, 
        357.5, 358.5, 359.5]

    # Modify the latitudes in the NetCDF file
    dataset.variables['lon'][:] = new_longitudes

    # Close the NetCDF file
    dataset.close()

def create_yaml(run_num): # Open the NetCDF file

  file_template = """
  experiment_dir: ../../Data/ace_output/
  n_forward_steps: 40
  forward_steps_in_memory: 10
  checkpoint_path: ../../Data/ace_data/ckpt/ace_ckpt.tar
  logging:
    log_to_screen: true
    log_to_wandb: false
    log_to_file: true
    project: ace
    # entity: your_wandb_entity
  initial_condition:
    path: ../../Data/ace_output/run""" + str(run_num) + """_restart.nc
    start_indices:
        first: 0
        n_initial_conditions: 1
  forcing_loader:
    dataset:
      data_path: ../../Data/ace_data/validation/ic_0011/
    num_data_workers: 4
  data_writer:
    save_prediction_files: true
    save_monthly_files: false
    names: ["surface_temperature"]
  """.strip()

  for i in range(1,100001):
      file_contents = file_template.format(i)

      with open(f"../configs/default_"+ str(run_num)+ "_test_config.yaml", "w") as f:
          f.write(file_contents)

def get_loc_temp(u_lat, u_lon, run_num):
  dataset = nc.Dataset('../../Data/ace_output/lat_lons.nc', 'r+')
  lats = np.asarray(dataset.variables['lat'][:])
  lons = np.asarray(dataset.variables['lon'][:])
  

  lat_ind = lats[(np.abs(lats - u_lat)).argmin()]
  lon_ind = lons[(np.abs(lons - u_lon)).argmin()]


  if (run_num == 0):
    dataset = nc.Dataset('../../Data/ace_data/initialization/initialization_data.nc', 'r+')
    temp = dataset['surface_temperature'][0,lat_ind,lon_ind]
  else:
    dataset = nc.Dataset('../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc', 'r+')
    temp = dataset['surface_temperature'][0,10,lat_ind,lon_ind]

  dataset.close()
  return float(temp)
# print(type(float(get_loc_temp(47.61, 360 + (-122.33),1))))
  
def update_plot(run_num):
  if (run_num == 0):
    filename = '../../Data/ace_data/initialization/initialization_data.nc'
  else:
    filename = '../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc'

  ds = xr.open_dataset(filename)
  ds["lon"] = np.arange(0, 360, 1)  # longitudes are a bit funny, so we need to redefine them

  var = "surface_temperature"  # "PRATEsfc"
  member = 0  # which member to plot  
  leadtime = 10  # in increments of 6 hours
  #-----------------------------------------
  # temperature colormap from pickle
  fp = open('../../Data/ace_data/sfc_temp_cmap.pkl', 'rb')
  sfc_temp_cmap = pickle.load(fp)
  fp.close()

  # create 500avo colormap
  colors1 = plt.cm.YlOrRd(np.linspace(0, 1, 36))
  colors2 = plt.cm.BuPu(np.linspace(0.5, 0.75, 8))
  colors_500avo = np.vstack((colors2, (1, 1, 1, 1), colors1))
  colors_500avo = plt.cm.colors.ListedColormap(np.vstack((colors2, (1, 1, 1, 1), colors1)))

  # Plot and compare the models
  fig = plt.figure(figsize=(15, 5))
  ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

  if (run_num == 0):
    img = ds[var][0,:,:].plot(ax=ax, transform=ccrs.PlateCarree(), cmap=sfc_temp_cmap)
  else:
    img = ds[var][member,leadtime,:,:].plot(ax=ax, transform=ccrs.PlateCarree(), cmap=sfc_temp_cmap)
  if var == "surface_temperature":
      img.set_clim(260, 310)
  else:
      img.set_cmap("Blues")
      img.set_clim(0, 0.0005)

  # Add coastlines and gridlines
  ax.coastlines()

  ds.close()

  plt.tight_layout()
  plt.savefig("../../Data/ace_output/figures/run" + str(run_num) + "_temps.png", dpi=300, bbox_inches = 'tight')
  # plt.show()

def modify_ace_input(u_lat, u_lon, temp_change, run_num):

  copy_files(run_num)

  dataset = nc.Dataset('../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc', 'r+')

  dataset_restart = nc.Dataset('../../Data/ace_output/run' + str(run_num) + '_restart.nc', 'r+')


  lats = np.asarray(dataset.variables['lat'][:])
  lons = np.asarray(dataset.variables['lon'][:])

  lat_ind = (np.abs(lats - u_lat)).argmin()
  lon_ind = (np.abs(lons - u_lon)).argmin()

  print(lat_ind)
  print(lon_ind)

  dataset['surface_temperature'][0,10,lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_change)

  dataset_restart['surface_temperature'][0,lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_change)
  dataset.close()
  dataset_restart.close()

def copy_files(run_num):
  if not os.path.exists('../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions_original.nc'):
    original_file = '../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc'
    copy_file = '../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions_original.nc'

    shutil.copyfile(original_file, copy_file)
  else:
    original_file  = '../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions_original.nc'
    copy_file = '../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc'
  
    shutil.copyfile(original_file, copy_file)

  if not os.path.exists('../../Data/ace_output/run' + str(run_num) + '_restart_original.nc'):
    original_file = '../../Data/ace_output/run' + str(run_num) + '_restart.nc'
    copy_file = '../../Data/ace_output/run' + str(run_num) + '_restart_original.nc'

    shutil.copyfile(original_file, copy_file)
  else:
    original_file  = '../../Data/ace_output/run' + str(run_num) + '_restart_original.nc'
    copy_file = '../../Data/ace_output/run' + str(run_num) + '_restart.nc'
  
    shutil.copyfile(original_file, copy_file)
  # fix_lons('../../Data/ace_output/run' + str(1) + '_autoregressive_predictions.nc')
# fix_lons('../../Data/ace_output/run' + str(1) + '_restart.nc')