# ============================
# Imports
# ============================
import subprocess
import os
import ace_data_funcs

# ============================
# Run ACE inference and process output files
# ============================
def inf_ace(run_num, fp, episode_num):

    # ============================
    # Check that initialization file exists before running inference
    # ============================
    # if os.path.exists('/Users/nicojg/Data/ace_data/initialization/initialization_data.nc'):
    if os.path.exists(fp + '/initialization_data.nc'):
        print("File exists")
    else:
        print("File does not exist")
        quit()

    # ============================
    # Construct command to run inference via subprocess
    # ============================
    activate_command = (
        f"conda run -n fme PYTHONPATH=ace/fme python -m fme.ace.inference "
        f"/home/nicojg/inference_engine/configs/default_{run_num-1}_test_config.yaml"
    )

    # ============================
    # Execute the inference command in a shell
    # ============================
    subprocess.run(activate_command, shell=True)

    # ============================
    # Rename output files to be episode/run-specific
    # ============================
    # os.rename('../../Data/ace_output/autoregressive_predictions.nc','../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc')
    os.rename(
        fp + 'ace_output/autoregressive_predictions.nc',
        fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions.nc'
    )

    # os.rename('../../Data/ace_output/restart.nc','../../Data/ace_output/run' + str(run_num) + '_restart.nc')
    os.rename(
        fp + 'ace_output/restart.nc',
        fp + 'ace_output/run' + str(run_num) + '_restart.nc'
    )

    # ============================
    # Fix longitudes in the output files
    # ============================
    # ace_data_funcs.fix_lons('../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc')
    # ace_data_funcs.fix_lons('../../Data/ace_output/run' + str(run_num) + '_restart.nc')

    ace_data_funcs.fix_lons(fp + 'ace_output/run' + str(run_num) + '_autoregressive_predictions.nc')
    ace_data_funcs.fix_lons(fp + 'ace_output/run' + str(run_num) + '_restart.nc')

    # ============================
    # (Redundant commented-out rename calls left for reference)
    # ============================
    # os.rename('../../Data/ace_output/autoregressive_predictions.nc','../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc')
    # os.rename('../../Data/ace_output/restart.nc','../../Data/ace_output/run' + str(run_num) + '_restart.nc')
