import subprocess
import os
import ace_data_funcs

def inf_ace(run_num):

    if os.path.exists('/Users/nicojg/Data/ace_data/initialization/initialization_data.nc'):
        print("File exists")
    else:
        print("File does not exist")
        quit()

    activate_command = f"conda run -n fme PYTHONPATH=ace/fme python -m fme.ace.inference ../configs/default_" + str(run_num-1) + "_test_config.yaml"

    # Execute the activate command in a new shell
    subprocess.run(activate_command, shell=True)

    os.rename('../../Data/ace_output/autoregressive_predictions.nc','../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc')
    os.rename('../../Data/ace_output/restart.nc','../../Data/ace_output/run' + str(run_num) + '_restart.nc')

    ace_data_funcs.fix_lons('../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc')
    ace_data_funcs.fix_lons('../../Data/ace_output/run' + str(run_num) + '_restart.nc')

    # os.rename('../../Data/ace_output/autoregressive_predictions.nc','../../Data/ace_output/run' + str(run_num) + '_autoregressive_predictions.nc')
    # os.rename('../../Data/ace_output/restart.nc','../../Data/ace_output/run' + str(run_num) + '_restart.nc')


# Create the directory if it doesn't exist
# data_dir = '../../Data'
# os.makedirs(data_dir, exist_ok=True)

# # Continue with the rest of your code
# # ...

