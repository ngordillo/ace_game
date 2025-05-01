# import gym
import numpy as np
# from gym import spaces
import subprocess
import netCDF4 as nc
import os
import ace_data_funcs

import gymnasium as gym
from gymnasium import spaces

# from gymnasium.envs.registration import register


class ClimateControlEnv(gym.Env):
    """Custom Gymnasium environment for climate temperature control."""
    
    def __init__(self):
        super(ClimateControlEnv, self).__init__()

        # Define observation space (Seattle Temp, Boston Temp, flattened temp grid)
        self.grid_size = (180, 360)  # Assuming global temperature grid
        self.observation_space = spaces.Box(low=260, high=310, shape=(2,), dtype=np.float32)

        # Define action space (Temperature modification in a predefined region)
        # self.action_space = spaces.Box(low=-30, high=30, shape=(1,), dtype=np.float32)
        

        self.discrete_actions = np.array([-15, -10, -5, 0, 5, 10, 15], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        # Initial conditions
        # self.seattle_temp = 283.4  # Initial Seattle temp (K)
        # self.boston_temp = 269.12  # Initial Boston temp (K)
        # self.temperature_field = 0 # Mock initial temp field
        self.temperature_field = nc.Dataset('../../Data/ace_output/run1_autoregressive_predictions.nc', 'r+')['surface_temperature'][0,:,:]

        self.current_step = 1
        self.max_steps = 14  # Total iterations per episode
        self.set_loc_temps()

        

    def step(self, action):
        """
        Apply the selected temperature change and update the environment.
        """
        # temp_modification = action[0]
        # temp_modification = action
        temp_modification = self.discrete_actions[action]


        u_lat = 47.61
        u_lon = 360 + (-122.33)

        # Modify the ocean region temperature (Mock update)

        ace_data_funcs.copy_files(self.current_step)
        dataset = nc.Dataset('../../Data/ace_output/run' + str(self.current_step) + '_autoregressive_predictions.nc', 'r+')

        dataset_restart = nc.Dataset('../../Data/ace_output/run' + str(self.current_step) + '_restart.nc', 'r+')


        lats = np.asarray(dataset.variables['lat'][:])
        lons = np.asarray(dataset.variables['lon'][:])

        lat_ind = (np.abs(lats - u_lat)).argmin()
        lon_ind = (np.abs(lons - u_lon)).argmin()

        # print(lat_ind)
        # print(lon_ind)

        dataset['surface_temperature'][0,10,lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_modification)

        dataset_restart['surface_temperature'][0,lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_modification)
        dataset.close()
        dataset_restart.close()

        # Run climate model (Replace with actual ACE integration)

        ace_data_funcs.create_yaml(self.current_step)
        self.run_climate_model()

        self.temperature_field = nc.Dataset('../../Data/ace_output/run' + str(self.current_step) + '_autoregressive_predictions.nc', 'r+')['surface_temperature'][0,:,:]

        self.set_loc_temps()

        # Extract new Seattle and Boston temperatures
        # self.seattle_temp = self.get_temperature_at_location("Seattle")
        # self.boston_temp = self.get_temperature_at_location("Boston")
        

        # Compute reward (absolute temperature difference)
        reward = abs(self.seattle_temp - self.boston_temp)

        # Check if the episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        trunc = self.current_step >= self.max_steps



        # Create observation vector
        observation = np.concatenate(([self.seattle_temp, self.boston_temp], self.temperature_field.flatten()))

        return observation, reward, done, trunc, {}

    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.seattle_temp = 0
        self.boston_temp = 0
        # self.temperature_field = 0
        self.current_step = 1

        self.temperature_field = nc.Dataset('../../Data/ace_output/run1_autoregressive_predictions.nc', 'r+')['surface_temperature'][0,:,:]
        observation = np.concatenate(([self.seattle_temp, self.boston_temp], self.temperature_field.flatten()))
        return observation

    def render(self, mode="human"):
        """
        Render the environment using Pygame (if needed).
        """
        pass  # Placeholder, integrate with your existing Pygame interface

  
    
    def get_loc_temps(self, u_lat, u_lon):
        dataset = nc.Dataset('../../Data/ace_output/lat_lons.nc', 'r+')
        lats = np.asarray(dataset.variables['lat'][:])
        lons = np.asarray(dataset.variables['lon'][:])
        

        lat_ind = lats[(np.abs(lats - u_lat)).argmin()]
        lon_ind = lons[(np.abs(lons - u_lon)).argmin()]


        if (self.current_step == 0):
            dataset = nc.Dataset('../../Data/ace_data/initialization/initialization_data.nc', 'r+')
            temp = dataset['surface_temperature'][0,lat_ind,lon_ind]
        else:
            # print("test:" + str(self.current_step))
            dataset = nc.Dataset('../../Data/ace_output/run' + str(self.current_step) + '_autoregressive_predictions.nc', 'r+')
            temp = dataset['surface_temperature'][0,10,lat_ind,lon_ind]

        dataset.close()
        return float(temp)
    

    def set_loc_temps(self):
        # Initial conditions
        self.seattle_temp = round(self.get_loc_temps(47.61, 360 + (-122.33)),2)  # Initial Seattle temp (K)
        self.boston_temp =  round(self.get_loc_temps(42.36, 360 + (-71.06)),2)

    def run_climate_model(self):
        """
        Runs the ACE climate model and updates the temperature field.
        """
        if os.path.exists('/Users/nicojg/Data/ace_data/initialization/initialization_data.nc'):
            print("File exists")
        else:
            print("File does not exist")
            quit()

        activate_command = f"conda run -n fme PYTHONPATH=ace/fme python -m fme.ace.inference ../configs/default_" + str(self.current_step) + "_test_config.yaml"

        # Execute the activate command in a new shell
        subprocess.run(activate_command, shell=True)

        os.rename('../../Data/ace_output/autoregressive_predictions.nc','../../Data/ace_output/run' + str(self.current_step+1) + '_autoregressive_predictions.nc')
        os.rename('../../Data/ace_output/restart.nc','../../Data/ace_output/run' + str(self.current_step+1) + '_restart.nc')

        ace_data_funcs.fix_lons('../../Data/ace_output/run' + str(self.current_step+1) + '_autoregressive_predictions.nc')
        ace_data_funcs.fix_lons('../../Data/ace_output/run' + str(self.current_step+1) + '_restart.nc')

    
# print(gym.envs.registry.keys())

# register(
#     id="ClimateControl-v0",
#     entry_point="climate_env.climate_control_env:ClimateControlEnv",
# )

# test = gym.make("ClimateControl-v0")
# ACE_driver = ClimateControlEnv()
# print(ACE_driver.boston_temp)
# ACE_driver.step([10], 47.61, 360 + (-122.33))
# print(ACE_driver.boston_temp)

# from enum import Enum

# import numpy as np
# import pygame

# import gymnasium as gym
# from gymnasium import spaces

# class Actions(Enum):
#     DEC_BIG = -2
#     DEC_SMALL = -1
#     NO_CHANGE = 0
#     INC_SMALL = 1
#     INC_LARGE = 2



# class CustomEnv(gym.Env):
#     metadata = {"render_modes": ["human"]}

#     def __init__(self, grid_size=(10, 10), temp_range=(0, 100), action_step=5, external_model=None):
#         super(CustomEnv, self).__init__()
#         self.grid_size = grid_size
#         self.temp_range = temp_range
#         self.action_step = action_step
#         self.external_model = external_model

#         self.observation_space = gym.spaces.Box(low=self.temp_range[0], high=self.temp_range[1], shape=self.grid_size, dtype=np.float32)  # Example observation space

#         self.action_space = spaces.MultiDiscrete([grid_size[0], grid_size[1], 2]) 
        
#         # Initialize temperature field
#         self.reset()

    
#     def reset(self):    
#         # Reset the environment
#         pass

#     def step(self, action):
#         row, col, change = action

#         # Send action to the external model and receive the updated temperature field
#         if self.external_model:
#             self.temperature_field = self.external_model(self.temperature_field, action)
#         else:
#             raise ValueError("Ace failed")

#         # Perform a step in the environment based on the given action
#         pass

