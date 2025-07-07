# ============================
# Imports
# ============================
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

# ============================
# Custom Gymnasium Environment for Climate Control
# ============================
class ClimateControlEnv(gym.Env):
    """Custom Gymnasium environment for climate temperature control."""

    def __init__(self, fp=None, ace_fp=None):
        super(ClimateControlEnv, self).__init__()
        print("printing fp")
        print(fp)
        self.fp = fp
        self.ace_fp = ace_fp

        # ============================
        # Define observation and action spaces
        # ============================
        self.grid_size = (180, 360)  # Global temperature grid
        self.observation_space = spaces.Box(low=260, high=310, shape=(2,), dtype=np.float32)

        # Discrete action space representing predefined temperature changes
        self.discrete_actions = np.array([-15, -10, -5, 0, 5, 10, 15], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        # ============================
        # Load initial temperature field from NetCDF
        # ============================
        print(self.fp + 'ace_output/run1_autoregressive_predictions.nc')
        self.temperature_field = nc.Dataset(
            fp + 'ace_output/run1_autoregressive_predictions.nc', 'r+'
        )['surface_temperature'][0, :, :]

        self.episode = 1
        self.current_step = 1
        self.max_steps = 5  # Total steps per episode
        self.set_loc_temps()

    # ============================
    # Environment step function
    # ============================
    def step(self, action):
        temp_modification = self.discrete_actions[action]

        u_lat = 47.61
        u_lon = 360 + (-122.33)

        ace_data_funcs.copy_files(self.current_step, self.fp)

        dataset = nc.Dataset(
            self.fp + 'ace_output/run' + str(self.current_step) + '_autoregressive_predictions.nc', 'r+'
        )
        dataset_restart = nc.Dataset(
            self.fp + 'ace_output/run' + str(self.current_step) + '_restart.nc', 'r+'
        )

        lats = np.asarray(dataset.variables['lat'][:])
        lons = np.asarray(dataset.variables['lon'][:])

        lat_ind = (np.abs(lats - u_lat)).argmin()
        lon_ind = (np.abs(lons - u_lon)).argmin()

        # Apply modification to a 43x51 patch centered on (lat_ind, lon_ind)
        dataset['surface_temperature'][0, 10, lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_modification)
        dataset_restart['surface_temperature'][0, lat_ind-21:lat_ind+22, lon_ind-25:lon_ind+26] += int(temp_modification)

        dataset.close()
        dataset_restart.close()

        # ============================
        # Run climate model and update state
        # ============================
        ace_data_funcs.create_yaml(self.current_step, self.fp, self.ace_fp, self.episode)
        self.run_climate_model()

        self.temperature_field = nc.Dataset(
            self.fp + 'ace_output/run' + str(self.current_step) + '_autoregressive_predictions.nc', 'r+'
        )['surface_temperature'][0, :, :]

        self.set_loc_temps()

        reward = abs(self.seattle_temp - self.boston_temp)

        self.current_step += 1
        done = self.current_step >= self.max_steps
        trunc = self.current_step >= self.max_steps

        # Observation includes current temps + full grid flattened
        observation = np.concatenate(
            ([self.seattle_temp, self.boston_temp], self.temperature_field.flatten())
        )

        return observation, reward, done, trunc, {}

    # ============================
    # Environment reset function
    # ============================
    def reset(self):
        self.seattle_temp = 0
        self.boston_temp = 0
        self.current_step = 1

        self.temperature_field = nc.Dataset(
            self.fp + 'ace_output/run1_autoregressive_predictions.nc', 'r+'
        )['surface_temperature'][0, :, :]

        observation = np.concatenate(
            ([self.seattle_temp, self.boston_temp], self.temperature_field.flatten())
        )
        return observation

    # ============================
    # (Optional) Render logic placeholder
    # ============================
    def render(self, mode="human"):
        pass  # Can integrate Pygame here if needed

    # ============================
    # Get temperature from specified lat/lon
    # ============================
    def get_loc_temps(self, u_lat, u_lon):
        dataset = nc.Dataset(self.fp + 'ace_output/lat_lons.nc', 'r+')
        lats = np.asarray(dataset.variables['lat'][:])
        lons = np.asarray(dataset.variables['lon'][:])

        lat_ind = lats[(np.abs(lats - u_lat)).argmin()]
        lon_ind = lons[(np.abs(lons - u_lon)).argmin()]

        if self.current_step == 0:
            print("error: current step is 0")
            dataset = nc.Dataset(self.fp + 'ace_data/initialization/initialization_data.nc', 'r+')
            temp = dataset['surface_temperature'][0, lat_ind, lon_ind]
        else:
            dataset = nc.Dataset(
                self.fp + 'ace_output/run' + str(self.current_step) + '_autoregressive_predictions.nc', 'r+'
            )
            temp = dataset['surface_temperature'][0, 10, lat_ind, lon_ind]

        dataset.close()
        return float(temp)

    # ============================
    # Set Seattle and Boston temps for current grid
    # ============================
    def set_loc_temps(self):
        self.seattle_temp = round(self.get_loc_temps(47.61, 360 + (-122.33)), 2)
        self.boston_temp = round(self.get_loc_temps(42.36, 360 + (-71.06)), 2)

    # ============================
    # Run ACE model for current step
    # ============================
    def run_climate_model(self):
        EXP_NAME = "falco_test"

        if os.path.exists(self.fp + 'ace_data/initialization/initialization_data.nc'):
            print("File exists")
        else:
            print("File does not exist")
            quit()

        activate_command = (
            f"conda run -n fme PYTHONPATH=ace/fme python -m fme.ace.inference "
            f"/home/nicojg/ace/configs/{EXP_NAME}/episode{self.episode}/{self.current_step}_test_config.yaml"
        )

        subprocess.run(activate_command, shell=True)

        # Rename model outputs to match run step
        os.rename(
            self.fp + 'ace_output/autoregressive_predictions.nc',
            self.fp + 'ace_output/run' + str(self.current_step + 1) + '_autoregressive_predictions.nc'
        )
        os.rename(
            self.fp + 'ace_output/restart.nc',
            self.fp + 'ace_output/run' + str(self.current_step + 1) + '_restart.nc'
        )

        # Fix longitude grid in outputs
        ace_data_funcs.fix_lons(
            self.fp + 'ace_output/run' + str(self.current_step + 1) + '_autoregressive_predictions.nc'
        )
        ace_data_funcs.fix_lons(
            self.fp + 'ace_output/run' + str(self.current_step + 1) + '_restart.nc'
        )

    # ============================
    # Used by wrapped agent to sync episode context
    # ============================
    def update_episode(self, episode):
        self.episode = episode


# ============================
# Commented-out registration and test code
# ============================

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

#         self.observation_space = gym.spaces.Box(low=self.temp_range[0], high=self.temp_range[1], shape=self.grid_size, dtype=np.float32)

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
