from gymnasium.envs.registration import register
import climate_env.climate_control_env  # <- you MUST import this

register(
    id="ClimateControl-v0",
    entry_point="climate_env.climate_control_env:ClimateControlEnv",
)