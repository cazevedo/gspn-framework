from gym.envs.registration import register

register(
    id='gspn-env-v0',
    entry_point='gspn_gym_env.envs:GSPNenv',
)