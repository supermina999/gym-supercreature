from gym.envs.registration import register

register(
    id='supercreature-v0',
    entry_point='gym_supercreature.envs:SupercreatureEnv'
)