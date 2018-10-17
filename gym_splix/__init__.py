from gym.envs.registration import register

register(
    id='splix-online-v0',
    entry_point='gym_foo.envs:FooEnv',
)