# SplixEnvironment
Provides a OpenAI-Gym-like environment on the online game Splix.io

How to Setup up the env :
```python
git clone https://github.com/M3g4lodon/gym-splix.git
cd gym-splix
pip install -e .
```

To try it :
```python
import gym
import gym_splix

def run():
    env = gym.make("splix-online-v0")
    for _ in range(1):
        env.reset()
        while not env.done:
            env.render()
            env.step(env.action_space.sample())  # random action
        env.close()

if __name__ == '__main__':
    run()

```
