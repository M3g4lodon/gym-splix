# SplixEnvironment
Provides a OpenAI-Gym-like environment on the online game Splix.io

You need to install Gym and Selenium : 
```sh
pip install gym
pip install selenium
```

How to Setup up the env :
```sh
pip install git+ .
```

To try it :
```python
import gym
import gym_splix


YOUR_FIREFOX_PATH="PATH\\TO\\FIREFOX"

def run():
    env = gym.make("splix-online-v0")
    env.firefox_path = YOUR_FIREFOX_PATH
    env.reset()
    while not env.done:
        env.render()
        env.step(env.action_space.sample())  # random action
    env.close()

if __name__ == '__main__':
    run()

```
