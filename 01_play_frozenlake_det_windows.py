import gym
from gym.envs.registration import register
from colorama import init
from kbhit import KBHit

init(autoreset=True)    # Reset the terminal mode to display ansi color

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')        # is_slippery False
env.render()                             # Show the initial board

key = KBHit()

while True:

    action = key.getarrow();
    if action not in [0, 1, 2, 3]:
        print("Game aborted!")
        break

    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break
