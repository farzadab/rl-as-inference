'Playing around with the 2D Walker environment'

import pybullet_envs
import time
import gym

def main():
    env = gym.make('Walker2DBulletEnv-v0')
    env.render(mode='human')
    state = env.reset()
    for _ in range(1000):
        next_state, reward, done, extra = env.step(env.action_space.sample())
        print('Reward achieved:', reward)
        time.sleep(1/60)
        if done:
            env.reset()
            print('The environment was reset!')
            time.sleep(0.5)

if __name__ == '__main__':
    main()
