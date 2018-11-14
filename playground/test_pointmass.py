'Playing around with the PointMass environment'

from envs.pointmass import PointMass
import time

def main():
    env = PointMass()
    state = env.reset()
    env.render()
    for i in range(100):
        next_state, reward, done, extra = env.step(env.action_space.sample())
        print('Reward achieved:', reward)
        env.render()
        time.sleep(env.dt * 0.5)

if __name__ == '__main__':
    main()
