# rl-as-inference
Solving RL posed as inference

## Install
```bash
pip3 install -r requirements.txt
```

## Run
```bash
python3 -m playground.test_pointmass
python3 -m playground.test_bullet Walker2DBulletEnv-v0
python3 -m playground.test_bullet AntBulletEnv-v0
python3 -m playground.test_bullet ReacherBulletEnv-v0
python3 -m playground.test_bullet HumanoidFlagrunBulletEnv-v0
python3 -m playground.test_bullet HalfCheetahBulletEnv-v0
python3 -m playground.test_bullet HumanoidBulletEnv-v0
```
For more environments refer to [PyBullet envs code](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/__init__.py).

