## Algorithm Repo
Algorithms (ppo2, ppo2_normal, ppo2_cvae) are in another repo forked from OpenAI Baseline:
https://github.com/chenziku/baselines/tree/master/baselines/ppo2

## Install Dependencies and Setting Up 
For fast experiments, run the following on Google Colab:
```
pip install procgen
git clone https://github.com/chenziku/smirl-gen.git
pip uninstall -y imgaug
pip install 'imgaug<0.2.7,>=0.2.5'
pip install -e train-procgen
git clone https://github.com/chenziku/baselines.git
pip install tensorflow-gpu==1.15
pip install mpi4py
pip uninstall -y tensorflow_probability
pip install tensorflow_probability==0.8.0
pip install gputil
cd baselines
```

## Training from scratch
Specify the training algorithm it before the learn function in train.py or test.py

Run the following for training from scratch (200 levels in easy mode on CoinRun, starting from level 0):
```
python -m smirl-gen.train --env_name coinrun --distribution_mode easy --num_levels 200
```

For PPO + VAE, we can also run the following for training from a loaded policy and VAE (vae280/560):
```
python -m smirl-gen.train_load --env_name coinrun --distribution_mode easy --num_levels 200
```
Run the following for test (starting from level 1000):
```
!python -m smirl-gen.test --env_name bossfight --distribution_mode easy --start_level 1000
```

Follow https://github.com/openai/train-procgen for other specifications


## References

OpenAI Procgen Benchmark:
https://openai.com/blog/procgen-benchmark/

Train Procgen:
https://github.com/openai/train-procgen

Surprise Minimization (SMiRL):
https://bair.berkeley.edu/blog/2019/12/18/smirl/
https://arxiv.org/abs/1912.05510


