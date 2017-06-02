import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

from DQN import LifeTimeModel

model = LifeTimeModel('uniform-assets')

model.train()
