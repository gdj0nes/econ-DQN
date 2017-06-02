import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""

from DDPG import LifeTimeModel

model = LifeTimeModel('large-sample')

model.train()