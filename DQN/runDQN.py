import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

from model import LifeTimeModel

model = LifeTimeModel('low-assets')

model.train()
