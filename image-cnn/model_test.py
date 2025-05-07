import warnings
from fastai.vision.all import *

# Suppress the warning about load_learner
warnings.filterwarnings("ignore", category=UserWarning, module="fastai.learner")

learn = load_learner('resnet34_nsfw.pkl')
output = learn.predict('hunny.jpg')
print(output)