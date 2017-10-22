
import numpy as np
import matplotlib.pyplot as plt

import prepare,align,classify,present

CLSNAMES = ['phnat','bualex']
CLSCOUNT = len(CLSNAMES)
# compose a video path for each classname
VIDPATHS = ['./data/vid'+n+'.mp4' for n in range(CLSCOUNT)]
images = [prepare.load_images(VIDPATHS[n]) for n in range(CLSCOUNT)]
X =[]
y = []
for i in range(CLSCOUNT):
    for img in images[i]:
        X.append(img)
        y.append(i)
X = np.asarray(X)
print "composed X matrix. Shape:", X.shape, "y shape:", y.shape

X_f = align.frontalize(X)
print "perormed frontalzation."
present.two_img(X[1],X_f[1])

net = classify.DeepRtlNet()
