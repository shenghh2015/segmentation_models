import segmentation_models_v1 as sm
import numpy as np
model = sm.BiFPN('efficientnetb0', input_shape =(512,512,3), classes=4)
y=model.predict(np.ones((1,512,512,3)))
y.shape