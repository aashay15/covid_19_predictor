from model2 import multi1
import numpy as np
MODEL_NAME = "Multi Layer Perceptron 1"

model = multi1()
model.load(MODEL_NAME)

predict = [[2021,3,30]]
prediction = model.predict(predict)

print(np.rint(prediction))