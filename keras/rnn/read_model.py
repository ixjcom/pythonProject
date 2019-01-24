# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# model = load_model("mnist_model")
# img = image.load_img("D://4.jpg",target_size=(28,28,1))
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis=0)
# img = preprocess_input(img)
# preds = model.predict(img)

from PIL import Image
import numpy as np
img = Image.open("D://4.jpg")
img = np.asarray(img)
print(type(img))
print(img.shape)
