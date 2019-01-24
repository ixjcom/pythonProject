###将RGB图片转换为灰度图
from skimage import io
import numpy as np
from PIL import Image
img = Image.open('D://3.jpg')
l = img.convert('L')
l.save("D://4.jpg")