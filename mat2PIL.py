import cv2
from PIL import Image


img = cv2.imread('../ceshi.jpg')  # opencv读取
print(type(img))
image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换代码
image.show()  # PIL显示