from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):

	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err

def compare_images(imageA, imageB, title):

	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	fig = plt.figure(title)

	plt.suptitle("MSE: %.1f, SSIM: %.3f" % (m/4, s))

	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	plt.show()

original = cv2.imread("origin.png")
prediction = cv2.imread("predicted.png")


original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)


fig = plt.figure("Images")
images = ("origin", original),("predicted", prediction)

for (i, (name, image)) in enumerate(images):

	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")

plt.show()

compare_images(original, original, "Origin vs. Origin")
compare_images(original, prediction, "Origin vs. predicted")

