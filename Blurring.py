import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

orjImage = cv2.imread("C:/Users/Kenan/Desktop/image processing/Python Projects/Blurring/NYC.jpg")

#Normalizing the image
orjImage = cv2.cvtColor(orjImage,cv2.COLOR_BGR2RGB) / 255

plt.figure()
plt.imshow(orjImage)
plt.title("Orginal Image")
plt.axis("off")
plt.show()

#%% Gaussian Blurring

#Creating the gauss noise
def createGaussNoise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    
    noisy = gauss + image
    
    return noisy

gaussianNoise = createGaussNoise(orjImage)

plt.figure()
plt.imshow(gaussianNoise)
plt.title("Gaussian Noise Image")
plt.axis("off")
plt.show()

#Gaussing Blurring
gaussianImage = cv2.GaussianBlur(gaussianNoise,ksize = (3,3), sigmaX = 7)

plt.figure()
plt.imshow(gaussianImage)
plt.title("Gaussian Blurring Image")
plt.axis("off")
plt.show()

#%% Median Blurring

#Creating Salt-Pepper Noise
def saltPepperNoise(image):
    row, col , ch = image.shape
    s_vs_ap = 0.3
    amount = 0.004
    
    noisy = np.copy(image)
    
    #salt
    num_salt = np.ceil(amount * image.size * s_vs_ap)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords] = 1
    
    #pepper
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_ap))
    coords = [np.random.randint(0, i -1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0
    
    return noisy
    
saltPepperImage = saltPepperNoise(orjImage)

plt.figure()
plt.imshow(saltPepperImage)
plt.title("Salt-Pepper Image")
plt.axis("off")
plt.show()

#Gaussing Blurring
medianBlurring = cv2.medianBlur(saltPepperImage.astype(np.float32), ksize = 3)

plt.figure()
plt.imshow(medianBlurring)
plt.title("Median Blurring Image")
plt.axis("off")
plt.show()



