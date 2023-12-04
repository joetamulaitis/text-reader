import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from emnist import extract_test_samples

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(62, activation = 'softmax')
])

model.load_weights('trained_model.keras')

image = cv2.imread('text.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.show()

def read_image(image2):
    image = image2.copy()
    #canny thresholding?????
    # canny = cv2.Canny(image, 30, 200)
    # plt.imshow(canny, cmap='gray')
    # plt.show()

    #make image black and white to make finding contours work better
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    #thresh = 255 - thresh
    plt.imshow(thresh, cmap='gray')
    plt.show()

    #for the test image, 12X12 boxes every line, 2X2 gets close to boxing letters
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12,12))
    dilation = cv2.dilate(thresh.copy(), rect_kernel, iterations = 1)

    plt.imshow(dilation, cmap='gray')
    plt.show()

    contours = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    letters = np.zeros((0, 28, 28, 1))
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


        line = thresh[y:y+h, x:x+w]

        rect_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        dilation2 = cv2.dilate(line.copy(), rect_kernel2, iterations = 1)

        contours2 = cv2.findContours(dilation2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        for j, contour2 in enumerate(contours2):
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            
            letter = line[y2:y2+h2, x2:x2+w2]
            
            letter = cv2.resize(letter, (28, 28))
            letter = np.reshape(letter, (1, 28, 28, 1))
            letter = tf.keras.utils.normalize(letter, axis=1)
            # if i == 3:
            #     plt.imshow(letter[0], cmap='gray')
            #     plt.show()
            letters = np.append(letters, letter, axis=0)

    plt.imshow(image, cmap='gray')
    plt.show()  

    print(letters.shape)
    for letter in letters:
        plt.imshow(letter, cmap='gray')
        plt.show()

    results = np.argmax(np.round(model.predict(letters)), axis=1)
    return results

emnist_to_ascii = {0: 48,
1: 49,
2: 50,
3: 51,
4: 52,
5: 53,
6: 54,
7: 55,
8: 56,
9: 57,
10: 65,
11: 66,
12: 67,
13: 68,
14: 69,
15: 70,
16: 71,
17: 72,
18: 73,
19: 74,
20: 75,
21: 76,
22: 77,
23: 78,
24: 79,
25: 80,
26: 81,
27: 82,
28: 83,
29: 84,
30: 85,
31: 86,
32: 87,
33: 88,
34: 89,
35: 90,
36: 97,
37: 98,
38: 99,
39: 100,
40: 101,
41: 102,
42: 103,
43: 104,
44: 105,
45: 106,
46: 107,
47: 108,
48: 109,
49: 110,
50: 111,
51: 112,
52: 113,
53: 114,
54: 115,
55: 116,
56: 117,
57: 118,
58: 119,
59: 120,
60: 121,
61: 122}

# test_samples, test_labels = extract_test_samples('byclass')
# test_data = tf.keras.utils.normalize(test_samples, axis = 1)
# test_data = np.expand_dims(test_samples, axis=3)

# results = np.argmax(np.round(model.predict(test_data[0:10])), axis=1)
# print(results)
# print(test_labels[0:10])

results = read_image(image)
results = np.vectorize(emnist_to_ascii.get)(results)
results = [chr(c) for c in results]

# labels = np.vectorize(emnist_to_ascii.get)(test_labels[0:10])
# labels = [chr(c) for c in labels]

print(results)
# print(labels)