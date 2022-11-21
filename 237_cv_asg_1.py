import numpy as np
import cv2 as cv
import streamlit as st
from PIL import Image

def filtering(img, kernel):
    kernel = np.ones((kernel, kernel), np.float32) / 25
    res = cv.filter2D(img, -1, kernel)
    return res

def Gaussian1(img, val, kernel):
    res = cv.GaussianBlur(img, (kernel, kernel), val)
    return res

def result():
    
    st.title("Assignment-1")
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    val = ['Filtering', 'Gaussian Filtering']
    vals = st.radio('Select one', val, index=1)

    if vals == 'Filtering':
        kernel = st.slider("Kernel", min_value=3, max_value=15)

        res = filtering(original_image, kernel)
        st.image(original_image)
        st.image(res)

    else:
        kernel = [5, 15]
        kernel_size = st.select_slider("Kernel", options=kernel)
        res = Gaussian1(original_image, 0, kernel_size)
        st.image(original_image)
        st.image(res)



if __name__ == '__main__':
    result()




# import numpy as np
# import cv2 
# import streamlit as st
# from PIL import Image


# def Average_filter(image, kernel_size):
#     kernel = np.ones((kernel_size, kernel_size), np.float32) / 25
#     dst = cv2.filter2D(image, -1, kernel)
#     return dst


# def add_noise(image):

#     image = image / 255
#     # blank image
#     x, y = image.shape
#     g = np.zeros((x, y), dtype=np.float32)

#     # salt and pepper amount
#     pepper = 0.1
#     salt = 0.95

#     # create salt and pepper noise image
#     for i in range(x):
#         for j in range(y):
#             rdn = np.random.random()
#             if rdn < pepper:
#                 g[i][j] = 0
#             elif rdn > salt:
#                 g[i][j] = 1
#             else:
#                 g[i][j] = image[i][j]

#     return g


# def GaussianBlur(image, amount, kernel_size):
#     blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), amount)
#     return blur

# def MedianBlur(image, amount, kernel_size):
#     median = cv2.medianBlur(image, kernel_size, amount)
#     return median


# def main_loop():
#     st.title("CV Assignment 1")
#     st.subheader("Average Filter (3x3, 5x5, 11x11, and 15x15). Analysis of using avg filters with different kernel sizes. Adding Salt and Pepper noise. Removing noise using a median filter with different kernel sizes. Analysis of using Gaussian kernels with different kernel sizes for Blur effect.")
#     st.text("Name: Kshitija Lade")
#     st.text("Roll No.: 253")
#     st.text("PRN: 0120190090")
#     st.text("Batch: CV2")

#     image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
#     if not image_file:
#         return None

#     original_image = Image.open(image_file)
#     original_image = np.array(original_image)

#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     salt_pepper = add_noise(gray)

#     options = ['Average Filtering', 'Salt and Pepper Noise', 'Gaussian Filtering', 'Median Filtering']
#     option = st.radio('Choose the option', options, index=1)

#     if option == 'Average Filtering':
#         kernel_size = st.slider("Kernel", min_value=3, max_value=15)

#         fil = Average_filter(original_image, kernel_size)

#         st.text("Average Filtering")
#         st.text("Original Image -----> 2D Filter Image with different Kernel Size")
#         st.image([original_image, fil])

#     elif option == 'Salt and Pepper Noise':
#         st.text("Adding Salt and Pepper Noise")
#         st.text("Original Image -----> GrayScale Image -----> Image with Salt and Pepper Noise")
#         st.image([original_image, gray, salt_pepper])

#     elif option == 'Gaussian Filtering':

#         kernel = [3, 5, 11, 15]

#         kernel_size = st.select_slider("Kernel", options=kernel)
#         blur_rate = st.slider("Blurring", min_value=0.5, max_value=3.5)

#         processed_image1 = GaussianBlur(original_image, blur_rate, kernel_size)

#         st.text("Gaussian Filtering")
#         st.text("Image with Salt and Pepper Noise -----> Processed image using Gaussian Filter")
#         st.image([salt_pepper, processed_image1])


#     elif option == 'Median Filtering':
#         kernel = [3, 5, 11, 15]

#         kernel_size = st.select_slider("Kernel", options=kernel)
#         blur_rate = st.slider("Blurring", min_value=0.5, max_value=3.5)

#         processed_image = MedianBlur(original_image, blur_rate, kernel_size)

#         st.text("Median Filtering")
#         st.text("Image with Salt and Pepper Noise -----> Processed image using Median Filter")
#         st.image([salt_pepper, processed_image])


# if __name__ == '__main__':
#     main_loop()