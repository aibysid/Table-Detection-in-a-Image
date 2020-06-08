import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

file = r'table_image1.jpg'
table_image_contour = cv2.imread(file,0)
table_image =  cv2.imread(file)

ret, thresh_value = cv2.threshold(
    table_image_contour, 180, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5),np.uint8)
dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)

contours, hierarchy = cv2.findContours(
    dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # bounding the images
    if y < 700:
        table_image = cv2.rectangle(table_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
plt.imshow(table_image)
plt.show()

cv2.imwrite(r'generated_image1.jpg', table_image)

#Preprocessing for pytesseract
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)


#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def arranging_text_output(text_positions):
    list_values = [i for i in text_positions.values()]
    unique_heights = list(set(list_values))
    unique_heights.sort()
    list_values.sort()
    list_items = []
    for i in text_positions.items():
        list_items.append(i)
    dict_unique_heights = {}
    for i in unique_heights:
        dict_unique_heights.update({i:[]})

    for heights in unique_heights:
        for element in list_items:
            if(element[1]==heights):
                dict_unique_heights[heights].append(element[0])

    for i in dict_unique_heights.keys():
        for j in dict_unique_heights[i]:
            print(j, " ", end=' ')
        print('')








#use these preprocessing  for increasing accuracy pass them in black_and_white() using passed image as parameter.
image = cv2.imread(file)
gray = get_grayscale(image)
thresh = thresholding(gray)
opening  = opening(gray)
canny = canny(gray)





def binarize(image_to_transform,threshold):
    output_image = image_to_transform.convert("L")
    for x in range(output_image.width):
        for y in range(output_image.height):
            if output_image.getpixel((x,y))<threshold:
                output_image.putpixel((x,y), 0)
            else:
                output_image.putpixel((x,y), 255)
    return output_image






#copy and paste your own tesseract path in place of 'C:\Program Files\Tesseract-OCR\tesseract' your path may look a bit different.
pytesseract.pytesseract.tesseract_cmd = (r'C:\Program Files\Tesseract-OCR\tesseract')
custom_config = r'--oem 3--psm 11'
from pytesseract import Output





def colored_image(file):
    image=cv2.imread(file)
    image_file = Image.open(file)
    for thresh in range(40,180,10):
        print("Trying with threshold " + str(thresh))
        display(binarize(image_file,thresh))
        #d=(pytesseract.image_to_string(binarize(Image.open(file), thresh)))
        d = pytesseract.image_to_data(image=binarize(image_file,thresh),config=custom_config,output_type = Output.DICT)
        n_boxes = len(d['text'])
        pixel_list = []
        for i in range(n_boxes):
            if(int(d['conf'][i])>5):
                (x,y,w,h) = (d['left'][i],d['top'][i],d['width'][i],d['height'][i])
                image = cv2.rectangle(image, (x,y), (x+w , y+h ), (0,255,0), 2)
                pixel_list.append((x,y,w,h))
        text_positions = {}
        for i in range(0,n_boxes):
            temp = {d['text'][i]:d['top'][i]}
            text_positions.update(temp)
        arranging_text_output(text_positions)
    





def black_and_white(file):
    image = cv2.imread(file)
    d = pytesseract.image_to_data(image,config=custom_config,output_type = Output.DICT)
    n_boxes = len(d['text'])
    pixel_list = []
    for i in range(n_boxes):
        if(int(d['conf'][i])>5):
            (x,y,w,h) = (d['left'][i],d['top'][i],d['width'][i],d['height'][i])
            image = cv2.rectangle(image, (x,y), (x+w , y+h ), (0,255,0), 2)
            pixel_list.append((x,y,w,h))
    
    text_positions = {}
    for i in range(0,n_boxes):
        temp = {d['text'][i]:d['top'][i]}
        text_positions.update(temp)
    arranging_text_output(text_positions)
    
    
    
    
def show_boxes(file):
    image = cv2.imread(file)
    d = pytesseract.image_to_data(image,config=custom_config,output_type = Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if(int(d['conf'][i])>5):
            (x,y,w,h) = (d['left'][i],d['top'][i],d['width'][i],d['height'][i])
            image = cv2.rectangle(image, (x,y), (x+w , y+h ), (0,255,0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    


#****BELOW ARE THE OUTPUT GENERATOR FUNCTIONS USE THEM ACCORDING TO INSTRUCTION IN THE COMMENTS****#


black_and_white(file)#Use this function for black and white image.
colored_image(file)# use this function for colored image
show_boxes(file) #use this function to see boxes
    

print("*****END --- OF --- PROGRAM*******")