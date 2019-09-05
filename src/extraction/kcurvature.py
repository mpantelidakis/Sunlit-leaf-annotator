"""
    Baseado no algoritmo abaixo
    https://github.com/accord-net/framework/blob/a5a2ea8b59173dd4e695da8017ba06bc45fc6b51/Sources/Accord.Math/Geometry/KCurvature.cs"""
import numpy as np
import math
from .extractor import Extractor
import sys
import cv2
import matplotlib.pyplot as plt
from skimage import feature
class KCURVATURE(Extractor):
     "5,30, [0,180]"
     def __init__(self):
            #__init__(self, k, band,  theta):
            self.k =35
            self.theta = [0,180]
            self.band=20
            self.suppression = self.k
            pass
     def angle_abc(self, a,  b,  c ):

            ab = [ b[0] - a[0], b[1] - a[1] ]
            cb = [ b[0] - c[0], b[1] - c[1] ]

            # dot product
            dot = (ab[0] * cb[0] + ab[1] * cb[1])

            cross = (ab[0] * cb[1] - ab[1] * cb[0])

            alpha = np.math.atan2(cross, dot)

            return int( np.math.floor(alpha * 180. / math.pi + 0.5))

     def find_angle(self,contours, image):
         list_angle=[]
         j=0
         for contour in contours:
             t=len(contour)
             map=[t]

             for i, con in enumerate(contour):
                    #im=np.copy(image)
                    ai =  (i + self.k) % t
                    ci =  (i - self.k) % t

                    aX=contour[ai][0][0]
                    aY=contour[ai][0][1]
                    bX=contour[i][0][0]
                    bY=contour[i][0][1]

                    cX = contour[ci][0][0]
                    cY = contour[ci][0][1]
                    list_angle.append(self.angle_abc([aX,aY],[bX,bY],[cX,cY]))
                    """test plot line and angle
                    pts = np.array([[aX,aY],[bX,bY],[cX,cY]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(im,[pts],True,(255,0,255))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im,'Angle:'+str(list_angle[i]),(bX,bY+8), font, 1,(255,0,255),1,cv2.LINE_AA)

                    cv2.imwrite("../../data/demo/teste/"+str(t)+"_" +str( j)+".png", im)
                    j=j+1
                    """

         return list_angle,image


     """"Source used https://dsp.stackexchange.com/questions/37281/improving-canny-edge-detection-and-contours-image-segmentation"""
     def auto_canny(self,image, sigma=0.5):
            v = np.median(image)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            return cv2.Canny(image, lower, upper)

     def run(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 9)

        ret,gray = cv2.threshold( np.copy(gray),127,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_BINARY_INV)
		#ret,gray = cv2.threshold( np.copy(gray),127,255,)

        #kernel = np.ones((5,5),np.uint8)
        #gray=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        #gray = cv2.dilate(gray,kernel,iterations = 1)
        #gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        #binario
        #ret, thresh = cv2.threshold(np.copy(gray), 127, 255, 0)

        edges = self.auto_canny(gray, 1)
        (major, minor, _) = cv2.__version__.split(".")

        if int(major) in [2, 4]:
            # OpenCV 2, OpenCV 4 case
            contour, hier = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            #print(contours)
            list,image=self.find_angle(contour,edges)
        else:
            # OpenCV 3 case
            img2,contours,hierarchy = cv2.findContours(edges.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #print(contours)
            list,image=self.find_angle(contours,img2)

        labels = []
        values = []
        list_band=[]
        for i in range(int(180/self.band)):
            list_band.append(0)
            for j,angle in enumerate(list):

                angle = abs(angle)
                #print(angle)
                if(angle>=i*self.band and angle<((i+1)*self.band)):
                    list_band[i]=list_band[i]+1
                elif (180==angle and int(180/self.band)==i):
                    list_band[i]=list_band[i]+1

        for i, c in enumerate(list_band):
            if(180==(i+1)*self.band):
                labels.append("K_%i_%i" % (i*self.band,180))
            else:
                labels.append("K_%i_%i" % (i*self.band,(i+1)*self.band-1))
            values.append(list_band[i])

        #compute_feats(image_float,kernels)

        types = ['numeric'] * len(labels)

        return labels, types, values

""""


img2,contours,hierarchy = cv2.findContours(edges ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
list,image=kcurve.find_angle(contours,img2)
list_band=[]
for i in range(180/10):
    list_band.append(0)
    for j,angle in enumerate(list):

        angle = abs(angle)
        print(angle)
        if(angle>=i*10 and angle<((i+1)*10)):
            list_band[i]=list_band[i]+1
        elif (180==angle and 180/10==i):
            list_band[i]=list_band[i]+1

print(list_band)

import matplotlib.image as mpimg
a = np.hstack(  list)
plt.hist(list_band, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

plt.imshow( gray)
plt.show()
import matplotlib.pyplot as plt
img = cv2.imread('../../data/demo/tabuleiro.jpg')
labels, types, values ,image= KCURVATURE().run(img)
print labels
print values
plt.imshow( image)
plt.show()

"""


