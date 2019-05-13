#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

def process_image(img):
    def grayscale(img):
        """Change 3 chanel to graylevel image"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    def canny(img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)
    def gaussian_blur(img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    def anglefilter(lines,low,high):
        filtedlines=[]
        if lines is None:
            return filtedtlines
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x1==x2 or y1==y2:
                    continue
                angle=abs(np.arctan((y2-y1)/(x2-x1))*180/np.pi)
                if low<angle<high:
                    filtedlines.append([[x1,y1,x2,y2]])
        return filtedlines
    def clean_lines(lines,threshold):
        slope=[]
        for line in lines:
            for x1,y1,x2,y2 in line:
                k=(y1-y2)/(x1-x2)
                slope.append(k)
        while len(lines)>0:
            mean=np.mean(slope)
            diff=[abs(s-mean) for s in slope]
            # find thd index
            idx=np.argmax(diff)
            if diff[idx]>threshold:
                slope.pop(idx)
                lines.pop(idx)
            else:
                break
        return lines
    def get_lane(lines):
        left=[]
        right=[]
        left_lane=[]
        right_lane=[]
        for line in lines:
            for x1,y1,x2,y2 in line:
                coef=np.polyfit((x1,x2),(y1,y2),1)
                if coef[0]<0:
                    left.append(line)
                elif coef[0]>0:
                    right.append(line)
        if len(left)>1:
            left_array=np.vstack(np.array(left))
            x1=np.min(left_array[:,0])
            x2=np.max(left_array[:,2])
            y1=left_array[np.argmin(left_array[:,0]),1]
            y2=left_array[np.argmax(left_array[:,2]),3]
            left_lane=[[x1,y1,x2,y2]]
        else:
            left_lane=[[0,0,0,0]]
        if len(right)>1:
            right_array=np.vstack(np.array(right))
            x1=np.min(right_array[:,0])
            x2=np.max(right_array[:,2])
            y1=right_array[np.argmin(right_array[:,0]),1]
            y2=right_array[np.argmax(right_array[:,2]),3]
            right_lane=[[x1,y1,x2,y2]]
        else:
            right_lane=[[0,0,0,0]]
        return [left_lane,right_lane]
    def draw_line(image,lines,color,thickness):
        blank=np.zeros_like(image)
        y_min,y_max=image.shape[0]*0.65,image.shape[0]*1
        y_min = int(y_min)
        y_max = int(y_max)
        for line in lines:
            for x1,y1,x2,y2 in line:
                coef=np.polyfit((x1,x2),(y1,y2),1)
                k1=coef[0]
                b1=coef[1]
                x1=int((y_max-b1)/k1)
                x2=int((y_min-b1)/k1)
                a=cv2.line(blank,(x1,y_max),(x2,y_min),color,thickness)
        final=cv2.addWeighted(image,0.8,a,1,0)
        return final
    #read image
    image=mpimg.imread(img)
    #change 3 chanels to gray
    gray=grayscale(image)
    #gaussian blur
    con_gaussian=gaussian_blur(gray,3)
    #canny edge detector
    edge=canny(con_gaussian,30,100)
    #get blank
    mask1=np.zeros_like(edge)
    mask2=np.zeros_like(edge)
    ignore_mask_color=255
    imshape=image.shape
    vertices_left=np.array([[(0,imshape[0]),(-100+imshape[1]/2,3*imshape[0]/5),(imshape[1]/2,3*imshape[0]/5), (imshape[1]/2,imshape[0]),(imshape[1],imshape[0])]],dtype=np.int32)
    vertices_right=np.array([[(imshape[1],imshape[0]),(100+imshape[1]/2,3*imshape[0]/5),(imshape[1]/2,3*imshape[0]/5),(imshape[1]/2,imshape[0]),(0,imshape[0])]],dtype=np.int32)
    mask_color=255
    #get mask
    mask_i_left=cv2.fillPoly(mask1,vertices_left,mask_color)
    mask_i_right=cv2.fillPoly(mask2,vertices_right,mask_color)
    #get roi from original img
    roi_edge_left=cv2.bitwise_and(edge,mask_i_left)
    roi_edge_right=cv2.bitwise_and(edge,mask_i_right)
    #get line[[[x1,y1,x2,y2]],......] via HoughLinesP line detector
    initial_lines_left=cv2.HoughLinesP(roi_edge_left,1,np.pi/180,15,np.array([]),50,120)
    initial_lines_right=cv2.HoughLinesP(roi_edge_right,1,np.pi/180,15,np.array([]),50,120)
    #just get line in anglerange abs(30,70)
    filtedlines_left=anglefilter(initial_lines_left,30,70)
    filtedlines_right=anglefilter(initial_lines_right,30,70)
    #filte the line again, via diff between mean(slope)
    #refilted_left=clean_lines(filtedlines_left,0.01)
    #refilted_right=clean_lines(filtedlines_right,0.01)
    refilted=filtedlines_left+filtedlines_right
    #find de leftst point and rightest poin in each side
    lanes=get_lane(refilted)
    #draw line 
    image_lane=draw_line(image,lanes,[255,0,0],15)
    plt.imshow(image_lane)
    return image_lane
test=process_image('test_images/solidWhiteCurve.jpg')
print(test.shape)
