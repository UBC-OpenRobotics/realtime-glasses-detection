#!/usr/bin/pyton3

import dlib
import cv2
import numpy as np 
import os
import argparse as ap

def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def get_centers(img, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    # cv2.polylines(img, [pts], False, (255,0,0), 1) #画回归线
    # cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    # cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)
    scale = desired_dist / dist
    angle = np.degrees(np.arctan2(dy,dx)) 
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face

def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0)

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) 
    sobel_y = cv2.convertScaleAbs(sobel_y)
    # cv2.imshow('sobel_y',sobel_y)

    edgeness = sobel_y

    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w]
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure_1*0.3 + measure_2*0.7
    
    # cv2.imshow('roi_1',roi_1)
    # cv2.imshow('roi_2',roi_2)
    # print(measure)
    
    if measure > 0.15:
        judge = True
    else:
        judge = False
    print(judge)
    return judge

def detect(img, predictor, detector, show=False):

    #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if show:
        img_show = img.copy()

    rects = detector(gray, 1)

    #Define dict to store results for each face
    results = {}

    for i, rect in enumerate(rects):

        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face
        
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
        
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        
        judge = judge_eyeglass(aligned_face)
        
        results[i] = {"face_bbox":[x_face, y_face, w_face, h_face],"class":judge}

        if show:
            cv2.rectangle(img_show, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
            cv2.putText(img_show, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
       
            for (x, y) in landmarks:
                cv2.circle(img_show, (x, y), 2, (0, 0, 255), -1)
            
            #cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)
        
            if judge == True:
                cv2.putText(img_show, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(img_show, "No Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    
    if show:
        cv2.imshow("Result", img_show)

        k = cv2.waitKey(-1) & 0xFF
        if k==27:
            exit()

    return results


if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument('-i','--input', required=True, help="Path to input image or directory.")
    parser.add_argument('-s','--show', required=False, action='store_true',help='Display processed images')
    args=parser.parse_args()

    #Determine whether input points to file or folder
    input_path=args.input
    
    if os.path.isdir(input_path):
        img_paths = [os.path.join(input_path, filename) for filename in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        img_paths = [input_path]
    else:
        print("[ERROR] %s is neither a recognized file or folder" % input_path)
        exit()


    #Load predictor and dlib face detector
    predictor_path = "./data/shape_predictor_5_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img_path in img_paths:

        #Read image and convert to grayscale
        image = cv2.imread(img_path)

        results = detect(image, predictor, detector, show=args.show)
        print(results)
