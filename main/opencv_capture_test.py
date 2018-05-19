import os
from time import time as timer
import numpy as np
import cv2
from darkflow.net.build import TFNet


def nothing(x):
    pass


def show_scanner(frame, time, DIR_SCAN):
    '''
    给frame加上特效扫描线
    :param frame:
    :param time:
    :param DIR_SCAN:
    :return:
    '''
    num_total = len(os.listdir(DIR_SCAN))
    if time % instance == 0:
        pic_num = time / instance % (1 + num_total)
        # print('reading {}'.format(pic_num))
        scan = cv2.imread('{}/scan_{}.png'.format(DIR_SCAN, int(pic_num)), cv2.IMREAD_UNCHANGED)

        if scan is None:
            print('scan is None! Cannot read', 'scan2/scan_{}.png'.format(int(pic_num)))
            exit(0)

        scan = cv2.resize(scan, (w, h))
        frame = cv2.add(frame, scan)
        if int(time) == num_total * instance:
            time = int()

    time += 1
    return frame, time


def add_box(frame, tfnet, thresh):
    '''
    给frame加上boxes
    :param frame:
    :return:
    '''
    if frame is not None:
        results = tfnet.return_predict_thresh(frame, float(thresh)/100.)
        for result in results:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, (0, 0, 0), 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)


if __name__ == "__main__":
    options = {
        'model': 'E:/python_projects/darkflow-master/cfg/tiny-yolo-voc.cfg',
        'load': 'E:/python_projects/darkflow-master/bin/yolov2-tiny-voc.weights',
        'threshold': 0.3,
        'gpu': 0.7
    }
    tfnet = TFNet(options)


    cv2.namedWindow('window')
    switch_threshold = 'threshold'
    switch_box = 'box'
    switch_scan = 'scanner'
    cv2.createTrackbar(switch_threshold, 'window',int(options['threshold']*100),100,nothing)
    cv2.createTrackbar(switch_box, 'window',0,1,nothing)
    cv2.createTrackbar(switch_scan, 'window',0,1,nothing)

    time = 1  #scanner循环计数
    elapse = 0  #总帧数循环计数
    elapse_per = 24  #每几帧算一次fps
    fps = '0'
    instance = 1  #scan的帧数间隔
    DIR_BOX = 'out/box2_2.png'
    DIR_SCAN = 'scan2'

    if not os.path.exists(DIR_SCAN):
        raise FileNotFoundError('The DIR_SCAN is not exists. Please check', DIR_SCAN)

    if not os.path.exists(DIR_BOX):
        raise FileNotFoundError('The DIR_BOX is not exists. Please check', DIR_BOX)

    cap = cv2.VideoCapture(0)
    start = timer()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while(cap.isOpened()):
        # capture frame-by-frame
        _, frame = cap.read()
        h, w, _ = frame.shape

        if frame is None:
            print('The end. The frame is None.')
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break

        thresh = cv2.getTrackbarPos(switch_threshold, 'window')

        # 给图像向画框
        frame = add_box(frame, tfnet, thresh)

        # 再加box特效
        s = cv2.getTrackbarPos(switch_box, 'window')
        if s == 1:
            img = cv2.imread(DIR_BOX, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (w, h))
            frame = cv2.add(frame, img)

        # scanner特效
        s = cv2.getTrackbarPos(switch_scan, 'window')
        if s == 1:
            frame, time = show_scanner(frame, time, DIR_SCAN)

        # 计算fps
        if elapse % elapse_per == 0:
            fps = '{0:3.3f} FPS'.format(elapse / (timer() - start))
        elapse += 1

        img_fps = cv2.imread('scan/scan_1.png')
        img_fps = cv2.putText(img_fps,str(fps),(100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        # display the resulting frame
        cv2.imshow('window', frame)
        cv2.imshow('window2', img_fps)

    # when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()



