import numpy as np
import cv2
import cv2.aruco as aruco
import os
import glob
import math
import pyautogui
import time

pyautogui.FAILSAFE = False

# ------------------ Marker Detection ------------------ #
class Marker:
    def __init__(self, dict_type=aruco.DICT_4X4_50, thresh_constant=1):
        self.aruco_dict = aruco.Dictionary_get(dict_type)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = thresh_constant
        self.corners = None
        self.marker_x2y = 1
        self.mtx, self.dist = Marker.calibrate()

    @staticmethod
    def calibrate():
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        objpoints, imgpoints = [], []
        path = os.path.dirname(os.path.abspath(__file__))
        images = glob.glob(os.path.join(path, 'calib_images', 'checkerboard', '*.jpg'))
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist

    def detect(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.parameters)
        if ids is not None:
            self.rvecs, self.tvecs, _ = aruco.estimatePoseSingleMarkers(self.corners, 0.05, self.mtx, self.dist)
        else:
            self.corners = None

    def is_detected(self):
        return self.corners is not None

    def draw_marker(self, frame):
        if self.corners is not None:
            aruco.drawDetectedMarkers(frame, self.corners)

# ------------------ ROI ------------------ #
def in_cam(val, max_val):
    return max(0, min(val, max_val))

def ecu_dis(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def find_HSV(samples):
    color = np.uint8([samples])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color

def draw_box(frame, points, color=(0,255,127)):
    if points:
        for i in range(4):
            cv2.line(frame, points[i], points[(i+1)%4], color, 2)

class ROI:
    def __init__(self, roi_alpha1=1.5, roi_alpha2=1.5, roi_beta=2.5, hsv_alpha=0.3, hsv_beta=0.5, hsv_lift_up=0.3):
        self.roi_alpha1 = roi_alpha1
        self.roi_alpha2 = roi_alpha2
        self.roi_beta = roi_beta
        self.hsv_alpha = hsv_alpha
        self.hsv_beta = hsv_beta
        self.hsv_lift_up = hsv_lift_up
        self.roi_corners = None
        self.hsv_corners = None
        self.marker_top = None
        self.hsv_glove = None

    def findROI(self, frame, marker):
        rec_coor = marker.corners[0][0]
        c1,c2,c3,c4 = [tuple(pt) for pt in rec_coor]
        self.marker_top = ((c1[0]+c2[0])/2, (c1[1]+c2[1])/2)
        l = ecu_dis(c1,c4)
        slope_12 = (c1[1]-c2[1])/(c1[0]-c2[0]+1e-6)
        slope_14 = -1/slope_12
        sign = 1 if slope_14<0 else -1
        bot_rx = int(self.marker_top[0] + self.roi_alpha2*l*np.sqrt(1/(1+slope_12**2)))
        bot_ry = int(self.marker_top[1] + self.roi_alpha2*slope_12*l*np.sqrt(1/(1+slope_12**2)))
        bot_lx = int(self.marker_top[0] - self.roi_alpha1*l*np.sqrt(1/(1+slope_12**2)))
        bot_ly = int(self.marker_top[1] - self.roi_alpha1*slope_12*l*np.sqrt(1/(1+slope_12**2)))
        top_lx = int(bot_lx + sign*self.roi_beta*l*np.sqrt(1/(1+slope_14**2)))
        top_ly = int(bot_ly + sign*self.roi_beta*slope_14*l*np.sqrt(1/(1+slope_14**2)))
        top_rx = int(bot_rx + sign*self.roi_beta*l*np.sqrt(1/(1+slope_14**2)))
        top_ry = int(bot_ry + sign*self.roi_beta*slope_14*l*np.sqrt(1/(1+slope_14**2)))
        cam_h, cam_w = frame.shape[:2]
        self.roi_corners = [(in_cam(bot_lx, cam_w), in_cam(bot_ly, cam_h)),
                            (in_cam(bot_rx, cam_w), in_cam(bot_ry, cam_h)),
                            (in_cam(top_rx, cam_w), in_cam(top_ry, cam_h)),
                            (in_cam(top_lx, cam_w), in_cam(top_ly, cam_h))]

    def find_glove_hsv(self, frame, marker):
        x1,y1 = int(self.marker_top[0]-self.hsv_alpha*50), int(self.marker_top[1]-self.hsv_alpha*50)
        x2,y2 = int(self.marker_top[0]+self.hsv_alpha*50), int(self.marker_top[1]+self.hsv_alpha*50)
        cam_h, cam_w = frame.shape[:2]
        x1, y1, x2, y2 = in_cam(x1, cam_w), in_cam(y1, cam_h), in_cam(x2, cam_w), in_cam(y2, cam_h)
        region = frame[y1:y2, x1:x2]
        b,g,r = np.mean(region, axis=(0,1))
        self.hsv_glove = find_HSV([[r,g,b]])
        self.hsv_corners = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

    def cropROI(self, frame):
        pts = np.array(self.roi_corners)
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        cropped = frame[y:y+h, x:x+w].copy()
        pts = pts - pts.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255,255,255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        lower_range = np.array([self.hsv_glove[0][0][0]-5,50,50])
        upper_range = np.array([self.hsv_glove[0][0][0]+5,255,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        return mask

# ------------------ Glove Detection ------------------ #
class Glove:
    def __init__(self):
        self.fingers = 0
        self.arearatio = 0
        self.gesture = 0

    def find_fingers(self, FinalMask):
        conts,_ = cv2.findContours(FinalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(conts, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            self.arearatio = ((areahull-areacnt)/areacnt)*100
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            l = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start, end, far = tuple(cnt[s][0]), tuple(cnt[e][0]), tuple(cnt[f][0])
                a = np.linalg.norm(np.array(end)-np.array(start))
                b = np.linalg.norm(np.array(far)-np.array(start))
                c = np.linalg.norm(np.array(end)-np.array(far))
                s_ = (a+b+c)/2
                ar = math.sqrt(s_*(s_-a)*(s_-b)*(s_-c))
                d = (2*ar)/a
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c))*57
                if angle <= 90 and d>30: l+=1
            l+=1
            self.fingers = l
        except:
            self.fingers = 0

    def find_gesture(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.fingers==1:
            if self.arearatio<15: self.gesture=0; cv2.putText(frame,'0',(0,50),font,2,(0,0,255),3)
            elif self.arearatio<25: self.gesture=2; cv2.putText(frame,'2 fingers',(0,50),font,2,(0,0,255),3)
            else: self.gesture=1; cv2.putText(frame,'1 finger',(0,50),font,2,(0,0,255),3)
        elif self.fingers==2: self.gesture=3; cv2.putText(frame,'2',(0,50),font,2,(0,0,255),3)
        else: self.gesture=0

# ------------------ Mouse Control ------------------ #
class Mouse:
    def __init__(self):
        self.tx_old = 0
        self.ty_old = 0
        self.flag = 0
        self.trial = True

    def move_mouse(self, frame, position, gesture):
        sx,sy = pyautogui.size()
        camx,camy = frame.shape[1], frame.shape[0]
        mx_old,my_old = pyautogui.position()
        tx,ty = position
        if self.trial: self.trial, self.tx_old, self.ty_old = False, tx, ty
        delta_tx, delta_ty = tx-self.tx_old, ty-self.ty_old
        self.tx_old, self.ty_old = tx, ty
        Damping = 2
        if gesture==3:
            mx = mx_old + delta_tx*sx//(camx*Damping)
            my = my_old + delta_ty*sy//(camy*Damping)
            pyautogui.moveTo(mx,my, duration=0.1)
        elif gesture==0:
            if self.flag==0: pyautogui.doubleClick(); self.flag=1
        elif gesture==1: print("1 Finger Open")

# ------------------ Gesture Controller ------------------ #
class GestureController:
    gc_mode = 1
    cap = None

    cam_width = 0
    cam_height = 0

    aru_marker = Marker()
    hand_roi = ROI(2.5,2.5,6,0.45,0.6,0.4)
    glove = Glove()
    mouse = Mouse()

    def __init__(self):
        GestureController.cap = cv2.VideoCapture(0)
        if GestureController.cap.isOpened():
            GestureController.cam_width = int(GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            GestureController.cam_height = int(GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            raise Exception("Cannot open camera")

    def start(self):
        fps = 30
        while GestureController.gc_mode:
            start_time = time.time()
            ret, frame = GestureController.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)

            # Marker detection
            GestureController.aru_marker.detect(frame)
            if GestureController.aru_marker.is_detected():
                self.hand_roi.findROI(frame, self.aru_marker)
                self.hand_roi.find_glove_hsv(frame, self.aru_marker)
                FinalMask = self.hand_roi.cropROI(frame)
                self.glove.find_fingers(FinalMask)
                self.glove.find_gesture(frame)
                self.mouse.move_mouse(frame, self.hand_roi.marker_top, self.glove.gesture)

            # Draw overlays
            if GestureController.aru_marker.is_detected():
                self.aru_marker.draw_marker(frame)
                draw_box(frame, self.hand_roi.roi_corners, (255,0,0))
                draw_box(frame, self.hand_roi.hsv_corners, (0,0,255))
                cv2.imshow("FinalMask", FinalMask)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            elapsed = time.time()-start_time
            time.sleep(max(1.0/fps - elapsed,0))

        GestureController.cap.release()
        cv2.destroyAllWindows()

# ------------------ Run ------------------ #
if __name__ == "__main__":
    gc = GestureController()
    gc.start()
