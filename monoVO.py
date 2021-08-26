import cv2 as cv
import numpy as np
import glob

kanade_lucas_parameters = dict(winSize  = (21, 21), criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

MatMinFeature = 1500

traj = np.zeros((800,800,3), dtype=np.uint8)

path = glob.glob("data_odometry_gray/dataset/sequences/00/image_0/*.png")
lst = []
for file in path:
    lst.append(file)
lst.sort()
arr = np.array(lst)

with open('data_odometry_gray/data_odometry_poses/dataset/poses/00.txt') as f:
	annotations = f.readlines()

def detect_keyPoints(frame):
	global key_points
	image = frame_1
	fast=cv.FastFeatureDetector_create(threshold=25,nonmaxSuppression=True,type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
	kp=fast.detect(image,None)
	blank = np.zeros((400,900,3), dtype='uint8')
	Keypoints = cv.drawKeypoints(blank,kp,blank,color=(0,255,0))
	key_points = np.array([x.pt for x in kp], dtype=np.float32)		
	#cv.imshow('Keypoints display Map',Keypoints)
	#print ("Total Keypoints with nonmaxSuppression: ", len(kp))
	return key_points

def featureTracking(frame_1, frame_2, key_points):
	global newpoints, nextPoints
	nextPoints, status, errors = cv.calcOpticalFlowPyrLK(frame_1, frame_2, key_points, None ,**kanade_lucas_parameters)
	status= status.reshape(status.shape[0])
	newpoints = key_points[status == 1]
	nextPoints = nextPoints[status == 1]
	return newpoints, nextPoints

def getAbsoluteScale(frame_id):  #specialized for KITTI odometry dataset
	ss = annotations[frame_id].strip().split()
	global prev_x, prev_y, prev_z
	prev_x = float(ss[3])
	prev_y = float(ss[7])
	prev_z = float(ss[11])
	ss = annotations[frame_id-1].strip().split()
	x = float(ss[3])
	y = float(ss[7])
	z = float(ss[11])
	trueX, trueY, trueZ = x, y, z
	global scale
	scale = np.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)
	return scale, prev_x, prev_y, prev_z

frame_1 = cv.imread(arr[0])
frame_2 = cv.imread(arr[1])
detect_keyPoints(frame_1)
featureTracking(frame_1, frame_2, key_points)
E, mask = cv.findEssentialMat(nextPoints, newpoints, focal=718.6560, pp=(607.1928,185.2157), method=cv.RANSAC, prob=0.99, threshold=1.0)
_, R, t, mask = cv.recoverPose(E, nextPoints, newpoints, focal=718.6560, pp=(607.1928,185.2157))
R_traj = R
t_traj = t
#print(R_traj)
#print(t_traj)
frame_1 = frame_2

for i in range(2,len(arr)-1):
    
    frame_2 = cv.imread(arr[i])
    featureTracking(frame_1, frame_2, key_points)
    #print(len(nextPoints))
    #print(len(newpoints))
    #print("\n\n")
    E, mask = cv.findEssentialMat(nextPoints, newpoints, focal=718.6560, pp=(607.1928,185.2157), method=cv.RANSAC, prob=0.99, threshold=1.0)
    #print(E)
    _, R, t, mask = cv.recoverPose(E, nextPoints, newpoints, focal=718.6560, pp=(607.1928,185.2157))
    #print(R)
    #print(t)

    getAbsoluteScale(i)
    #print(scale)
    t_traj = t_traj + scale*(np.dot(R_traj,t))
    #t_traj = t_traj + np.dot(R_traj,t)
    R_traj = np.dot(R,R_traj)

    if(key_points.shape[0] < MatMinFeature):
    	detect_keyPoints(frame_1)
    	featureTracking(frame_1, frame_2, key_points)

    #print(cur_T)
    key_points = nextPoints
    frame_1 = frame_2
    x, y, z = t_traj[0], t_traj[1], t_traj[2]
    x_1,y_1,z_1 = prev_x, prev_y, prev_z
    #print(i)
    print(x," ", z)
    true_x, true_y = int(x)+290, int(z)+90
    true_x_1, true_y_1 = int(x_1)+290, int(z_1)+90
    cv.circle(traj, (true_x,true_y), 2, (0,0,255), -1)
    cv.circle(traj, (true_x_1,true_y_1), 2, (0,255,0), -1)

    cv.imshow('Road facing camera', frame_2)
    cv.imshow('Trajectory', traj)



    cv.waitKey(10)