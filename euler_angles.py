import cv2
import numpy as np
import os
import dlib
import math
from loguru import logger
import glob
import tqdm
from multiprocessing import Process
import multiprocessing 
logger.add("data_mult2.log", rotation="500 MB") 
DEBUG = False

class PnpHeadPoseEstimator:
    """ Head pose estimation class which uses the OpenCV PnP algorithm.

        It finds Roll, Pitch and Yaw of the head given a figure as input.
        It uses the PnP algorithm and it requires the dlib library
    """

    def __init__(self,dlib_shape_predictor_file_path ,cam_w=640, cam_h=480 ):
        """ Init the class

        @param cam_w the camera width. If you are using a 640x480 resolution it is 640
        @param cam_h the camera height. If you are using a 640x480 resolution it is 480
        @dlib_shape_predictor_file_path path to the dlib file for shape prediction (look in: deepgaze/etc/dlib/shape_predictor_68_face_landmarks.dat)
        """
        # if(IS_DLIB_INSTALLED == False): raise ValueError('[DEEPGAZE] PnpHeadPoseEstimator: the dlib libray is not installed. Please install dlib if you want to use the PnpHeadPoseEstimator class.')

        if(os.path.isfile(dlib_shape_predictor_file_path)==False): raise ValueError('[DEEPGAZE] PnpHeadPoseEstimator: the files specified do not exist.')

        #Defining the camera matrix.
        #To have better result it is necessary to find the focal
        # lenght of the camera. fx/fy are the focal lengths (in pixels)
        # and cx/cy are the optical centres. These values can be obtained
        # roughly by approximation, for example in a 640x480 camera:
        # cx = 640/2 = 320
        # cy = 480/2 = 240
        # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
        c_x = cam_w / 2
        c_y = cam_h / 2
        f_x = c_x / np.tan(60/2 * np.pi / 180)
        f_y = f_x

        #Estimated camera matrix values.
        self.camera_matrix = np.float32([[f_x, 0.0, c_x],
                                         [0.0, f_y, c_y],
                                         [0.0, 0.0, 1.0] ])
        # K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
        #
        #      0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
        #
        #      0.0, 0.0, 1.0]
        # self.camera_matrix = np.array(K).reshape(3,3).astype(np.float32)
        #These are the camera matrix values estimated on my webcam with
        # the calibration code (see: src/calibration):
        #camera_matrix = np.float32([[602.10618226,          0.0, 320.27333589],
                                   #[         0.0, 603.55869786,  229.7537026],
                                   #[         0.0,          0.0,          1.0] ])

        #Distortion coefficients
        self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
        # self.camera_distortion = np.float32([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000])
        #Distortion coefficients estimated by calibration in my webcam
        #camera_distortion = np.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])

        if(DEBUG==True): print("[DEEPGAZE] PnpHeadPoseEstimator: estimated camera matrix: \n" + str(self.camera_matrix) + "\n")

        #Declaring the dlib shape predictor object
        self._detector = dlib.get_frontal_face_detector()
        self._shape_predictor = dlib.shape_predictor(dlib_shape_predictor_file_path)
        # self.landmarks = []


    def _return_landmarks(self, inputImg, points_to_return=range(0,68)):
        """ Return the the roll pitch and yaw angles associated with the input image.

        @param image It is a colour image. It must be >= 64 pixel.
        @param radians When True it returns the angle in radians, otherwise in degrees.
        """
        #Creating a dlib rectangle and finding the landmarks
        # dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
        dets = self._detector(inputImg)
        try:
            det = dets[0]
        except IndexError :
            print('no face detected')
            img_h, img_w, img_d = inputImg.shape
            det =  dlib.rectangle(left=0, top=0, right=int(img_w), bottom=int(img_h))
            return None
        dlib_landmarks = self._shape_predictor(inputImg, det)

        #It selects only the landmarks that
        #have been indicated in the input parameter "points_to_return".
        #It can be used in solvePnP() to estimate the 3D pose.
        landmarks = np.zeros((len(points_to_return),2), dtype=np.float32)
        counter = 0
        for point in points_to_return:
            landmarks[counter] = [dlib_landmarks.parts()[point].x, dlib_landmarks.parts()[point].y]
            counter += 1
        self.landmarks = landmarks
        return landmarks

    def return_pitch_yaw_roll(self, image, radians=False):
         """ Return the the roll pitch and yaw angles associated with the input image.

         @param image It is a colour image. It must be >= 64 pixel.
         @param radians When True it returns the angle in radians, otherwise in degrees.
         """

         #The dlib shape predictor returns 68 points, we are interested only in a few of those
         # TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
         TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

         #Antropometric constant values of the human head.
         #Check the wikipedia EN page and:
         #"Head-and-Face Anthropometric Survey of U.S. Respirator Users"
         #
         #X-Y-Z with X pointing forward and Y on the left and Z up.
         #The X-Y-Z coordinates used are like the standard
         # coordinates of ROS (robotic operative system)
         #OpenCV uses the reference usually used in computer vision:
         #X points to the right, Y down, Z to the front
         #
         #The Male mean interpupillary distance is 64.7 mm (https://en.wikipedia.org/wiki/Interpupillary_distance)
         #
         # P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
         # P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
         # P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
         # P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
         # P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
         # P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
         # P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
         # P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27 This is the world origin
         # P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
         # P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
         # P3D_RIGHT_EYE = np.float32([-20.0, -32.35,-5.0]) #36
         # P3D_RIGHT_TEAR = np.float32([-10.0, -20.25,-5.0]) #39
         # P3D_LEFT_TEAR = np.float32([-10.0, 20.25,-5.0]) #42
         # P3D_LEFT_EYE = np.float32([-20.0, 32.35,-5.0]) #45
         # #P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
         # #P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
         # P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62
         #
         # #This matrix contains the 3D points of the
         # # 11 landmarks we want to find. It has been
         # # obtained from antrophometric measurement
         # # of the human head.
         # landmarks_3D = np.float32([P3D_RIGHT_SIDE,
         #                          P3D_GONION_RIGHT,
         #                          P3D_MENTON,
         #                          P3D_GONION_LEFT,
         #                          P3D_LEFT_SIDE,
         #                          P3D_FRONTAL_BREADTH_RIGHT,
         #                          P3D_FRONTAL_BREADTH_LEFT,
         #                          P3D_SELLION,
         #                          P3D_NOSE,
         #                          P3D_SUB_NOSE,
         #                          P3D_RIGHT_EYE,
         #                          P3D_RIGHT_TEAR,
         #                          P3D_LEFT_TEAR,
         #                          P3D_LEFT_EYE,
         #                          P3D_STOMION])
         LEFT_EYEBROW_LEFT = [6.825897, 6.760612, 4.402142]
         LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
         RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
         RIGHT_EYEBROW_RIGHT = [-6.825897, 6.760612, 4.402142]
         LEFT_EYE_LEFT = [5.311432, 5.485328, 3.987654]
         LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
         RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
         RIGHT_EYE_RIGHT = [-5.311432, 5.485328, 3.987654]
         NOSE_LEFT = [2.005628, 1.409845, 6.165652]
         NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
         MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
         MOUTH_RIGHT = [-2.774015, -2.080775, 5.048531]
         LOWER_LIP = [0.000000, -3.116408, 6.097667]
         CHIN = [0.000000, -7.415691, 4.070434]
         landmarks_3D = np.float32([LEFT_EYEBROW_LEFT,
                                    LEFT_EYEBROW_RIGHT,
                                    RIGHT_EYEBROW_LEFT,
                                    RIGHT_EYEBROW_RIGHT,
                                    LEFT_EYE_LEFT,
                                    LEFT_EYE_RIGHT,
                                    RIGHT_EYEBROW_LEFT,
                                    RIGHT_EYEBROW_RIGHT,
                                    NOSE_LEFT,
                                    NOSE_RIGHT,
                                    MOUTH_LEFT,
                                    MOUTH_RIGHT,
                                    LOWER_LIP,
                                    CHIN])
         #Return the 2D position of our landmarks
         landmarks_2D = self._return_landmarks(inputImg=image, points_to_return=TRACKED_POINTS)
         if landmarks_2D is not None :
             #Print som red dots on the image
             #for point in landmarks_2D:
                 #cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)


             #Applying the PnP solver to find the 3D pose
             #of the head from the 2D position of the
             #landmarks.
             #retval - bool
             #rvec - Output rotation vector that, together with tvec, brings
             #points from the world coordinate system to the camera coordinate system.
             #tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
             retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                               landmarks_2D,
                                               self.camera_matrix,
                                               self.camera_distortion)
             #Get as input the rotational vector
             #Return a rotational matrix
             rmat, _ = cv2.Rodrigues(rvec)
             pose_mat = cv2.hconcat((rmat,tvec))
             #euler_angles contain (pitch, yaw, roll)
             # euler_angles = cv2.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=self.camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)
             _, _, _, _, _, _,euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
             return list(euler_angles)

             head_pose = [ rmat[0,0], rmat[0,1], rmat[0,2], tvec[0],
                           rmat[1,0], rmat[1,1], rmat[1,2], tvec[1],
                           rmat[2,0], rmat[2,1], rmat[2,2], tvec[2],
                                 0.0,      0.0,        0.0,    1.0 ]
             #print(head_pose) #TODO remove this line
             return self.rotationMatrixToEulerAngles(rmat)
         else:return None



    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R) :

        #assert(isRotationMatrix(R))

        #To prevent the Gimbal Lock it is possible to use
        #a threshold of 1e-6 for discrimination
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

def get_pitch_yaw_roll(img_dir,estimator,output_path):
    # img_lst = os.listdir(img_dir)
    temp_path   = os.path.join(img_dir,'*/')
    pathes      = glob.glob(temp_path)
    pathes.sort()
    # logger.debug(f"pathes:{pathes}")
    dataset = []
    for dir_item in pathes:
        join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
        temp_list = []
        for item in join_path:
            temp_list.append(item)
        dataset.append(temp_list)
    dataset.sort()
    with open('data.txt',"w+") as fw:
        for dir_lst in tqdm.tqdm(dataset):
            for img_file_path in dir_lst:
                # img_file_path = os.path.join(img_dir,img_file_name)
                # logger.debug(f"img_file_path:{img_file_path}")
                frame = cv2.imread(img_file_path)
                pitch_yaw_roll = estimator.return_pitch_yaw_roll(frame)
                if pitch_yaw_roll is not None :
                    # type(pitch_yaw_roll) array[array]
                    pitch,yaw,roll =map(lambda x:x[0],pitch_yaw_roll)
                    fw.write(f"{img_file_path},{pitch},{yaw},{roll} \n")
                    logger.debug(f"img_file_path,pitch,yaw,roll:{img_file_path},{pitch},{yaw},{roll}")
                    # 进行图片标记
                    # cv2.putText(frame, 'pitch:{:+.2f}'.format(pitch),(0,20),cv2.FONT_HERSHEY_PLAIN,1, (0,0,255), 1 )
                    # cv2.putText(frame, 'yaw:{:+.2f}'.format(yaw), (0, 35),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    # cv2.putText(frame, 'roll:{:+.2f}'.format(roll), (0, 50),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    # landmarks = estimator.landmarks
                    # if landmarks is not None:
                    #     # logger.debug(f"landmarks:{landmarks.shape}")
                    #     for i in range(landmarks.shape[0]):
                    #         x,y = landmarks[i]
                    #         cv2.circle(frame,(int(x),int(y)),1,(0,255,0))
                    #     cv2.imwrite(os.path.join(output_path ,img_file_name),frame)


def get_pitch_yaw_roll_mult(i,dataset,estimator):
    with open(f'./data_{i}.txt',"w+") as fw:
        for dir_lst in tqdm.tqdm(dataset):
            # logger.debug(f"dir_lst:{dir_lst}")
            for img_file_path in dir_lst:
                frame = cv2.imread(img_file_path)
                pitch_yaw_roll = estimator.return_pitch_yaw_roll(frame)
                if pitch_yaw_roll is not None :
                    pitch,yaw,roll =map(lambda x:x[0],pitch_yaw_roll)
                    fw.write(f"{img_file_path},{pitch},{yaw},{roll} \n")
                    logger.info(f"img_file_path,pitch,yaw,roll:{img_file_path},{pitch}，{yaw}，{roll}")
                    

def start_process(img_dir,estimator):
    processes = multiprocessing.cpu_count() // 2
    temp_path   = os.path.join(img_dir,'*/')
    pathes      = glob.glob(temp_path)
    pathes.sort()
    # logger.debug(f"pathes:{pathes}")
    dataset = []
    for dir_item in pathes:
        join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
        join_path += glob.glob(os.path.join(dir_item,'*.png'))
        join_path += glob.glob(os.path.join(dir_item,'*.JPG'))
        join_path += glob.glob(os.path.join(dir_item,'*.PNG'))
        temp_list = []
        for item in join_path:
            temp_list.append(item)
        dataset.append(temp_list)
    dataset.sort()
    len_data = len(dataset)
    unit_len = len_data // processes
    logger.debug(f"len_data:{len_data},unit_len:{unit_len},processes:{processes}")
    p_lst = []
    for i in range(processes):
        logger.debug(f"i*unit_len:{i*unit_len},unit_len*(i+1):{unit_len*(i+1)}")
        p = Process(target=get_pitch_yaw_roll_mult,args=(i,dataset[(i*unit_len):unit_len*(i+1)],estimator) )
        p_lst.append(p)
        p.start()
    for p in p_lst:
        p.join()

def get_data(data_path):
    with open(data_path,'r') as rf:
        rf.read()

    ...
if __name__ == '__main__':
    estimator = PnpHeadPoseEstimator('./shape_predictor_68_face_landmarks.dat')
    print(f"estimator:{estimator}")
    
    img_dir = '/Users/chinasilva/face_dataset/all'
    start_process(img_dir,estimator)
    






