# 5552 Localisation and Navigation Project

from controller import Supervisor
from controller import CameraRecognitionObject
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

wheelRadius = 0.0205
axleLength = 0.0568 # Data from Webots website seems wrong. The real axle length should be about 56-57mm
updateFreq = 10 # update every 200 timesteps
plotFreq = 100 # plot every 50 time steps



# helper functions
def omegaToWheelSpeeds(omega, v):
    wd = omega * axleLength * 0.5
    return v - wd, v + wd

def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def plot_cov(cov_mat, prob=0.95, num_pts=50):
    conf = chi2.ppf(0.95, df=2)
    L, V = np.linalg.eig(cov_mat)
    s1 = np.sqrt(conf*L[0])
    s2 = np.sqrt(conf*L[1])
    
    thetas = np.linspace(0, 2*np.pi, num_pts)
    xs = np.cos(thetas)
    ys = np.sin(thetas)

    standard_norm = np.vstack((xs, ys))
    S = np.array([[s1, 0],[0, s2]])
    scaled = np.matmul(S, standard_norm)
    R= V
    rotated = np.matmul(R, scaled)
    
    return(rotated)

# Task: Finish EKFPropagate and EKFRelPosUpdate
def EKFPropagate(x_hat_t, # robot position and orientation
                 Sigma_x_t, # estimation uncertainty
                 u, # control signals
                 Sigma_n, # uncertainty in control signals
                 dt # timestep
    ):
    # TODO: Calculate the robot state estimation and variance for the next timestep
    x_hat_t[0] = x_hat_t[0] + dt * u[0] * np.cos(x_hat_t[2])
    x_hat_t[1] = x_hat_t[1] + dt * u[0] * np.sin(x_hat_t[2])
    x_hat_t[2] = x_hat_t[2] + dt * u[1]

    # Jacobian for State
    phi = [[1, 0, -dt * u[0] * np.sin(x_hat_t[2])], 
           [0, 1, dt * u[0] * np.cos(x_hat_t[2])], 
           [0, 0, 1]]
           
    # Jacobian of Noise
    G = [[dt * np.cos(x_hat_t[2]), dt * np.sin(x_hat_t[2]), 0], [0, 0, dt]]
    phi, G = np.array(phi), np.array(G).T
    
    # Calculate P_k+1|k
    Sigma_x_t = phi @ Sigma_x_t @ phi.T + G @ Sigma_n @ G.T
    
    return x_hat_t, Sigma_x_t

def EKFRelPosUpdate(x_hat_t, # robot position and orientation
                    Sigma_x_t, # estimation uncertainty
                    z, # measurements
                    Sigma_m, # measurements' uncertainty
                    G_p_L, # landmarks' global positions
                    dt # timestep
                   ):
    # TODO: Update the robot state estimation and variance based on the received measurement
    x_hat_t, Sigma_x_t = np.array(x_hat_t).T, np.array(Sigma_x_t)
    z, Sigma_m, G_p_L = np.array(z).T, np.array(Sigma_m), np.array(G_p_L)
    
    # rotMat messes it up?
    C = np.array([[np.cos(x_hat_t[2]), -np.sin(x_hat_t[2])], 
                  [np.sin(x_hat_t[2]), np.cos(x_hat_t[2])]])

    z_hat = C.T @ (G_p_L[:2] - x_hat_t[:2])
    r = z - z_hat

    H1 = -C.T
    J = np.array([[0, -1], [1, 0]])
    H2 = -C.T @ J @ (G_p_L[:2] - x_hat_t[:2])
    H = np.append(H1, np.array([H2]).T, axis=1)
    
    S = H @ Sigma_x_t @ H.T + Sigma_m

    K = Sigma_x_t @ H.T @ np.linalg.inv(S)

    x_hat_t += K @ r
    
    Sigma_x_t -= Sigma_x_t @ H.T @ np.linalg.inv(S) @ H @ Sigma_x_t
    
    return x_hat_t, Sigma_x_t

# create the Robot instance.
robot = Supervisor()

timestep = int(robot.getBasicTimeStep())
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

camera = robot.getDevice('camera')
camera.enable(1)

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

if camera.hasRecognition():
    camera.recognitionEnable(1)
    camera.enableRecognitionSegmentation()
else:
    print("Your camera does not have recognition")


timestep = int(robot.getBasicTimeStep())
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

robotNode = robot.getFromDef("e-puck")
G_p_R = robotNode.getPosition()
G_ori_R = robotNode.getOrientation()

# plot settings
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 10) 



# Variables for Navigation Algorithm

# goal position
g = [-2.89, 0, 0]
# initial position
x = [2.89, 0, 0]



# Values for EKF
dt = timestep / 1000.0
x_hat_t = np.array([2.89, 0, 0])
Sigma_x_t = np.zeros((3,3))
Sigma_x_t[0,0], Sigma_x_t[1,1], Sigma_x_t[2,2] = 0.01, 0.01, np.pi/90
Sigma_n = np.zeros((2,2))
std_n_v = 0.01
std_n_omega = np.pi/60
Sigma_n[0,0] = std_n_v * std_n_v
Sigma_n[1,1] = std_n_omega * std_n_omega
counter = 0
timer = 0

v = 0
w = 0

b = 0.3
L = 0.052
gain = 0.01




while robot.step(timestep) != -1:
    dt = timestep / 1000
    
    G_p_R = robotNode.getPosition()[:2]
    G_ori_R = np.array(robotNode.getOrientation())
    
    # Update robot position
    x = x_hat_t
    
    # Update goal orientation
    g[2] = np.arctan2((g[1] - x[1]), (g[0] - x[0]))
    
    # Velocity Update
    v = 0.25
    
    dists = lidar.getRangeImage()
    
    # Obtain Lidar index leading to goal
    angle = ((g[2] - x[2]) % (2*np.pi))
    beam = int(-512 * (angle / (2*np.pi)) % 512)
    theta = beam / 512 * 2 * np.pi 
    path = int((256 - 512 * (angle / (2*np.pi))) % 512)

    # If there is a path to goal, rotate fast
    if dists[path] > 0.7:
        gain = 0.1
    else:
        gain = 0.01
    
    # Turn counter clockwise towards goal
    if abs(theta) > 0.2:
        v = 0.02
        if theta > 0:
            w = gain * ((-(theta + np.pi) % (2*np.pi)) - np.pi)
        else:
            w = gain * (((theta + np.pi) % (2*np.pi)) - np.pi)
    else:
        w = 0


    # Avoid nearby obstacles
    # Increase angular velocity as closer to wall
    left, mid, right = int(512 * 3/8), int(512 / 2), int(512 * 5/8)
    if dists[left] < b and dists[left] < dists[right]:
        w = w - math.sqrt(abs(dists[left] - b))
        v = 0.01

    if dists[right] < b and dists[left] > dists[right]:
        w = w + math.sqrt(abs(dists[right] - b))
        v = 0.01
    

    # End Motion at Goal
    dist = math.sqrt((g[0] - x[0]) ** 2 + (g[1] - x[1]) ** 2)

    if dist < 0.2:
        plt.show()
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)
        print("WAAZAMMMMM!!! Goal Obtained")
        
        break
        
    # Add noise and set wheel velocities
    n_v = v + np.random.normal(0, std_n_v / 6)
    n_w = w + np.random.normal(0, std_n_omega / 6)
    l = (n_v - n_w * L * 0.5)
    r = (n_v + n_w * L * 0.5)
    leftMotor.setVelocity(min(l * 6.28 / (l + r + 1e-5), 6.28))
    rightMotor.setVelocity(min(r * 6.28 / (l + r + 1e-5), 6.28)) 
    
    # Set Controls
    omega = -n_v / 0.5
    u = np.array([n_v, omega])
    
    # EKF Propagation
    x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)
    
    real = np.array([G_p_R[0], G_p_R[1], np.arctan2(G_ori_R[3], G_ori_R[0])])
    # print("Error:", x_hat_t - real)

    # EKF Update
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    z_pos = np.zeros((recObjsNum, 2)) # relative position measurements   
    z_dis = np.zeros((recObjsNum, ))
    if counter % updateFreq == 0:
        for i in range(0, recObjsNum):
            landmark = robot.getFromId(recObjs[i].get_id())
            G_p_L = landmark.getPosition()
            rel_lm_trans = landmark.getPose(robotNode)

            std_m = 0.05
            Sigma_m = [[std_m*std_m, 0], [0,std_m*std_m]]
            z_pos[i] = [rel_lm_trans[3]+np.random.normal(0,std_m), rel_lm_trans[7]+np.random.normal(0,std_m)]                
            x_hat_t, Sigma_x_t = EKFRelPosUpdate(x_hat_t, Sigma_x_t, z_pos[i], Sigma_m, G_p_L, dt)
            z_dis[i] = np.sqrt(rel_lm_trans[3]**2 + rel_lm_trans[7]**2)+np.random.normal(0,std_m)
        
    counter = counter + 1

    if counter % plotFreq == 0:
        pts = plot_cov(Sigma_x_t[0:2,0:2])
        pts[0] += x_hat_t[0]
        pts[1] += x_hat_t[1]
        plt.scatter([pts[0,:]], [pts[1,:]])
        plt.scatter(x_hat_t[0],x_hat_t[1])
        plt.axis('equal')
    pass

# Enter here exit cleanup code.
