import glob
import os
import sys
import random
import time
import cv2
import numpy as np
#import Transform
import carla
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D


SHOW_PREVIEW=False
IM_HEIGHT=480
IM_WIDTH=640
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE //4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = 'Xception'
MEMORY_FRACTION = 0.8
MIN_REWARD = -200
DISCOUNT = 0.99
EPISODES = 100
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
actor_list=[]
class CarlEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    
    def __init__(self):
        self.client=carla.client("localhost",2000)
        self.client.set_timeout(5.0)
        self.world= self.client.get_world()
        self.blueprint_library= self.world.get_blueprint_library()
        self.model_3=self.blueprint_library.filter('model3')[0]
    
    def reset(self):
        self.collision_hist = []
        self.actor_list=[]
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attributes("image_size,x", f"{IM_WIDTH}")
        self.rgb_cam.set_attributes("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam.set_attributes("fov", "110")
        self.cam_transform= carla.Transform(carla.Location(x=2.5, z=0.7))
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, self.cam_transform, attach= self.vehicle)
        self.actor_list.append(self.cam_sensor)
        self.cam_sensor.listen(lambda data: self.camera_sensor(data))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor=self.blueprint_library.find('sensor.other.collision')
        self.colsensor= self.world.spawn_actor(colsensor, self.cam_transform, attach=self.vehicle)
        self.actor_list.append(colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        return self.front_camera
    
    def collision_data(self,event):
        self.collision_hist.append(event)

    def camera_sensor(self,data):
        i=np.array(data.raw_data)
        i2= i.reshape((480,640,4))
        image=i2[:, :, :3]
        if(self.SHOW_CAM):
            cv2.imshow("carla image", image)
            cv2.waitKey(1)
        self.front_camera = image
        
    def step(self,action):
        if(action==0):
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif(action==1):
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0*self.STEER_AMT))
        elif(action==2):
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
        velocity= self.vehicle.get_velocity()
        velocity_kmph=int(3.6*math.sqrt(velocity.x**2, velocity.y**2, velocity.z**2))
        if(len(self.collision_hist)!=0):
            done=True
            reward = -200
        elif(velocity_kmph <= 50):
            done = False
            reward = -1
        else:
            done= False
            reward = 1
        if(self.episode_start+ SECONDS_PER_EPISODE < time.time()):
            done = True
        return self.front_camera, reward, done, None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard= ModifiedTensorBoard(Log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()
        self.terminate = false
        self.last_logged_episode = 0
        self.training_initialized = False
    def create_model():
        base_model = Xception(weights=None, include_top = False, input_shape(IM_HEIGHT,IM_WIDTH,3))
        x= base_model.output
        x=GlobalAveragePooling2D()(x)
        predictions = Dense(3, activation= "linear")(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(loss="mse", optimizer =Adam(lr = 0.001), metrics=['accuracy'])
        return model


# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass






# IM_WIDTH=640
# IM_HEIGHT=480
# actor_list=[]
# try:
#     client=carla.Client("localhost", 2000)
#     client.set_timeout(2.0)

#     world = client.get_world()
#     blueprint_library=world.get_blueprint_library()
#     bp=blueprint_library.filter('model3')[0]
#     print(bp)
#     spawn_point=random.choice(world.get_map().get_spawn_points())
#     #vehiclespawn=Transform(Location(x=230, y=195, z=40), Rotation(yaw=180))
#     vehicle = world.spawn_actor(bp,spawn_point)
#     vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
#     actor_list.append(vehicle)

#     cam_bp= blueprint_library.find('sensor.camera.rgb')
#     cam_bp.set_attribute("image_size_x","640")
#     cam_bp.set_attribute("image_size_y","480")
#     #cam_bp.set_attribute("fov","110")
#     spawn_point=carla.Transform(carla.Location(x=2.5,z=0.7))
#     sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
#     actor_list.append(sensor)
#     #sensor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))
#     sensor.listen(lambda data: camera_sensor(data))


#     time.sleep(10)


# finally:
#     for actor in actor_list:
#         actor.destroy()
#     print('All cleaned up')


