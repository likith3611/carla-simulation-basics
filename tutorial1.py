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
#from keras import backend
from collections import deque
from requests import session
#from tensorflow.python.keras.models import Model
#from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
#from tensorflow.python.keras.callbacks import TensorBoard
#from keras.applications.xception import Xception
#from tensorflow.python.keras.models import Xception
#from keras.layers import Dense, GlobalAveragePooling2D
#from keras.optimizers import Adam
#from tensorflow.python.keras.optimizer_v1 import Adam
#from keras.models import Model
#from keras.callbacks import TensorBoard
import tensorflow as tf
#import keras.backend as backend
#import tensorflow.python.keras.backend as backend
from threading import Thread
from tqdm import tqdm
#from tensorflow.python.keras import backend
from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model
tf.compat.v1.disable_eager_execution()

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
MEMORY_FRACTION = 0.7
MIN_REWARD = -200
DISCOUNT = 0.99
EPISODES = 50
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
epsilon = 1
actor_list=[]

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class ModifiedTensorBoard(tf.compat.v1.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
    def set_model(self, model):
        pass
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
    def on_batch_end(self, batch, logs=None):
        pass
    def on_train_end(self, _):
        pass
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class CarlEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    
    def __init__(self):
        self.client=carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world= self.client.get_world()
        self.blueprint_library= self.world.get_blueprint_library()
        self.model_3=self.blueprint_library.filter('model3')[0]
    
    def reset(self):
        self.collision_hist = []
        self.vehicle = None
        i=0
        self.actor_list=[]
        #self.transform = random.choice(self.world.get_map().get_spawn_points())
        #self.transform = carla.Transform(carla.Location(x=199.419632, y=-5.502129, z=0.0009), carla.Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000))
        while(self.vehicle == None):
            self.transform = random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.transform)
            i+=1
            print(i)
        self.actor_list.append(self.vehicle)
        print(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam.set_attribute("fov", "110")
        self.cam_transform= carla.Transform(carla.Location(x=2.5, z=0.7))
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, self.cam_transform, attach_to= self.vehicle)
        self.actor_list.append(self.cam_sensor)
        self.cam_sensor.listen(lambda data: self.camera_sensor(data))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor=self.blueprint_library.find('sensor.other.collision')
        self.colsensor= self.world.spawn_actor(colsensor, self.cam_transform, attach_to=self.vehicle)
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
        #img = cv2.imread('1.jpg', 1)
        # path = '/home/lightyagami/carla-simulation-basics/models'
        # cv2.imwrite(os.path.join(path , 'waka.jpg'), image)
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
        velocity_kmph=int(3.6*math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
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
        # sess = tf.compat.v1.Session()
        # sess.run(tf.compat.v1.global_variables_initializer())
        # sess.run(tf.compat.v1.initialize_all_variables())
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()
        # self.session = keras.backend.get_session()
        # self.init = tf.global_variables_initializer()
        # self.session.run(self.init)
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
    def create_model(self):
        #tf.compat.v1.disable_v2_behavior()
        #init_op = tf.compat.v1.initialize_all_variables()
        #init_op=tf.compat.v1.global_variables_initializer()
        
        # #tf.compat.v1.global_variables_initializer()
        #sess.run(tf.compat.v1.global_variables_initializer())
        
        # tf.compat.v1.keras.backend.set_session(sess)
        #base_model = Xception(weights=None, include_top = False, input_shape=(IM_HEIGHT,IM_WIDTH,3))
        base_model = tf.compat.v1.keras.applications.xception.Xception(weights=None, include_top = False, input_shape=(IM_HEIGHT,IM_WIDTH,3))
        
        x = base_model.output
        x=tf.compat.v1.keras.layers.GlobalAveragePooling2D()(x)
        y=tf.compat.v1.keras.optimizers.SGD()
        y.learning_rate=0.001
        # with tf.compat.v1.Session() as session:
        #     session.run(tf.compat.v1.tables_initializer)
        tf.compat.v1.keras.backend.set_session(self.sess)
        predictions = tf.compat.v1.keras.layers.Dense(3, activation= "linear")(x)
        model = tf.compat.v1.keras.Model(inputs=base_model.input, outputs = predictions)
        model.compile(loss="mse", optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy']) 
        
        return model
    def update_replay_memory(self,transition):
        #transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)
    def train(self):
        #sess = tf.compat.v1.Session()
        if(len(self.replay_memory)< MIN_REPLAY_MEMORY_SIZE):
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            # sess.run(tf.compat.v1.global_variables_initializer())
            # tf.compat.v1.keras.backend.set_session(sess)
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            # sess.run(tf.compat.v1.global_variables_initializer())
            # tf.compat.v1.keras.backend.set_session(sess)
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)
        X=[]
        y=[]
        for i, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if (not done):
                max_future_q= np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[i]
            current_qs[action] = new_q
            X.append(current_state) 
            y.append(current_qs)
        log_this_step = False
        if(self.tensorboard.step > self.last_logged_episode):
            log_this_step= True
            self.last_log_episode = self.tensorboard.step
        with self.graph.as_default():
            # sess.run(init_op)
            # tf.compat.v1.keras.backend.set_session(sess)
            tf.compat.v1.keras.backend.set_session(self.sess)
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose = 1, shuffle = False, callbacks = [self.tensorboard] if(log_this_step) else None)
        if(log_this_step):
            self.target_update_counter +=1
        if(self.target_update_counter > UPDATE_TARGET_EVERY):
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter=0
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]
    def train_in_loop(self):
        # sess = tf.compat.v1.Session()
        # gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
        #sess = tf.compat.v1.keras.backend.get_session()
        print("hello")

        X = np.random.uniform(size = (1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1,3)).astype(np.float32)
        with self.graph.as_default():
            # self.sess = tf.compat.v1.Session()
            # self.sess.run(tf.compat.v1.global_variables_initializer())
            # sess= tf.compat.v1.keras.backend.get_session()
            
            #tf.compat.v1.disable_v2_behavior()
            
            #sess.run(tf.compat.v1.global_variables_initializer())
            #tf.compat.v1.keras.backend.set_session(swe)
            tf.compat.v1.keras.backend.set_session(self.sess)
            #sesssion_1=tf.compat.v1.keras.backend.get_session()
            # tf.compat.v1.Session().run(tf.compat.v1.global_variables_initializer())
            self.model.fit(X,y, verbose= False, batch_size =1)
        self.training_initialized = True

        while True:
            print("hello")
            if(self.terminate):
                return
            self.train()
            time.sleep(0.01)

if(__name__=="__main__"):
    FPS=40
    ep_rewards = [-200]       
    #tf.compat.v1.disable_eager_execution() 
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    tf.compat.v1.disable_v2_behavior()

    #gpu_options = tf.compact.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    #backend.tf.compat.v1.keras.backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    # init_op = tf.compat.v1.global_variables_initializer()
    # sess = tf.compat.v1.Session()
    # sess.run(tf.compat.v1.global_variables_initializer())
    # tf.compat.v1.keras.backend.set_session(sess)
    # gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # x=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    # x.run(tf.compat.v1.global_variables_initializer())
    # tf.compat.v1.keras.backend.set_session(x)

    
    if(not os.path.isdir('models')):
        os.makedirs('models')

    agent = DQNAgent()
    env = CarlEnv()

    #tf.compat.v1.disable_v2_behavior()
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        #print("hello")
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height,env.im_width,3)))
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist=[]
        agent.tensorboard.step = episode
        episode_reward=0
        step=1
        current_state= env.reset()
        done= False
        episode_start=time.time() 
        
        while True:
            if (np.random.random()>epsilon):
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0,3)
                time.sleep(1/FPS)
            new_state, reward, done, _ =env.step(action)
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            step += 1

            if(done):
                break
        for j in range(len(env.actor_list)):
            #print(j)
            if("ActorBlueprint" in str(env.actor_list[j])):
                pass
            else:
                env.actor_list[j].destroy()
            print(env.actor_list)

        ep_rewards.append(episode_reward)
        if(not episode % AGGREGATE_STATS_EVERY or episode==1):
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon= epsilon)
            if(min_reward>=MIN_REWARD):

                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                pass
            if(epsilon > MIN_EPSILON):
                epsilon *=EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
    agent.terminate = True
    trainer_thread.join()
    # tf.compat.v1.keras.models.save_model
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')



# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass



# def camera_sensor(data):
#     i=np.array(data.raw_data)
#     i2= i.reshape((IM_HEIGHT,IM_WIDTH,4))
#     # print(i2)
#     image=i2[:, :, :3]
#     #if(self.SHOW_CAM):
#     cv2.imshow("carla_image", image)
#     cv2.waitKey(1)
#     return image/255.0


# # IM_WIDTH=640
# # IM_HEIGHT=480
# # actor_list=[]
# try:
#     client=carla.Client("localhost", 2000)
#     client.set_timeout(2.0)

#     world = client.get_world()
#     blueprint_library=world.get_blueprint_library()
#     bp=blueprint_library.filter('model3')[0]
#     spawn_point=random.choice(world.get_map().get_spawn_points())
#     #vehiclespawn=Transform(Location(x=230, y=195, z=40), Rotation(yaw=180))
#     vehicle = world.spawn_actor(bp,spawn_point)
#     vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
#     actor_list.append(vehicle)

#     cam_bp= blueprint_library.find('sensor.camera.rgb')
#     cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
#     cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
#     #cam_bp.set_attribute("fov","110")
#     spawn_point=carla.Transform(carla.Location(x=12.5,z=5.7))
#     sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
#     actor_list.append(sensor)
#     #sensor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))
#     sensor.listen(lambda data: camera_sensor(data))


#     time.sleep(20)


# finally:
#     for actor in actor_list:
#         actor.destroy()
#     print('All cleaned up')


