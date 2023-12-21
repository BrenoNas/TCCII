import airsim
import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces
from airgym.envs.airsim_env import AirSimEnv
from collections import OrderedDict

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)
        # Environment initialization and attributes
        self.forward_speed_1 = 0.5
        self.forward_speed_2 = 1.0
        self.yaw_rate = 120 * 0.5
        self.rotation_duration = 0.1

        self.info = {}
        self.info["log"] = "None"
        self.debug = False

        self.image_shape = image_shape
        self.step_count = 0
        self.episode_count = 0
        self.state = {
            "position": np.zeros(3),
            "collision": False,
        }

        obs_space = {
            "image" : gym.spaces.Box(low=0, high=255, shape=(image_shape), dtype=np.uint8),
            "vector" : gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(8,), dtype=float)
        }

        self.observation_space =  gym.spaces.Dict(obs_space)

        self.action_space = spaces.Discrete(6)
        
        self.drone = airsim.MultirotorClient(ip=ip_address)

        self.images = self.setupInitalState()
    
        self.previous_images = self.images

        self.target_x, self.target_y, self.target_z = self.drone.simGetObjectPose("Target").position

        self.setupFlight()

        self.image_request = airsim.ImageRequest(
            0, airsim.ImageType.DepthPerspective, True, False
        )

        self.last_image = None

    def setupInitalState(self):
        # Returns a sequence of four depth images
        d=[]
        for i in range(4):
            d.append(self.getDepthImage())
        
        d=np.stack(d)
        return d
    
    def step(self, action):
        # The main execution of the environment occurs in this role
        # Agent executes the received action
        self.takeAction(action)
        self.step_count += 1

        # Collects the new state of the environment
        obs_img, env_state = self.getObservation()
        self.updateStackedImages(obs_img)

        # Calculate the reward and check the end of the episode
        reward, done = self.computeReward()

        complete_state = OrderedDict()
        complete_state["image"] = self.images
        complete_state["vector"] = env_state
        return complete_state, reward, done, False, self.info
        
    def takeAction(self, action):
        if action == 0:
            self.straight(self.forward_speed_2, 1)
        if action == 1:
            self.straight(self.forward_speed_1, 1)
        elif action == 2:
            self.yaw_right(self.yaw_rate, self.rotation_duration)
        elif action == 3:
            self.yaw_right(-1*self.yaw_rate, self.rotation_duration)
        elif action == 4:
            self.up(-1*self.forward_speed_2, 1)
        elif action == 5:
            self.up(self.forward_speed_2, 1)

    def getObservation(self):
        image = self.getDepthImage()
        self.last_image = image
        self.target_x, self.target_y, self.target_z = self.drone.simGetObjectPose("Target").position
        self.drone_state = self.drone.getMultirotorState()
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        collision = self.drone.simGetCollisionInfo().penetration_depth > 0.0
        self.state["collision"] = collision
        pose = self.drone.simGetVehiclePose()
        pitch, roll, yaw  = airsim.to_eularian_angles(pose.orientation)

        env_state = [
                    self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val,
                    
                    self.target_x - self.state["position"].x_val, self.target_y - self.state["position"].y_val, self.target_z - self.state["position"].z_val,
                    yaw,
                    self.dotProductTarget()
                    ]

        return image, env_state

    def computeReward(self):
        self.info["log"] = "None"
        done = 0
        
        # Distance reward calculation
        distance_now = self.calculeDistanceToTarget()
        if self.step_count <= 1:
            self.distanceBefore = distance_now
        reward_distance = -1 + max(0, (10 * (self.distanceBefore - distance_now)))

        # Calculation and sum of the extra reward
        dot = self.dotProductTarget()
        speed = np.sqrt((self.state["velocity"].x_val ** 2) + (self.state["velocity"].y_val ** 2))
        if (dot >= 0):
            extra_reward = speed * math.pow(math.e, dot)
            reward_distance += extra_reward
        
        # Checking completion conditions
        if distance_now <= 2.5:
            self.info["log"] = "goal"
            print(self.info["log"])
            done = 1
            reward = 100
        else:
            reward = reward_distance

        if self.step_count >= 512 and not done:
            self.info["log"] = "exceeded"
            print(self.info["log"])
            done = 1
            reward = -100

        elif self.state["collision"] and not done:
            self.info["log"] = "collision"
            print(self.info["log"])
            done = 1
            reward = -100

        self.distanceBefore = distance_now
        return reward, done

        
    def calculeDistanceToTarget(self):
        drone_x, drone_y, _ = self.state["position"]
        return np.sqrt(np.power(self.target_x - drone_x, 2) + np.power(self.target_y - drone_y, 2))
    
    def dotProductTarget(self):
        # Suppose that orientation_drone is a quaternion
        # and target_position is a vector [x, y, z]
        drone_x, drone_y, _ = self.state["position"]
        # Calculates the direction of the vector towards the goal
        direction_to_target = [self.target_x - drone_x, self.target_y - drone_y]

        # Normalize the vector
        norm_direction_to_target = math.sqrt(sum(x**2 for x in direction_to_target))
        direction_to_target = [x / norm_direction_to_target for x in direction_to_target]

        pose = self.drone.simGetVehiclePose()
        roll, pitch, yaw = airsim.to_eularian_angles(pose.orientation)

        # Calculates the forward direction of the drone (e.g. along the x-axis)
        forward_direction = [math.cos(yaw), math.sin(yaw)]

        # Normalize the vector
        norm_forward_direction = math.sqrt(sum(x**2 for x in forward_direction))
        forward_direction = [x / norm_forward_direction for x in forward_direction]

        # Calculate the dot product between the direction_to target and forward direction vectors
        dot_product = sum(a * b for a, b in zip(direction_to_target, forward_direction))

        # The dot product is used as a measure of how aligned the drone is with the direction of the objective 
        # Closer to 1 means that is align to objective
        return dot_product

    def setupFlight(self):
        if (self.info["log"] and self.info["log"] != "goal" and self.info["log"] != "collision") or not self.debug:
            # Reset the drone to its initial state
            self.drone.reset()
            self.drone.enableApiControl(True)
            self.drone.armDisarm(True)
            
            result = None
            while result == None:
                future = self.drone.takeoffAsync()
                future.join()
                result = future.result
            
            self.target_x, self.target_y, self.target_z = self.drone.simGetObjectPose("Target").position
        
        
    def getDepthImage(self):
        try:
            responses = self.drone.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        except:
            responses = None
            
        if (responses == None):
            print("Image size:" + str(responses[0].height) + "," + str(responses[0].width))
            img = [np.array([0]) for _ in responses]
        else:
            img = []
            for res in responses:
                img.append(np.array(res.image_data_float, dtype=np.float32))
            img = np.stack(img, axis=0)

        # image pre processing
        img = img.clip(max=20)
        scale=255/20
        img=img*scale

        img2d=[]
        for i in range(len(responses)):
            if ((responses[i].width != 0 or responses[i].height != 0)):
                img2d.append(np.reshape(img[i], (responses[i].height, responses[i].width)))
            else:
                img2d.append(self.last_img[i])

        self.last_img = np.stack(img2d, axis=0)

        if len(img2d)>1:
            return img2d
        else:
            return img2d[0]

    def straight(self, speed, duration):
        pose = self.drone.simGetVehiclePose()
        _, _, yaw  = airsim.to_eularian_angles(pose.orientation)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.drone.moveByVelocityZAsync(vx, vy, pose.position.z_val, duration, 1, yaw_mode).join()

    def up(self, speed, duration):
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.drone.moveByVelocityAsync(0, 0, speed, duration, 1, yaw_mode).join()

    def yaw_right(self, rate, duration):
        self.drone.rotateByYawRateAsync(rate, duration).join()
        start = time.time()
        return start, duration

    def updateStackedImages(self, obs_img):
        self.images = []
        for i in range(4):
           if i <3:
               self.images.append(self.previous_images[i+1])
           else:
               self.images.append(obs_img)
        self.previous_images = self.images

    def reset(self, seed=None, options=None):
        self.setupFlight()

        self.step_count = 0
        self.episode_count += 1
        obs_img, env_state = self.getObservation()
        self.updateStackedImages(obs_img)

        complete_state = OrderedDict()
        complete_state["image"] = self.images
        complete_state["vector"] = env_state
        self.state["collision"] = False
        return complete_state, self.info
    
    def __del__(self):
        self.drone.reset()