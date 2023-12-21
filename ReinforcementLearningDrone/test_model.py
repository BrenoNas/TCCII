import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

gym.register(
    id='airgym-v2',
    entry_point='airgym.envs.drone_env:AirSimDroneEnv', 
)

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make("airgym-v2", ip_address="127.0.0."+str(rank+1), image_shape=(4, 100, 100))
        env.reset(seed=rank+seed)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    num_cpu = 1 
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    model = PPO.load("saved_policy/best_model", env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    
    i = 0
    goals = 0
    fails = 0
    exceeded = 0
    collision = 0
    Ep = 50
    while True:
        if i == Ep:
            break
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        if dones:
            i+=1
        if (info[0]["log"]):
            if info[0]["log"] == "goal":
                goals += 1
            if info[0]["log"] == "exceeded" or info[0]["log"] == "collision":
                fails += 1
            if info[0]["log"] == "exceeded":
                exceeded += 1
            if info[0]["log"] == "collision":
                collision += 1
    
    print("Collisions", collision)
    print("Exceeded", exceeded)
    print((100 * (Ep - fails))/Ep)


