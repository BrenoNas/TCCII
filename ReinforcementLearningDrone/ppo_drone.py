import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from scripts.network2 import CustomCombinedExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from airgym.envs.drone_env import AirSimDroneEnv

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

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=[512, 256, 128],
        normalize_images = False,
    )
    
    model = PPO(
        'MultiInputPolicy', 
        env,
        batch_size=128,
        n_steps=1024,
        n_epochs=10,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
        tensorboard_log="./tb_logs/",
        policy_kwargs=policy_kwargs,
    )

    # Evaluation callback
    callbacks = []

    checkpoint = CheckpointCallback(save_freq= 10000, save_path='model')

    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        best_model_save_path="saved_policy",
        log_path=".",
        eval_freq=10000,
    )

    callbacks.append(eval_callback)
    callbacks.append(checkpoint)

    kwargs = {}
    kwargs["callback"] = callbacks

    log_name = "ppo_run_" + str(time.time())
    model.learn(
        total_timesteps=1000000,
        tb_log_name=log_name,
        **kwargs,
        reset_num_timesteps=False
    )