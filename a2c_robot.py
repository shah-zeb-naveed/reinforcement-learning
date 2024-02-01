import os
import gymnasium as gym
import panda_gym
from pyvirtualdisplay import Display
from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from huggingface_hub import notebook_login

def setup_virtual_display():
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

def create_environment(env_id):
    return gym.make(env_id)

def print_observation_space(env):
    s_size = env.observation_space.shape
    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print("Sample observation", env.observation_space.sample())

def print_action_space(env):
    a_size = env.action_space
    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Action Space Sample", env.action_space.sample())

def make_vec_env_with_normalize(env_id, n_envs=4):
    env = make_vec_env(env_id, n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return env

def train_model(env, model_name, num_steps=1_000_000):
    model = A2C(policy="MultiInputPolicy", env=env, verbose=1)
    model.learn(num_steps)
    model.save(model_name)
    env.save("vec_normalize.pkl")
    return model

def evaluate_model(model, eval_env):
    mean_reward, std_reward = evaluate_policy(model, eval_env)
    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

def main():
    setup_virtual_display()

    env_id = "PandaReachDense-v3"
    env = create_environment(env_id)

    print_observation_space(env)
    print_action_space(env)

    env = make_vec_env_with_normalize(env_id, n_envs=4)

    model = train_model(env, "a2c-PandaReachDense-v3")

    eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
    eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

    eval_env.render_mode = "rgb_array"
    eval_env.training = False
    eval_env.norm_reward = False

    evaluate_model(model, eval_env)


    package_to_hub(
        model=model,
        model_name=f"a2c-{env_id}",
        model_architecture="A2C",
        env_id=env_id,
        eval_env=eval_env,
        repo_id=f"shahzebnaveed/a2c-{env_id}",
        commit_message="Initial commit",
    )


if __name__ == "__main__":
    main()
