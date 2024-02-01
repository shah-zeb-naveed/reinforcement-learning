import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from pyvirtualdisplay import Display
from huggingface_sb3 import package_to_hub, load_from_hub

# Set up gym environment
env_id = "LunarLander-v2"
env = gym.make(env_id)
eval_env = Monitor(gym.make(env_id))


def train_model():
    # Train PPO model
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1
    )
    model.learn(total_timesteps=1000000)
    model.save("ppo-LunarLander-v2")

    # Evaluate trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Package trained model to Hugging Face Hub
    package_to_hub(
        model=model,
        model_name="ppo-LunarLander-v2",
        model_architecture="PPO",
        env_id=env_id,
        eval_env=eval_env,
        repo_id="shahzebnaveed/ppo-LunarLander-v2",
        commit_message="Upload PPO LunarLander-v2 trained agent"
    )


def load_model():
    # Load model from Hugging Face Hub
    checkpoint = load_from_hub(
        repo_id="shahzebnaveed/ppo-LunarLander-v2",
        filename="ppo-LunarLander-v2.zip"
    )
    model = PPO.load(checkpoint, print_system_info=True)

    # Evaluate loaded model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=2, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def load_model_with_custom_objects():
    # Load model with custom objects
    repo_id = "Classroom-workshop/assignment2-omar"
    filename = "ppo-LunarLander-v2.zip"
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    checkpoint = load_from_hub(repo_id, filename)
    model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)

    # Evaluate model with custom objects
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def main():
    # Create virtual display
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

    train_model()
    # load_model()
    # load_model_with_custom_objects()


if __name__ == "__main__":
    main()
