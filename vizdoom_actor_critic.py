import functools
from base64 import b64encode
from IPython.display import HTML
from huggingface_hub import notebook_login

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.enjoy import enjoy

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def train_vizdoom():
    env = "doom_health_gathering_supreme"
    cfg = parse_vizdoom_cfg(argv=[f"--env={env}", "--num_workers=8", "--num_envs_per_worker=4", "--train_for_env_steps=4000000"])
    status = run_rl(cfg)


def enjoy_vizdoom():
    env = "doom_health_gathering_supreme"
    cfg = parse_vizdoom_cfg(argv=[f"--env={env}", "--num_workers=1", "--save_video", "--no_render", "--max_num_episodes=10"], evaluation=True)
    status = enjoy(cfg)


def display_video():
    mp4 = open('/content/train_dir/default_experiment/replay.mp4','rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=640 controls>
          <source src="%s" type="video/mp4">
    </video>
    """ % data_url)


def push_to_hub():
    hf_username = "shahzebnaveed"
    env = "doom_health_gathering_supreme"
    cfg = parse_vizdoom_cfg(argv=[f"--env={env}", "--num_workers=1", "--save_video", "--no_render", "--max_num_episodes=10", "--max_num_frames=100000", "--push_to_hub", f"--hf_repository={hf_username}/rl_course_vizdoom_health_gathering_supreme"], evaluation=True)
    status = enjoy(cfg)


def main():
    register_vizdoom_components()
    train_vizdoom()
    enjoy_vizdoom()
    # display_video()
    # notebook_login()
    push_to_hub()


if __name__ == "__main__":
    main()
