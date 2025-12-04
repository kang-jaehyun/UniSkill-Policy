import argparse
import json
import numpy as np
import time
import os
import torch
import h5py
import yaml
import pprint
from collections import OrderedDict
from easydict import EasyDict

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.libero.benchmark import get_benchmark
from libero.libero import get_libero_path
import robosuite.utils.transform_utils as T

def evaluate(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    
    if args.ckpt is not None:
        config.experiment.ckpt_path = args.ckpt

    # Initialize ObsUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_config(config)

    # Device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # Load model
    print(f"Loading model from {config.experiment.ckpt_path}")
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=config.experiment.ckpt_path)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=ckpt_dict["shape_metadata"]["all_shapes"],
        ac_dim=ckpt_dict["shape_metadata"]["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()

    # Wrap in RolloutPolicy
    rollout_model = RolloutPolicy(model)

    # Define repo root (robomimic/scripts/evaluate.py -> ../../)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))

    # Setup LIBERO
    libero_config_path = os.path.join(repo_root, "LIBERO/libero/configs")
    
    if not os.path.exists(libero_config_path):
        raise FileNotFoundError(f"Could not find LIBERO configs at {libero_config_path}")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=os.path.relpath(libero_config_path, start=current_dir))
    hydra_cfg = compose(config_name="config")
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # Prepare LIBERO paths
    cfg.folder = os.path.join(repo_root, "datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    
    # Validate task against config
    valid_tasks = []
    for data_cfg in config.train.data:
        # path is like datasets/libero_90/TASK_NAME.hdf5
        path = data_cfg["path"]
        task_name = os.path.splitext(os.path.basename(path))[0]
        # Handle _demo suffix if present in config paths
        if task_name.endswith("_demo"):
            task_name = task_name[:-5]
        valid_tasks.append(task_name)
    
    target_task_name = args.task
    # Strip _demo for validation if user provided it
    check_task_name = target_task_name.replace("_demo", "")
    
    if check_task_name not in valid_tasks:
        print(f"Error: Task '{target_task_name}' is not defined in the configuration file.")
        print("Available tasks in config:")
        for t in valid_tasks:
            print(f"  - {t}")
        return

    # Benchmark
    task_order = cfg.data.task_order_index
    cfg.benchmark_name = "libero_90"
    benchmark = get_benchmark(cfg.benchmark_name)(task_order)
    
    # Find task ID
    # Benchmark task names do not have _demo suffix
    benchmark_task_name = target_task_name.replace("_demo", "")
    
    task_id = None
    for i in range(benchmark.n_tasks):
        if benchmark.get_task(i).name == benchmark_task_name:
            task_id = i
            break
    
    if task_id is None:
        print(f"Task {benchmark_task_name} not found in benchmark {cfg.benchmark_name}")
        return

    print(f"Evaluating task: {target_task_name} (ID: {task_id})")

    # Load precomputed skill
    # Use repo_root/skills
    skill_path = os.path.join(repo_root, "skills", target_task_name, "demo_0", "base.npy")
    if not os.path.exists(skill_path):
        print(f"Skill file not found at {skill_path}")
        return
    
    print(f"Loading skill from {skill_path}")
    skill_all = np.load(skill_path)
    skill_all = torch.from_numpy(skill_all).float().to(device)
    
    # Env setup
    task = benchmark.get_task(task_id)
    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }

    num_episodes = args.rollout_num
    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_episodes)]
    )
    env.reset()
    env.seed(args.seed)

    init_states_path = os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path)
    indices = np.arange(num_episodes) % init_states.shape[0]
    init_states_ = init_states[indices]
    
    obs = env.set_init_state(init_states_)
    
    # Video writer
    video_folder = os.path.join(config.train.output_dir, "videos", args.task)
    os.makedirs(video_folder, exist_ok=True)
    
    dones = [False] * num_episodes
    steps = 0
    num_success = 0
    
    rollout_model.start_episode()

    with Timer() as t, VideoWriter(video_folder, True) as video_writer:
        # Warmup
        for _ in range(5):
            env.step(np.zeros((num_episodes, 7)))
            
        while steps < args.horizon:
            steps += 1
            
            # Prepare obs
            task_emb = torch.zeros(1, 50) 
            data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
            del data['task_emb']
            
            ee_states = np.hstack(
                (
                    np.stack([obs[i]['robot0_eef_pos'] for i in range(num_episodes)]),
                    (np.stack([T.quat2axisangle(obs[i]['robot0_eef_quat']) for i in range(num_episodes)])),
                )
            )
            ee_pos = ee_states[:, :3]
            ee_ori = ee_states[:, 3:]
            
            data['obs']['ee_ori'] = torch.tensor(ee_ori).to(device)
            data['obs']['ee_pos'] = torch.tensor(ee_pos).to(device)
            
            if steps <= skill_all.shape[0]:
                skill = skill_all[steps-1:steps]
            else:
                skill = skill_all[-1:]
            
            # Policy forward
            action = rollout_model(
                ob=data['obs'],
                skill=skill.expand(num_episodes, -1, -1),
                batched=True
            )
            
            obs, reward, done, info = env.step(action)
            video_writer.append_vector_obs(
                obs, dones, camera_name="agentview_image"
            )
            
            for k in range(num_episodes):
                dones[k] = dones[k] or done[k]
            
            if all(dones):
                break
        
        for k in range(num_episodes):
            num_success += int(dones[k])

    success_rate = num_success / num_episodes
    env.close()
    
    print(f"Success Rate: {success_rate}")
    
    # Save results
    results = {
        "task": args.task,
        "success_rate": success_rate,
        "num_episodes": num_episodes,
        "seed": args.seed
    }
    output_path = os.path.join(config.train.output_dir, f"eval_{args.task}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--rollout_num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    args = parser.parse_args()
    
    evaluate(args)
