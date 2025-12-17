"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings



def rollout(config, device, task, epoch=0):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    eval_env_meta_list = []
    eval_shape_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
    for (dataset_i, dataset_cfg) in enumerate(config.train.data):
        # hardcoded task definition for now
        if task.lower() not in dataset_cfg.get('path').lower():
            continue
        
        eval_env_meta_list.append(env_meta_list[dataset_i])
        eval_shape_meta_list.append(shape_meta_list[dataset_i])
        eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
        horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)
    
    # create environments
    def env_iterator():
        for (env_meta, shape_meta, env_name) in zip(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list):
            def create_env_helper(env_i=0):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                    seed=config.train.seed * 1000 + env_i,
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                # handle environment wrappers
                env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

                return env

            if config.experiment.rollout.batched:
                from tianshou.env import SubprocVectorEnv
                env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
                env = SubprocVectorEnv(env_fns)
                # env_name = env.get_env_attr(key="name", id=0)[0]
            else:
                env = create_env_helper()
                # env_name = env.name
            print(env)
            yield env

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # if checkpoint is specified, load in model weights
    ckpt_path = config.experiment.ckpt_path
    if ckpt_path is not None:
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    lang_encoder = LangUtils.LangEncoder(
        device=device,
    )
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=lang_encoder)

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in eval_env_name_list} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in eval_env_name_list} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    video_paths = None
    should_save_ckpt = False
    epoch_ckpt_name = "model_epoch_{}".format("rollout")
    rollout_model = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
        lang_encoder=lang_encoder,
    )

    num_episodes = config.experiment.rollout.n
    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=env_iterator(),
        horizon=eval_env_horizon_list,
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir if config.experiment.render_video else None,
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        del_envs_after_rollouts=True,
        data_logger=data_logger,
    )

    # checkpoint and video saving logic

    updated_stats = TrainUtils.should_save_from_rollout_logs(
        all_rollout_logs=all_rollout_logs,
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
        save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
    )
    best_return = updated_stats["best_return"]
    best_success_rate = updated_stats["best_success_rate"]
    epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
    should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt

    
    # Finally, log memory usage in MB
    process = psutil.Process(os.getpid())
    mem_usage = int(process.memory_info().rss / 1000000)
    data_logger.record("System/RAM Usage (MB)", mem_usage, 00)
    print("\nEpoch {} Memory Usage: {} MB\n".format(0, mem_usage))

    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.ckpt is not None:
        config.experiment.ckpt_path = args.ckpt

    if args.rollout_num is not None:
        config.experiment.rollout.n = args.rollout_num
        
    if args.seed is not None:
        config.train.seed = args.seed
        
    if "train" in config.experiment.logging.wandb_proj_name:
        config.experiment.logging.wandb_proj_name = config.experiment.logging.wandb_proj_name.replace("train", "rollout")
    
    if args.name is not None:
        config.experiment.name = "_".join([config.experiment.name, args.name])
        
    config.experiment.name = os.path.join(config.experiment.name, f"{args.task}_epoch_{args.epoch}_rolloutseed_{config.train.seed}")
    
    config.train.output_dir = os.path.join(config.train.output_dir, f'rollout')
    
    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        rollout(config, device=device, epoch=args.epoch, task=args.task)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )
    
    parser.add_argument(
        "--rollout_num",
        type=int,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default='opendrawer',
        help="(optional) if provided, override the dataset path defined in the config",
    )
    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    parser.add_argument(
        "--ckpt",
        required=False,
        type=str,
        help="Path to a checkpoint to load in model weights from."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generators."
    )
    
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help=""
    )
        
    args = parser.parse_args()
    main(args)
