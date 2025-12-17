import os
import yaml
from easydict import EasyDict
from hydra import compose, initialize
from omegaconf import OmegaConf
from libero.libero.benchmark import get_benchmark

def list_tasks():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    libero_config_path = os.path.abspath(os.path.join(current_dir, "../../LIBERO/libero/configs"))
    
    initialize(config_path=os.path.relpath(libero_config_path, start=os.path.dirname(os.path.abspath(__file__))))
    hydra_cfg = compose(config_name="config")
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    
    benchmark_name = "libero_90"
    task_order = cfg.data.task_order_index
    benchmark = get_benchmark(benchmark_name)(task_order)
    
    print(f"Tasks in {benchmark_name}:")
    for i in range(benchmark.n_tasks):
        print(benchmark.get_task(i).name)

if __name__ == "__main__":
    list_tasks()
