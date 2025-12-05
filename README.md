# UniSkill Policy

This repository is the **policy learning component of UniSkill**, based on [robomimic](https://github.com/ARISE-Initiative/robomimic). It is used to train and evaluate policies for the UniSkill project.

### Prerequisites
- Linux (tested on Ubuntu 20.04/22.04)
- Python 3.10
- [uv](https://github.com/astral-sh/uv) (optional, installed automatically by script)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone --recursive https://github.com/kang-jaehyun/Uniskillpolicy.git
    cd Uniskillpolicy
    ```
    *Note: The `--recursive` flag is important to initialize submodules (LIBERO, robocasa, robosuite).*

2.  **Run the installation script:**
    This script initializes submodules, and installs the environment with `uv` .
    ```bash
    bash install_env.sh
    ```

3.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Setup Private Macros:**
    Create a private macros file to configure local paths and settings.
    ```bash
    python robomimic/scripts/setup_macros.py
    ```
    
## Data Preparation

Before training, you need to prepare the dataset and skill embeddings.

### 1. LIBERO Dataset
Download the LIBERO dataset from the [official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Place the dataset files in the `datasets/libero_90` directory within the project root.

### 2. LIBERO Skills
Download the pre-computed skill embeddings from the following link:
*   [Google Drive Link](https://drive.google.com/file/d/104LGZZ4P4hDLcQCOq0YVOD2hfSSmBxPg/view?usp=drive_link)

Extract the skills to the `skills/` directory. The structure should look like this:
```
skills/
├── [TASK_NAME] (e.g., KITCHEN_SCENE10_...)
│   ├── demo_0/
│   │   ├── base.npy
│   │   ├── aug_0.npy
│   │   ├── ...
│   ├── demo_1/
│   ├── ...
├── [ANOTHER_TASK_NAME]
│   ├── ...
```
Each task folder contains subfolders for each demonstration (`demo_0`, `demo_1`, etc.), and each demonstration folder contains the skill embeddings (`.npy` files).

### 3. Pretrained Checkpoint
We release a pretrained checkpoint for the UniSkill policy.
*   [Checkpoint Link](https://drive.google.com/file/d/1bBiOaJNK21x6ePoWQ3Kg3N4GDnMa6Duf/view?usp=drive_link)

## Configuration

The training configuration is defined in `configs/uniskill_policy.json`. Key parameters include:

*   **`train.data`**: A list of dataset paths. Each entry specifies a `.hdf5` file containing demonstration data.
    *   `path`: Path to the dataset file (relative to the project root, e.g., `datasets/libero_90/...`).
*   **`train.skill_dir`**: Path to the directory containing skill embeddings (e.g., `skills/`).
*   **`train.output_dir`**: Directory where training logs and checkpoints will be saved.

## Prerequisites for Training

Before training, ensure that you have prepared the necessary data.
- **Skill Directory**: You must extract or prepare the skill directory (skill embeddings/data) in advance before running the training script.

## Usage

### Training

To train the UniSkill policy using the default configuration:

```bash
python robomimic/scripts/train.py --config configs/uniskill_policy.json
```

### Evaluation
You can evaluate trained policies using the provided `evaluate.sh` script located in `robomimic/scripts/`.

#### Usage

```bash
./robomimic/scripts/evaluate.sh --ckpt <CHECKPOINT_PATH> --task <TASK_NAME> [options]
```

#### Arguments

| Argument | Description | Required | Default |
|----------|-------------|:--------:|:-------:|
| `--ckpt` | Path to the model checkpoint (`.pth` file). | Yes | - |
| `--task` | Name of the task to evaluate. | Yes | - |
| `--config` | Path to the config JSON file. | No | `configs/uniskill_policy.json` |
| `--rollout_num` | Number of rollout episodes. | No | 10 |
| `--horizon` | Horizon length for each episode. | No | 500 |
| `--seed` | Random seed. | No | 0 |

#### Example

```bash
./robomimic/scripts/evaluate.sh \
    --ckpt path/to/checkpoint/model_epoch_2000.pth \
    --task LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo \
    --rollout_num 50 \
    --horizon 500
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
