# Whole-body Representation Learning
This repository contains the code used in the research paper: [Whole Heart 3D+T Representation Learning Through Sparse 2D Cardiac MR Images](https://link.springer.com/chapter/10.1007/978-3-031-72378-0_34#Abs1), accepted by MICCAI 2024. For more details, please refer to the paper.
![Diagram](main_structure.png)

## Table of Contents
- [Installation](#installation)
- [Data File Structure](#data-file-structure)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

To get a local copy up and running, follow these steps:

#### Prerequisites
Before you begin, ensure you have met the following requirements:
- **Python 3.9+** as the programming language.
- **Conda** installed (part of the Anaconda or Miniconda distribution).
- **pip** installed for package management.
- **Git** installed to clone the repository.

#### Steps

1. **Clone the repository**
    ```bash
    git clone https://github.com/Yundi-Zhang/WholeHeartRL.git
    cd WholeHeartRL
    ```

2. **Create and activate a Conda environment**
    ```bash
    # Create a new Conda environment with Python 3.9 (or your required version)
    conda create --name wholeheart python=3.9

    # Activate the Conda environment
    conda activate wholeheart
    ```

3. **Install dependencies**
    ```bash
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchsummary -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    ```

4. **Configure environment variables**
    Rename `.env.name` to `.env` and update the necessary environment variables.
    ```bash
    mv .env.name .env
    ```

## Changes compared to the Yundi's code
    - New Whole-body Dataloader
    - Masking the ROI and reconstruction of it





## Usage
This project supports three tasks: **Pertaining**, **Segmentation**, and **Regression**. Follow the instructions below to run the application for each task:

1. **Pertaining**: Follow the specific instructions provided for the Pertaining task.

2. **Segmentation**: To run the Segmentation task, make sure to specify the pretraining checkpoint path by setting the `general.ckpt_path` parameter.

3. **Regression**: For the Regression task, you also need to provide the pretraining checkpoint path using the `general.ckpt_path` parameter.

#### Pertaining via reconstruction
```bash
source .env
python3 main.py train \
-c ./configs/config_reconstruction_wb.yaml \
-g your_wandb_group_name \
-n your_wandb_job_name
```
```
#### Regression
```bash
source .env
python3 main.py train \
-c ./configs/config_regression_age.yaml \
-g your_wandb_group_name \
-n your_wandb_job_name
```

#### Embedding Extraction
```bash
python3 main.py eval \
-c ./configs/config_reconstruction_wb.yaml \
-g mae_emb \
-n mae_emb \
--labels_file /path/to/labels.csv
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, contact [yundi.zhang@tum.de](mailto:yundi.zhang@tum.de) or [jiazhen.pan@tum.de](mailto:jiazhen.pan@tum.de). If you use this code in your research, please cite the above mentioned paper.
