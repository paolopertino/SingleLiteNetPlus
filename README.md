# SingleLiteNet+
## :gear: Installation
To install the python dependencies, we recommend to do so using a virtual environment.

1. Using `uv` (recommended):
   
    Install `uv` if you haven't already done it yet:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Then:
    ```bash
    uv sync
    source .venv/bin/activate
    ```

    Finally, install the `weightslab` dependencies:
    ```bash
    uv pip install weightslab/
    ```
2. Using `pip`:
   
   Create a virtual environment first and activate it:
   ```bash
    python -m venv .venv
    source .venv/bin/activate
   ```

   Install the dependencies:
   ```bash
    pip install .
   ```

   Finally, install the `weightslab` dependencies:
    ```bash
    pip install -e weightslab/
    ```

## :book: Datasets
The main, large-scale, dataset of reference is `BDD100k`.

The folder structure is the following:
```bash
.BDD100K
├── images_folder
│   ├──train
│   │   ├── img_1.jpg
│   │   ├── ... 
│   │   └── img_n.jpg
│   └──val
│       ├── img_1.jpg
│       ├── ... 
│       └── img_n.jpg
└── annotations_folder
    ├──train
    │   ├── img_1.png
    │   ├── ... 
    │   └── img_n.png
    └──val
        ├── img_1.png
        ├── ... 
        └── img_n.png
```
where `images_folder` and `annotations_folder` could be any path to your data given the specified structure.

## :rocket: Usage
Before explaining the commands to run experiments for the model, we'll briefly cover how the configuration system works.

### :page_facing_up: Configuration System – Hydra
This repo leverages the [Hydra Library][hydra-website] to manage configurations in a modular and composable way. Each configuration domain (model, dataset, augmentations, etc.) is stored in a dedicated folder, and Hydra automatically combines them when launching experiments. It is recommended to review the [Basic Tutorial][hydra-tutorial] and [Experiment Setup][hydra-exp-setup] for background.

Configurations are split into seven groups:
- `dataset`: contains configuration files for the supported datasets.  
- `augmentation`: contains configuration files for dataset-specific augmentations used during training.  
- `logging`: defines logging backends (local logging or Weights & Biases).  
- `loss`: specifies the loss functions used in training.  
- `model`: defines model architectures and their size variants.  
- `optimizer`: defines optimizer configurations.  
- `routine`: defines high-level routines (training, inference, evaluation, finetuning).  

The `config.yaml` file acts as the entry point and defines the default composition of configurations.

---

#### :question: How do we change parameters for different experiments?
This project follows Hydra’s [experiment configuration pattern][hydra-exp-setup].  
Instead of relying solely on command-line overrides, each experiment has its own YAML file under `configs/experiment/`, where all relevant configuration choices (dataset, model, loss, optimizer, logging, etc.) are defined in one place. This ensures reproducibility and easy sharing of experimental setups. \
Specifically, we override in the configuration file all those configuration we want to change for that experiment.

To launch an experiment:

```bash
# Run an experiment defined in configs/experiment/default_train.yaml
python train.py +experiment=default_train
```
Example configuration files for the various tasks can be found in `/configs/experiment`.

> [!IMPORTANT]
> The `default_*.yaml` files in the `experiment` folder are meant to be templates for you to use. To create your own experiment please copy and modify one of those! For the demo the `custom_train_single_task_local_tests.yaml` file has been already compiled. You have just to overwrite the dataset paths and run the script.

```bash
uv run train_singletask.py +experiment=custom_train_single_task_local_tests
```

### Running the scripts
Once you defined you desired configuration, you can run the scripts:
```bash
uv run <routine_name>.py +experiment=<experiment_name>
```
where routine name is either `train_singletask`, `finetune_singletask`, `inference_singletask`, or `val_singletask`, and experiment name is the name of the yaml file defining the experiment (without the `.yaml` extension).

---

_Last Update: 05/09/2025_

<!-- LINKS -->
[bdd-setup]: ../BDD100k_Tools/README.md
[hydra-exp-setup]: https://hydra.cc/docs/patterns/configuring_experiments/
[hydra-tutorial]: https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/
[hydra-website]: https://hydra.cc/docs/intro/
[wandb-website]: https://wandb.ai/site/