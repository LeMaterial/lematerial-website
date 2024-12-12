---
title: "LeMaterial with FAIRChem"
description: "LeMaterial with FAIRChem"
summary: ""
date: 2023-09-07T16:04:48+02:00
lastmod: 2023-09-07T16:04:48+02:00
draft: false
weight: 820
toc: true
seo:
  title: "LeMaterial x FAIRChem"
  description: "Train LeMaterial with FAIRChem EquiformerV2 model"
  canonical: "" # custom canonical URL (optional)
  noindex: false # false (default) or true
---

<a target="_blank" href="https://colab.research.google.com/drive/1y8_CzKM5Rgsiv9JoPmi9mXphi-kf6Lec?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<br/><br/>

The goal of this tutorial is to show how to use LeMaterial's dataset with Geometric GNNs designed for molecular property prediction and relaxation from the [FAIRChem repository](https://github.com/FAIR-Chem/fairchem).

For more information on how to use FAIRChem's models, please refer to the [FAIRChem repository](https://github.com/FAIR-Chem/fairchem) and their [documentation](https://fair-chem.github.io/).

## Setup the environment

The best way to setup an environment for FAIRChem is to use the provided conda environment file and to create it with the following command:

```bash
wget https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/env.gpu.yml
conda env create -f env.gpu.yml
conda activate fair-chem
```

Or to separately install the `torch_geometric` dependencies:

```bash
!pip install torch_geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Then we need to install FAIRChem on the environment:

```bash
pip install fairchem-core
```

Or manually (currently recommended way):

```bash
git clone https://github.com/FAIR-Chem/fairchem
pip install -e fairchem/packages/fairchem-core
```

We also need to install PyTorch dependencies, make sure to pick the correct version of cuda for PyTorch Geometric, along with the right PyTorch version.

```python
%%capture --no-display

!pip install fairchem-core
!pip install torch_geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

```python
CPU = False # Run on CPU
BATCH_SIZE = 2 # Train and evaluation batch size
```

## Load the dataset

We use the dataset available at [LeMaterial's Hugging Face space](https://huggingface.co/LeMaterial).

```python
%%capture --no-display
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

!pip install datasets
!huggingface-cli login --token $HF_TOKEN
```

```python
%%capture --no-display
from datasets import load_dataset

HF_DATASET_PATH = "LeMaterial/LeMat-Bulk"
SUBSET = "compatible_pbe"

dataset = load_dataset(HF_DATASET_PATH, SUBSET)["train"]
```

    Downloading data:   0%|          | 0/17 [00:00<?, ?files/s]
    Generating train split:   0%|          | 0/5335299 [00:00<?, ? examples/s]
    Loading dataset shards:   0%|          | 0/17 [00:00<?, ?it/s]
    Resolving data files:   0%|          | 0/17 [00:00<?, ?it/s]

## Load a model

We need to start by loading a trained model on which we can run predictions. For example, we can download a checkpoint from EquiformerV2 available [here](https://huggingface.co/yilunliao/equiformer_v2).

```python
%%capture --no-display
from huggingface_hub import hf_hub_download
from fairchem.core import OCPCalculator

HF_REPOID = "fairchem/OMAT24"
HF_MODEL_PATH = "eqV2_31M_omat_mp_salex.pt"

def download_model(hf_repo_id, hf_model_path):
    model_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_model_path)
    return model_path

model_path = download_model(HF_REPOID, HF_MODEL_PATH)

calc = OCPCalculator(checkpoint_path=model_path, cpu=CPU)
```

    0it [00:00, ?it/s]
    eqV2_31M_omat_mp_salex.pt:   0%|          | 0.00/126M [00:00<?, ?B/s]
    INFO:root:local rank base: 0
    INFO:root:amp: true
    ...
    INFO:root:Loading model: hydra
    WARNING:root:equiformerV2_energy_head (EquiformerV2EnergyHead) class is deprecated in favor of equiformerV2_scalar_head  (EqV2ScalarHead)
    WARNING:root:equiformerV2_force_head (EquiformerV2ForceHead) class is deprecated in favor of equiformerV2_rank1_head  (EqV2Rank1Head)
    INFO:root:Loaded HydraModel with 31207434 parameters.
    INFO:root:Loading checkpoint in inference-only mode, not loading keys associated with trainer state!
    WARNING:root:No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run

## Inference on a single structure

We first need to convert a row from the dataset of the material that we want to predict the property to an ASE molecule which can be digested by the model.

```python
from ase import Atoms
from pymatgen.core.structure import Structure
import numpy as np
from collections import defaultdict
from ase.calculators.singlepoint import SinglePointCalculator # To add targets
from pymatgen.io.ase import AseAtomsAdaptor

random_sample = np.random.randint(len(dataset))
row = dataset[random_sample]

def get_atoms_from_row(row, add_targets=False, add_forces=True, add_stress=False):
      # Convert row to PyMatGen
      structure = Structure(
          [x for y in row["lattice_vectors"] for x in y],
          species=row["species_at_sites"],
          coords=row["cartesian_site_positions"],
          coords_are_cartesian=True,
      )

      atoms = AseAtomsAdaptor.get_atoms(structure)

      # Add the forces and energy as targets
      if add_targets:
        forces, stres = None, None
        if add_forces:
          if np.array(row["forces"]).shape[0] == np.array(row["cartesian_site_positions"]).shape[0]:
            forces = row["forces"]
          else:
            return None
        # OMAT uses the stress tensor as output as well
        if add_stress:
          if np.array(row["stress_tensor"]).shape[0] == np.array(row["cartesian_site_positions"]).shape[0]:
            stress=row["stress_tensor"]
          else:
            return None
        atoms.calc = SinglePointCalculator(atoms, forces=forces, energy=row["energy"], stress=stress)

      return atoms

atoms = get_atoms_from_row(row)
```

We can now run the inference on the chosen row of the dataset. Since most models inside FAIRChem are designed to predict the energy and the forces of a material at a given structure (S2EF), we can run relaxation (MD) on the structure to get the energy at the relaxed state as well.

We first show how to predict the energy property of a material without relaxation and then with relaxation.

```python
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

def relax_atoms(atoms, steps=0, fmax=0.05):
    atoms.calc = calc
    dyn = FIRE(FrechetCellFilter(atoms))
    dyn.run(fmax=fmax, steps=steps)

    return atoms

print(f"{'--' * 5} No Relaxation {'--' * 5}")
atoms = relax_atoms(atoms, steps=0)
predicted_energy = atoms.get_potential_energy()

print(f"Predicted energy: {predicted_energy} eV")
print(f"DFT energy: {row['energy']} eV")

print("\n"*2)

print(f"{'--' * 5} With Relaxation {'--' * 5}")
atoms = get_atoms_from_row(row)
atoms = relax_atoms(atoms, steps=200)
predicted_energy = atoms.get_potential_energy()

print(f"Predicted energy: {predicted_energy} eV")
print(f"DFT energy: {row['energy']} eV")
```

    ---------- No Relaxation ----------
      Step     Time          Energy          fmax
    FIRE:    0 14:54:04       -5.517979        0.267703
    Predicted energy: -5.517978668212891 eV
    DFT energy: -5.57597026 eV
    ---------- With Relaxation ----------
          Step     Time          Energy          fmax
    FIRE:    0 14:54:04       -5.517979        0.267703
    FIRE:    1 14:54:05       -5.520627        0.276172
    FIRE:    2 14:54:05       -5.524797        0.263472
    FIRE:    3 14:54:06       -5.527802        0.213701
    FIRE:    4 14:54:06       -5.529005        0.166641
    FIRE:    5 14:54:07       -5.529183        0.136835
    FIRE:    6 14:54:07       -5.529638        0.124812
    FIRE:    7 14:54:07       -5.531461        0.123644
    FIRE:    8 14:54:08       -5.532405        0.125816
    FIRE:    9 14:54:08       -5.534833        0.127752
    FIRE:   10 14:54:08       -5.536007        0.130083
    FIRE:   11 14:54:09       -5.535960        0.128654
    FIRE:   12 14:54:09       -5.536138        0.120737
    FIRE:   13 14:54:09       -5.533516        0.108630
    FIRE:   14 14:54:10       -5.530128        0.100276
    FIRE:   15 14:54:10       -5.528875        0.096984
    FIRE:   16 14:54:10       -5.529051        0.104039
    FIRE:   17 14:54:11       -5.570351        0.031070
    Predicted energy: -5.5703511238098145 eV
    DFT energy: -5.57597026 eV

## Create an LMDB dataset

In order to run batched inference, we need to create a database compatible with FAIRChem's dataloader. The recommended way to do is to currently create an ASE LMDB database and pass it to FAIRChem's config.

```python
from pathlib import Path

import tqdm
from ase import Atoms

from fairchem.core.datasets.lmdb_database import LMDBDatabase
```

We will create an LMDB database based on LeMaterial's entire dataset. Note that you can also use a subset of the dataset if you want to by filtering relevant structures for example. This could allow to fine-tune on selected materials, or test the model on a specific subset of materials.

We discuss about training and fine-tuning in the last section of this notebook.

```python
# REF: https://github.com/FAIR-Chem/fairchem/issues/787
output_path = Path("leMat.aselmdb")

select_range = 100
small_dataset = dataset.select(range(select_range))

for row in tqdm.tqdm(small_dataset, total=len(small_dataset)):
    with LMDBDatabase(output_path) as db:
        atoms = get_atoms_from_row(row, add_targets=False) # not needed for inference

        db.write(atoms, data={"id": row["immutable_id"]})
```

    100%|██████████| 100/100 [00:00<00:00, 118.21it/s]
    100%|██████████| 100/100 [00:01<00:00, 86.01it/s]

## Run batched inference

#### Load the created LMDB dataset in model

The model object loaded from the checkpoint contains all the necessary information on the config file, including the paths to the train, test and validation splits. We can use this information to load our newly created LeMaterial's LMDB dataset to the model.

```python
from fairchem.core.common.tutorial_utils import generate_yml_config
import yaml

yml_path = generate_yml_config(
    model_path,
    "/tmp/config.yml",
    delete=[
        "logger",
        "task",
        "model_attributes",
        "dataset",
        "slurm",
        "optim.load_balancing",
    ],
    # Load balancing works only if a metadata.npz file is generated using the make_lmdb script (see: https://github.com/FAIR-Chem/fairchem/issues/876)
    update={
        "amp": True,
        "gpus": 1,
        "task.prediction_dtype": "float32",
        "logger": "tensorboard",
        # Test data - prediction only so no regression
        "test_dataset.src": "leMat.aselmdb",
        "test_dataset.format": "ase_db",
        "test_dataset.2g_args.r_energy": False,
        "test_dataset.a2g_args.r_forces": False,
        "optim.eval_batch_size": BATCH_SIZE,
    },
)

```

    INFO:root:Loading model: hydra
    WARNING:root:equiformerV2_energy_head (EquiformerV2EnergyHead) class is deprecated in favor of equiformerV2_scalar_head  (EqV2ScalarHead)
    WARNING:root:equiformerV2_force_head (EquiformerV2ForceHead) class is deprecated in favor of equiformerV2_rank1_head  (EqV2Rank1Head)
    INFO:root:Loaded HydraModel with 31207434 parameters.
    INFO:root:Loading checkpoint in inference-only mode, not loading keys associated with trainer state!
    WARNING:root:No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run
    /usr/local/lib/python3.10/dist-packages/fairchem/core/common/relaxation/ase_utils.py:190: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    INFO:root:amp: true
    ...
    INFO:root:Loading model: hydra
    WARNING:root:equiformerV2_energy_head (EquiformerV2EnergyHead) class is deprecated in favor of equiformerV2_scalar_head  (EqV2ScalarHead)
    WARNING:root:equiformerV2_force_head (EquiformerV2ForceHead) class is deprecated in favor of equiformerV2_rank1_head  (EqV2Rank1Head)
    INFO:root:Loaded HydraModel with 31207434 parameters.
    INFO:root:Loading checkpoint in inference-only mode, not loading keys associated with trainer state!
    WARNING:root:No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run

#### Run batched inference

```python
from fairchem.core.common.tutorial_utils import fairchem_main

# Recommended way to run inference
yml_path = generate_yml_config(
    model_path,
    "/tmp/config.yml",
    delete=[
        "logger",
        "task",
        "model_attributes",
        "slurm",
        "optim.load_balancing",
    ],
    # Load balancing works only if a metadata.npz file is generated using the make_lmdb script (see: https://github.com/FAIR-Chem/fairchem/issues/876)
    update={
        "amp": True,
        "gpus": 1,
        "task.prediction_dtype": "float32",
        "logger": "tensorboard",
        # Compatibility issues between current fairchem version and OMAT24 model? (not needed for inference)
        "loss_functionsn": "mae",
        # Test data - prediction only so no regression
        "test_dataset.src": "leMat.aselmdb",
        "test_dataset.format": "ase_db",
        "test_dataset.2g_args.r_energy": False,
        "test_dataset.a2g_args.r_forces": False,
        "optim.eval_batch_size": BATCH_SIZE,
    },
)


import locale
locale.getpreferredencoding = lambda: "UTF-8" # For running the main script

!python {fairchem_main()} --mode predict --config-yml {yml_path} --checkpoint {model_path} {'--cpu' if CPU else ''}
```

```python
# If you want to have control over the trainer object (and for example add hooks on the modules)

yml_path = generate_yml_config(
    model_path,
    "/tmp/config.yml",
    delete=[
        "logger",
        "task",
        "dataset"
        "model_attributes",
        "slurm",
        "optim.load_balancing",
    ],
    # Load balancing works only if a metadata.npz file is generated using the make_lmdb script (see: https://github.com/FAIR-Chem/fairchem/issues/876)
    update={
        "amp": True,
        "gpus": 1,
        "task.prediction_dtype": "float32",
        "logger": "tensorboard",
        # Test data - prediction only so no regression
        "test_dataset.src": "leMat.aselmdb",
        "test_dataset.format": "ase_db",
        "test_dataset.2g_args.r_energy": False,
        "test_dataset.a2g_args.r_forces": False,
        "optim.eval_batch_size": BATCH_SIZE,
    },
)

config = yaml.safe_load(open(yml_path))
config["dataset"] = {}
config["val_dataset"] = {}

config["optim"]["scheduler_params"] = {'lambda_type': 'cosine',
   'warmup_factor': 0.2,
   'warmup_epochs': 463,
   'lr_min_factor': 0.01,
}

calc.trainer.config = config
calc.trainer.load_datasets()
calc.trainer.is_debug = False
calc.trainer.predict(
    calc.trainer.test_loader, calc.trainer.test_sampler
)
```

    INFO:root:Loading model: hydra
    WARNING:root:equiformerV2_energy_head (EquiformerV2EnergyHead) class is deprecated in favor of equiformerV2_scalar_head  (EqV2ScalarHead)
    WARNING:root:equiformerV2_force_head (EquiformerV2ForceHead) class is deprecated in favor of equiformerV2_rank1_head  (EqV2Rank1Head)
    INFO:root:Loaded HydraModel with 31207434 parameters.
    INFO:root:Loading checkpoint in inference-only mode, not loading keys associated with trainer state!
    WARNING:root:No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run
    WARNING:root:Could not find dataset metadata.npz files in '[PosixPath('leMat.aselmdb')]'
    WARNING:root:Disabled BalancedBatchSampler because num_replicas=1.
    WARNING:root:Failed to get data sizes, falling back to uniform partitioning. BalancedBatchSampler requires a dataset that has a metadata attributed with number of atoms.
    INFO:root:rank: 0: Sampler created...
    INFO:root:Created BalancedBatchSampler with sampler=<fairchem.core.common.data_parallel.StatefulDistributedSampler object at 0x794e8f76eec0>, batch_size=2, drop_last=False
    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    INFO:root:Predicting on test.
    device 0:   0%|          | 0/50 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/fairchem/core/trainers/ocp_trainer.py:471: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=self.scaler is not None):
    device 0: 100%|██████████| 50/50 [00:13<00:00,  3.78it/s]

    INFO:root:Loading model: hydra
    WARNING:root:equiformerV2_energy_head (EquiformerV2EnergyHead) class is deprecated in favor of equiformerV2_scalar_head  (EqV2ScalarHead)
    WARNING:root:equiformerV2_force_head (EquiformerV2ForceHead) class is deprecated in favor of equiformerV2_rank1_head  (EqV2Rank1Head)
    INFO:root:Loaded HydraModel with 31207434 parameters.
    INFO:root:Loading checkpoint in inference-only mode, not loading keys associated with trainer state!
    WARNING:root:No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run
    WARNING:root:Could not find dataset metadata.npz files in '[PosixPath('leMat.aselmdb')]'
    WARNING:root:Disabled BalancedBatchSampler because num_replicas=1.
    WARNING:root:Failed to get data sizes, falling back to uniform partitioning. BalancedBatchSampler requires a dataset that has a metadata attributed with number of atoms.
    INFO:root:rank: 0: Sampler created...
    INFO:root:Created BalancedBatchSampler with sampler=<fairchem.core.common.data_parallel.StatefulDistributedSampler object at 0x794e8f6f2260>, batch_size=2, drop_last=False
    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    INFO:root:Predicting on test.
    device 0:   0%|          | 0/100 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/fairchem/core/trainers/ocp_trainer.py:471: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=self.scaler is not None):
    device 0: 100%|██████████| 100/100 [00:22<00:00,  4.35it/s]

    defaultdict(list,
                {'energy': [array([-92.6361], dtype=float32),
                  array([-67.37538], dtype=float32),
                  array([-46.047863], dtype=float32),
                  array([-14.097839], dtype=float32),
                  ...

## Train / fine-tune a model

In order to train models with our dataset, we need to create the train and validation splits as well. This will require specifying the targets in the LMDB and letting the model correctly pick them up.

Since LeMaterial's database is composed of atomic forces and energies at a given structure (no trajectories for now), we want to use the energy and the forces as targets of an S2EF model. Note that EquiformerV2 trained on OMAT is an S2EFS (stress) model, so stress needs to be added to targets.

In order to train models with our dataset, we need to create the train and validation splits as well. This will require specifying the targets in the LMDB and letting the model correctly pick them up.

We provide an example of how it is possible to generate a few LMDB datasets and then plug use them for training a model. Notice that we need the targets in here which are directly read by the `Atoms2Graph` class internally.

```python
ADD_STRESS = True # Need for omat24 models! (S2E'S)

splits = {
    "train": range(1000),
    "val": range(1000, 2000),
    "test": range(2000, 3000)
}

for split in splits:
  small_dataset = dataset.select(splits[split])

  for row in tqdm.tqdm(small_dataset, total=len(small_dataset)):
      with LMDBDatabase(f"leMat_{split}.aselmdb") as db:
          atoms = get_atoms_from_row(row, add_targets=True, add_forces=True, add_stress=ADD_STRESS) # Reject if there are no forces in the dataset
          if atoms is None:
             continue

          db.write(atoms, data={"id": row["immutable_id"]})
```

    100%|██████████| 1000/1000 [00:02<00:00, 345.02it/s]
    100%|██████████| 1000/1000 [00:02<00:00, 384.47it/s]
    100%|██████████| 1000/1000 [00:01<00:00, 604.11it/s]
    100%|██████████| 1000/1000 [00:03<00:00, 266.96it/s]
    100%|██████████| 1000/1000 [00:02<00:00, 372.12it/s]
    100%|██████████| 1000/1000 [00:01<00:00, 580.56it/s]

Since creating LMDB files takes significantly more time as the size increases, we recommend separating the datasets into smaller chunks of .aselmdb files that are in the same directory and then redirect the config to this directory instead of the aselmdb file. The data loaders are then able to correctly concatenate the files as needed.

Many model implementations exist in fairchem. This is an example of a few and how we use them. More of these can be found [here](https://github.com/FAIR-Chem/fairchem/tree/main/configs).

```python
example_configs = {
    "eqv2_omat_S": "configs/omat24/all/eqV2_31M.yml",
    "eqv2_omat_S": "configs/omat24/all/eqV2_31M.yml",
    "eqv2_s2ef_L": "configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml",
    "eqv2_s2ef_S": "configs/s2ef/all/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml",
    "dpp": "configs/s2ef/all/dimenet_plus_plus/dpp.yml",
}

CHOSEN_MODEL = "dpp"
```

We need to apply a little bit of processing on the yaml config files to be able to read them with the main script of FAIRChem.

- Apply the datasets
- Change some config parameters (this can be adjusted depending on what you want to put in the model)

More information on how to tweak this config file can be found in FAIRChem's documentation as well. Note that you can also modify some config arguments with the cli parameters directly.

```python
from fairchem.core.common.tutorial_utils import fairchem_main
from pathlib import Path

import yaml

!git clone https://github.com/FAIR-Chem/fairchem.git # to download configs








config_path = Path("fairchem") / example_configs[CHOSEN_MODEL]
yaml_obj = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

# Change these parameters according to need
include_dict = {'trainer': 'ocp',
                'logger': 'tensorboard', # or wandb
                'outputs': {'energy': {'shape': 1, 'level': 'system'}, 'forces': {'irrep_dim': 1, 'level': 'atom', 'train_on_free_atoms': True, 'eval_on_free_atoms': True}},
                'loss_functions': [{'energy': {'fn': 'mae', 'coefficient': 1}}, {'forces': {'fn': 'l2mae', 'coefficient': 100}}],
                'evaluation_metrics': {'metrics': {'energy': ['mae'], 'forces': ['mae', 'cosine_similarity', 'magnitude_error'], 'misc': ['energy_forces_within_threshold']}, 'primary_metric': 'forces_mae'}
                }

yaml_obj["dataset"] = {
    "train": {"src": "leMat_train.aselmdb", "format": "ase_db", "a2g_args": {"r_energy": True, "r_forces": True}},
    "val": {"src": "leMat_val.aselmdb", "format": "ase_db", "a2g_args": {"r_energy": True, "r_forces": True}},
    "test": {"src": "leMat_test.aselmdb", "format": "ase_db", "a2g_args": {"r_energy": True, "r_forces": True}},
}
if "includes" in yaml_obj:
  del yaml_obj["includes"]

yaml_obj.update(include_dict)
yaml_obj["model"]["otf_graph"] = True

# For equiformer models: set the trainer to the equiformerv2_forces one
if "eqv2" in CHOSEN_MODEL:
  yaml_obj["trainer"] = "equiformerv2_forces"
  yaml_obj["optim"]["scheduler_params"] = {'lambda_type': 'cosine',
   'warmup_factor': 0.2,
   'warmup_epochs': 463,
   'lr_min_factor': 0.01,
  }

# No metadata.npz file, disabling load_balancing
if "load_balancing" in yaml_obj["optim"]:
  del yaml_obj["optim"]["load_balancing"]

new_yaml_path = Path(f"/tmp/{CHOSEN_MODEL}_leMat.yml")
with open(new_yaml_path, "w") as f:
    yaml.dump(yaml_obj, f)
```

    fatal: destination path 'fairchem' already exists and is not an empty directory.

```python
# Unclear whether it is possible to access this main.py script from the pip package
!python fairchem/main.py --mode train --config-yml /tmp/{CHOSEN_MODEL}_leMat.yml {"--cpu" if CPU else ""}
```

    2024-12-12 14:38:02 (INFO): Running in local mode without elastic launch (single gpu only)
    2024-12-12 14:38:02 (INFO): Setting env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    [W1212 14:38:02.705129468 CUDAAllocatorConfig.h:28] Warning: expandable_segments not supported on this platform (function operator())
    2024-12-12 14:38:02 (INFO): Project root: /usr/local/lib/python3.10/dist-packages/fairchem
    2024-12-12 14:38:02.967649: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-12-12 14:38:02.986891: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-12-12 14:38:02.992752: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-12-12 14:38:04.287556: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    2024-12-12 14:38:05 (INFO): NumExpr defaulting to 2 threads.
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/scn/spherical_harmonics.py:23: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/equiformer_v2/wigner.py:10: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/equiformer_v2/layer_norm.py:75: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/equiformer_v2/layer_norm.py:175: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/equiformer_v2/layer_norm.py:263: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/equiformer_v2/layer_norm.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/escn/so3.py:23: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
    2024-12-12 14:38:07 (INFO): local rank base: 0
    2024-12-12 14:38:07 (INFO): amp: false
    ...
    2024-12-12 14:38:07 (INFO): Loading model: dimenetplusplus
    2024-12-12 14:38:25 (INFO): Loaded DimeNetPlusPlusWrap with 1810182 parameters.
    2024-12-12 14:38:25 (WARNING): log_summary for Tensorboard not supported
    2024-12-12 14:38:25 (INFO): Loading dataset: ase_db
    2024-12-12 14:38:25 (WARNING): Could not find dataset metadata.npz files in '[PosixPath('leMat_train.aselmdb')]'
    2024-12-12 14:38:25 (WARNING): Disabled BalancedBatchSampler because num_replicas=1.
    2024-12-12 14:38:25 (WARNING): Failed to get data sizes, falling back to uniform partitioning. BalancedBatchSampler requires a dataset that has a metadata attributed with number of atoms.
    2024-12-12 14:38:25 (INFO): rank: 0: Sampler created...
    2024-12-12 14:38:25 (INFO): Created BalancedBatchSampler with sampler=<fairchem.core.common.data_parallel.StatefulDistributedSampler object at 0x7ed8d5dc2a70>, batch_size=8, drop_last=False
    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    2024-12-12 14:38:25 (WARNING): Could not find dataset metadata.npz files in '[PosixPath('leMat_val.aselmdb')]'
    2024-12-12 14:38:25 (WARNING): Disabled BalancedBatchSampler because num_replicas=1.
    2024-12-12 14:38:25 (WARNING): Failed to get data sizes, falling back to uniform partitioning. BalancedBatchSampler requires a dataset that has a metadata attributed with number of atoms.
    2024-12-12 14:38:25 (INFO): rank: 0: Sampler created...
    2024-12-12 14:38:25 (INFO): Created BalancedBatchSampler with sampler=<fairchem.core.common.data_parallel.StatefulDistributedSampler object at 0x7ed8d5b5b490>, batch_size=8, drop_last=False
    2024-12-12 14:38:25 (WARNING): Could not find dataset metadata.npz files in '[PosixPath('leMat_test.aselmdb')]'
    2024-12-12 14:38:25 (WARNING): Disabled BalancedBatchSampler because num_replicas=1.
    2024-12-12 14:38:25 (WARNING): Failed to get data sizes, falling back to uniform partitioning. BalancedBatchSampler requires a dataset that has a metadata attributed with number of atoms.
    2024-12-12 14:38:25 (INFO): rank: 0: Sampler created...
    2024-12-12 14:38:25 (INFO): Created BalancedBatchSampler with sampler=<fairchem.core.common.data_parallel.StatefulDistributedSampler object at 0x7ed8d5e109d0>, batch_size=8, drop_last=False
    /usr/local/lib/python3.10/dist-packages/fairchem/core/trainers/ocp_trainer.py:164: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=self.scaler is not None):
    /usr/local/lib/python3.10/dist-packages/fairchem/core/models/dimenet_plus_plus.py:445: UserWarning: Using torch.cross without specifying the dim arg is deprecated.
    Please either pass the dim explicitly or simply use torch.linalg.cross.
    The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at ../aten/src/ATen/native/Cross.cpp:62.)
      b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
    /usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:825: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
    grad.sizes() = [1, 192], strides() = [1, 1]
    bucket_view.sizes() = [1, 192], strides() = [192, 1] (Triggered internally at ../torch/csrc/distributed/c10d/reducer.cpp:327.)
      return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    2024-12-12 14:38:29 (INFO): energy_mae: 1.66e+01, forces_mae: 8.61e-04, forces_cosine_similarity: 1.55e-01, forces_magnitude_error: 2.25e-03, energy_forces_within_threshold: 0.00e+00, loss: 1.68e+01, lr: 2.00e-05, epoch: 9.09e-01, step: 1.00e+01
    2024-12-12 14:38:32 (INFO): energy_mae: 1.75e+01, forces_mae: 6.67e-04, forces_cosine_similarity: 3.53e-01, forces_magnitude_error: 1.69e-03, energy_forces_within_threshold: 0.00e+00, loss: 1.82e+01, lr: 2.00e-05, epoch: 1.82e+00, step: 2.00e+01
    2024-12-12 14:38:36 (INFO): energy_mae: 1.65e+01, forces_mae: 8.29e-04, forces_cosine_similarity: 3.85e-01, forces_magnitude_error: 2.20e-03, energy_forces_within_threshold: 0.00e+00, loss: 1.68e+01, lr: 2.00e-05, epoch: 2.73e+00, step: 3.00e+01
    2024-12-12 14:38:39 (INFO): energy_mae: 1.64e+01, forces_mae: 4.72e-04, forces_cosine_similarity: 3.32e-01, forces_magnitude_error: 1.28e-03, energy_forces_within_threshold: 0.00e+00, loss: 1.62e+01, lr: 2.00e-05, epoch: 3.64e+00, step: 4.00e+01
    2024-12-12 14:38:42 (INFO): energy_mae: 1.67e+01, forces_mae: 5.07e-04, forces_cosine_similarity: 3.96e-01, forces_magnitude_error: 1.24e-03, energy_forces_within_threshold: 0.00e+00, loss: 1.74e+01, lr: 2.00e-05, epoch: 4.55e+00, step: 5.00e+01
    2024-12-12 14:38:45 (INFO): energy_mae: 1.78e+01, forces_mae: 5.62e-04, forces_cosine_similarity: 5.06e-01, forces_magnitude_error: 1.52e-03, energy_forces_within_threshold: 0.00e+00, loss: 1.76e+01, lr: 2.00e-05, epoch: 5.45e+00, step: 6.00e+01
    2024-12-12 14:38:47 (INFO): energy_mae: 1.60e+01, forces_mae: 4.84e-04, forces_cosine_similarity: 4.39e-01, forces_magnitude_error: 1.29e-03, energy_forces_within_threshold: 0.00e+00, loss: 1.57e+01, lr: 2.00e-05, epoch: 6.36e+00, step: 7.00e+01
    2024-12-12 14:38:49 (INFO): Total time taken: 23.739033460617065

Thanks to the FAIRChem team for their valuable feedback on this tutorial!
