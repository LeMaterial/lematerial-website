---
title: "LeMaterial with fairchem"
description: "LeMaterial with fairchem"
summary: ""
date: 2023-09-07T16:04:48+02:00
lastmod: 2023-09-07T16:04:48+02:00
draft: false
weight: 820
toc: true
seo:
  title: "" # custom title (optional)
  description: "" # custom description (recommended)
  canonical: "" # custom canonical URL (optional)
  noindex: false # false (default) or true
---

The goal of this notebook is to show how to use LeMaterial's dataset with Geometric GNNs designed for molecular property prediction and relaxation from the [fairchem repository](https://github.com/FAIR-Chem/fairchem).

For more information on how to use fairchem's models, please refer to the [fairchem repository](https://github.com/FAIR-Chem/fairchem) and their [documentation](https://fair-chem.github.io/).

## Setup the environment

The best way to setup an environment for fairchem is to use the provided conda environment file and to create it with the following command:

```bash
wget https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/env.gpu.yml
conda env create -f env.gpu.yml
conda activate fair-chem
```

Or to separately install the torch_geometric dependencies:

```bash
!pip install torch_geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Then we need to install fairchem on the environment:

```bash
git clone https://github.com/FAIR-Chem/fairchem
pip install -e fairchem/packages/fairchem-core
```

```python
%%capture --no-display

!pip install torch_geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

!git clone https://github.com/FAIR-Chem/fairchem
!pip install -e fairchem/packages/fairchem-core
```

```python
CPU = False
BATCH_SIZE = 2
```

## Load the dataset

We use the dataset available at [LeMaterial's Hugging Face space](https://huggingface.co/LeMaterial).

```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

!pip install datasets
!huggingface-cli login --token $HF_TOKEN
```

    Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (3.1.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from datasets) (3.16.1)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (1.26.4)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (17.0.0)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.2.2)
    Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.10/dist-packages (from datasets) (2.32.3)
    Requirement already satisfied: tqdm>=4.66.3 in /usr/local/lib/python3.10/dist-packages (from datasets) (4.66.6)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.5.0)
    Requirement already satisfied: multiprocess<0.70.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.70.16)
    Requirement already satisfied: fsspec<=2024.9.0,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from fsspec[http]<=2024.9.0,>=2023.1.0->datasets) (2024.9.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.11.9)
    Requirement already satisfied: huggingface-hub>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.26.3)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (6.0.2)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (2.4.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: async-timeout<6.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (24.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.1.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (0.2.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.18.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.23.0->datasets) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (2024.8.30)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)
    The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.
    Token is valid (permission: read).
    The token `colab` has been saved to /root/.cache/huggingface/stored_tokens
    Your token has been saved to /root/.cache/huggingface/token
    Login successful.
    The current active token is: `colab`

```python
from datasets import load_dataset

HF_DATASET_PATH = "LeMaterial/LeMat-Bulk"
SUBSET = "compatible_pbe"

dataset = load_dataset(HF_DATASET_PATH, SUBSET)["train"]
```

    Resolving data files:   0%|          | 0/17 [00:00<?, ?it/s]



    Resolving data files:   0%|          | 0/17 [00:00<?, ?it/s]



    Downloading data:   0%|          | 0/17 [00:00<?, ?files/s]



    train-00016-of-00017.parquet:  54%|#####3    | 157M/292M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/5335299 [00:00<?, ? examples/s]



    Loading dataset shards:   0%|          | 0/17 [00:00<?, ?it/s]

## Load a model

We need to start by loading a trained model on which we can run predictions. For example, we can download a checkpoint from EquiformerV2 available [here](https://huggingface.co/yilunliao/equiformer_v2).

```python
HF_REPOID = "fairchem/OMAT24"
# HF_MODEL_PATH = "eqV2_86M_omat_mp_salex.pt"
HF_MODEL_PATH = "eqV2_31M_omat_mp_salex.pt"
```

```python
from huggingface_hub import hf_hub_download
from fairchem.core import OCPCalculator

HF_REPOID = "fairchem/OMAT24"
# HF_MODEL_PATH = "eqV2_86M_omat_mp_salex.pt"
HF_MODEL_PATH = "eqV2_31M_omat_mp_salex.pt"

def download_model(hf_repo_id, hf_model_path):
    model_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_model_path)
    return model_path

model_path = download_model(HF_REPOID, HF_MODEL_PATH)

calc = OCPCalculator(checkpoint_path=model_path, cpu=CPU)
```

    The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.



    0it [00:00, ?it/s]



    eqV2_31M_omat_mp_salex.pt:   0%|          | 0.00/126M [00:00<?, ?B/s]


    /content/fairchem/src/fairchem/core/models/scn/spherical_harmonics.py:23: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
    /content/fairchem/src/fairchem/core/models/equiformer_v2/wigner.py:10: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
    /content/fairchem/src/fairchem/core/models/equiformer_v2/layer_norm.py:75: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /content/fairchem/src/fairchem/core/models/equiformer_v2/layer_norm.py:175: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /content/fairchem/src/fairchem/core/models/equiformer_v2/layer_norm.py:263: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /content/fairchem/src/fairchem/core/models/equiformer_v2/layer_norm.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /content/fairchem/src/fairchem/core/models/escn/so3.py:23: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
    /content/fairchem/src/fairchem/core/common/relaxation/ase_utils.py:190: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    INFO:root:amp: true
    cmd:
      checkpoint_dir: /content/checkpoints/2024-12-10-09-50-56
      commit: 6ab6ad72
      identifier: ''
      logs_dir: /content/logs/wandb/2024-12-10-09-50-56
      print_every: 100
      results_dir: /content/results/2024-12-10-09-50-56
      seed: null
      timestamp_id: 2024-12-10-09-50-56
      version: 1.3.1.dev2+g6ab6ad72
    dataset:
      a2g_args:
        r_energy: true
        r_forces: true
        r_stress: true
      format: ase_db
      transforms:
        decompose_tensor:
          decomposition:
            stress_anisotropic:
              irrep_dim: 2
            stress_isotropic:
              irrep_dim: 0
          rank: 2
          tensor: stress
        element_references:
          file: /fsx-ocp-med/shared/alex-10M/alex-mp-norms-refs/element_references.pt
        normalizer:
          file: /fsx-ocp-med/shared/alex-10M/alex-mp-norms-refs/normalizers.pt
    evaluation_metrics:
      metrics:
        energy:
        - mae
        - mae_density
        forces:
        - mae
        - forcesx_mae
        - forcesy_mae
        - forcesz_mae
        - cosine_similarity
        stress:
        - mae
        - mae_density
        stress_anisotropic:
        - mae
        stress_isotropic:
        - mae
      primary_metric: energy_mae
    gp_gpus: null
    gpus: 0
    logger: wandb
    loss_functions:
    - energy:
        coefficient: 20
        fn: mae_density
    - forces:
        coefficient: 10
        fn: l2mae
    - stress_isotropic:
        coefficient: 1
        fn: mae
    - stress_anisotropic:
        coefficient: 1
        fn: mae
        reduction: mean_all
    model:
      backbone:
        alpha_drop: 0.1
        attn_activation: silu
        attn_alpha_channels: 64
        attn_hidden_channels: 64
        attn_value_channels: 16
        avg_degree: 61.94676351484548
        avg_num_nodes: 31.16592360068011
        distance_function: gaussian
        drop_path_rate: 0.1
        edge_channels: 128
        enforce_max_neighbors_strictly: false
        ffn_activation: silu
        ffn_hidden_channels: 128
        grid_resolution: 18
        lmax_list:
        - 4
        max_neighbors: 20
        max_num_elements: 96
        max_radius: 12.0
        mmax_list:
        - 2
        model: equiformer_v2_backbone
        norm_type: layer_norm_sh
        num_distance_basis: 512
        num_heads: 8
        num_layers: 8
        num_sphere_samples: 128
        otf_graph: true
        proj_drop: 0.0
        share_atom_edge_embedding: false
        sphere_channels: 128
        use_atom_edge_embedding: true
        use_attn_renorm: true
        use_gate_act: false
        use_grid_mlp: true
        use_m_share_rad: false
        use_pbc: true
        use_pbc_single: true
        use_s2_act_attn: false
        use_sep_s2_act: true
        weight_init: uniform
      heads:
        energy:
          module: equiformer_v2_energy_head
        forces:
          module: equiformer_v2_force_head
        stress:
          decompose: true
          module: rank2_symmetric_head
          output_name: stress
          use_source_target_embedding: true
      name: hydra
      otf_graph: true
      pass_through_head_outputs: true
    optim:
      batch_size: 8
      clip_grad_norm: 100
      ema_decay: 0.999
      eval_batch_size: 12
      eval_every: 3000
      load_balancing: atoms
      lr_initial: 0.0002
      max_epochs: 16
      num_workers: 8
      optimizer: AdamW
      optimizer_params:
        weight_decay: 0.001
      scheduler: LambdaLR
      scheduler_params:
        epochs: 741904
        lambda_type: cosine
        lr: 0.0002
        lr_min_factor: 0.01
        warmup_epochs: 463
        warmup_factor: 0.2
    outputs:
      energy:
        level: system
        property: energy
      forces:
        eval_on_free_atoms: true
        level: atom
        property: forces
        train_on_free_atoms: true
      stress:
        decomposition:
          stress_anisotropic:
            eval_on_free_atoms: true
            irrep_dim: 2
            level: system
            parent: stress
            train_on_free_atoms: true
          stress_isotropic:
            eval_on_free_atoms: true
            irrep_dim: 0
            level: system
            parent: stress
            train_on_free_atoms: true
        level: system
        property: stress
    relax_dataset: {}
    slurm:
      account: ocp
      cpus_per_task: 9
      folder: /fsx-ocp-med/lbluque/logs/omat-alex-mp/S2EFS/train/4478177
      gpus_per_node: 8
      job_id: '4478177'
      job_name: eqV2_31M_ft_alexmptraj_e20_f10_s1_cos16
      mem: 480GB
      nodes: 4
      ntasks_per_node: 8
      partition: learn
      qos: ocp_high
      time: 4320
    task: {}
    test_dataset: {}
    trainer: ocp
    val_dataset: {}

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

def get_atoms_from_row(row, add_targets=False):
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
        atoms.calc = SinglePointCalculator(atoms, forces=row["forces"], energy=row["energy"])

      return atoms

atoms = get_atoms_from_row(row)
```

#### Visualize the material with pymatgen

```python
structure = Structure(
    [x for y in row["lattice_vectors"] for x in y],
    species=row["species_at_sites"],
    coords=row["cartesian_site_positions"],
    coords_are_cartesian=True,
)

# TODO: Plot the structure
```

We can now run the inference on the chosen row of the dataset. Since most models inside fairchem are designed to predict the energy and the forces of a material at a given structure (S2EF), we can run relaxation (MD) on the structure to get the energy at the relaxed state as well.

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
    FIRE:    0 10:00:30      -23.413174        0.435615
    Predicted energy: -23.41317367553711 eV
    DFT energy: -23.55740359 eV



    ---------- With Relaxation ----------
          Step     Time          Energy          fmax
    FIRE:    0 10:00:30      -23.413174        0.435615


    /content/fairchem/src/fairchem/core/trainers/ocp_trainer.py:471: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=self.scaler is not None):


    FIRE:    1 10:00:31      -23.415485        0.433691
    FIRE:    2 10:00:32      -23.420135        0.430173
    FIRE:    3 10:00:33      -23.426865        0.425503
    FIRE:    4 10:00:34      -23.435635        0.419379
    FIRE:    5 10:00:35      -23.446156        0.409960
    FIRE:    6 10:00:36      -23.458027        0.392978
    FIRE:    7 10:00:37      -23.469940        0.362187
    FIRE:    8 10:00:38      -23.481274        0.301665
    FIRE:    9 10:00:41      -23.492664        0.205986
    FIRE:   10 10:00:42      -23.507843        0.049325
    Predicted energy: -23.507843017578125 eV
    DFT energy: -23.55740359 eV

## Create an LMDB dataset

In order to run batched inference, we need to create a database compatible with fairchem's dataloader. The recommended way to do is to currently create an ASE LMDB database and pass it to fairchem's config.

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

## Train / fine-tune a model

In order to train models with our dataset, we need to create the train and validation splits as well. This will require specifying the targets in the LMDB and letting the model correctly pick them up.

Since LeMaterial's database is composed of atomic forces and energies at a given structure (no trajectories for now), we want to use the energy and the forces as targets of an S2EF model.

In order to train models with our dataset, we need to create the train and validation splits as well. This will require specifying the targets in the LMDB and letting the model correctly pick them up.

We provide an example of how it is possible to generate a few LMDB datasets and then plug use them for training a model. Notice that we need the targets in here which are directly read by the `Atoms2Graph` class internally.

```python
splits = {
    "train": range(1000),
    "val": range(1000, 2000),
    "test": range(2000, 3000)
}

for split in splits:
  small_dataset = dataset.select(splits[split])

  for row in tqdm.tqdm(small_dataset, total=len(small_dataset)):
      with LMDBDatabase(f"leMat_{split}.aselmdb") as db:
          atoms = get_atoms_from_row(row, add_targets=True)

          db.write(atoms, data={"id": row["immutable_id"]})
```

Many model implementations exist in fairchem. We provide an example of a few and how we use them. More of these can be found [here](https://github.com/FAIR-Chem/fairchem/tree/main/configs).

```python
example_configs = {
    "eqv2_omat_M": "configs/omat24/all/eqV2_86M.yml",
    "eqv2_omat_S": "configs/omat24/all/eqV2_31M.yml",
    "eqv2_s2ef_L": "configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml",
    "eqv2_s2ef_S": "configs/s2ef/all/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml",
    "dpp": "configs/s2ef/all/dimenet_plus_plus/dpp.yml",
}

CHOSEN_MODEL = "dpp"
```

We need to apply a little bit of processing on the yaml config files to be able to read them with the main script of fairchem.

- Apply the datasets
- Change some config parameters (this can be adjusted depending on what you want to put in the model)

More information on how to tweak this config file can be found in fairchem's documentation as well. Note that you can also modify some config arguments with the cli parameters directly.

```python
from fairchem.core.common.tutorial_utils import fairchem_main
from pathlib import Path

import yaml
yaml_obj = yaml.load(open(fairchem_main().parent / example_configs[CHOSEN_MODEL], "r"), Loader=yaml.FullLoader)

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

# No metadata.npz file, disabling load_balancing
if "load_balancing" in yaml_obj["optim"]:
  del yaml_obj["optim"]["load_balancing"]

new_yaml_path = Path(f"/tmp/{CHOSEN_MODEL}_leMat.yml")
with open(new_yaml_path, "w") as f:
    yaml.dump(yaml_obj, f)
```

```python
# # If you want to manipulate the trainer object:
# from fairchem.core import OCPCalculator
# ocpcalculator = OCPCalculator(yaml_obj, cpu=True)
```

```python
!python {fairchem_main()} --mode train --config-yml /tmp/{CHOSEN_MODEL}_leMat.yml {"--cpu" if CPU else ""}
```

#### Fine-tune from checkpoint

```python
yml = generate_yml_config(model_path, '/tmp/config.yml',
                   delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes',
                           'optim.loss_force', # the checkpoint setting causes an error
			   'optim.load_balancing',
                           'dataset', 'test_dataset', 'val_dataset'],
                   update={'gpus': 1,
                           'optim.eval_every': 10,
                           'optim.max_epochs': 1,
                           'optim.batch_size': 4,
                            'logger':'tensorboard', # don't use wandb!
                           # Train data
                           'dataset.train.src': 'train.aselmdb',
                           'dataset.train.format': 'ase_db',
                           'dataset.train.a2g_args.r_energy': True,
                           'dataset.train.a2g_args.r_forces': True,
                            # Test data
                           'dataset.test.src': 'test.aselmdb',
                           'dataset.test.format': 'ase_db',
                           'dataset.test.a2g_args.r_energy': False,
                           'dataset.test.a2g_args.r_forces': False,
                           # val data
                           'dataset.val.src': 'val.aselmdb',
                           'dataset.val.format': 'ase_db',
                           'dataset.val.a2g_args.r_energy': True,
                           'dataset.val.a2g_args.r_forces': True,
                          })
```

```python
!python {fairchem_main()} --mode train --config-yml {yml} --checkpoint {model_path} --run-dir fine-tuning --cpu > train.txt 2>&1
```
