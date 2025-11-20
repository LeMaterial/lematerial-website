---
title: "Datasets"
description: "Datasets"
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

As part of LeMaterial, we released and will maintain different datasets, unifying and standardizing data from existing databases:

- LeMat-Bulk
  - Released in December 2024
  - This dataset unifies data from Materials Project, Alexandria, and OQMD into a high-quality resource with consistent and systematic properties (6,7M entries, 7 material properties)
  - [https://huggingface.co/datasets/LeMaterial/LeMat-Bulk](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk)

- LeMat-BulkUnique
  - Released in December 2024
  - This dataset provides de-duplicated material from Materials Project, Alexandria, and OQMD using our structure fingerprint algorithm. It is available in 3 subsets, for PBE, PBESol, and SCAN functionals
  - [https://huggingface.co/datasets/LeMaterial/LeMat-BulkUnique](https://huggingface.co/datasets/LeMaterial/LeMat-BulkUnique)

- LeMat-Traj
    - Released in August 2025
    - LeMat-Traj provides a large-scale dataset, aggregating over 120 million atomic configurations of ab-initio relaxation trajectories, curated from multiple sources (MP, Alexandria, OQMD) and simulation protocols. It enables training and benchmarking of MLIPs and trajectory-aware models (e.g. force regressors, uncertainty quantifiers).
    - [https://huggingface.co/datasets/LeMaterial/LeMat-Traj](https://huggingface.co/datasets/LeMaterial/LeMat-Traj)
    - [https://arxiv.org/pdf/2508.20875](https://arxiv.org/pdf/2508.20875)

- LeMat-Synth
    - Released in September 2025
    - LeMat-Synth is a multi-modal dataset that links materials, their synthesis procedures, and performance data. It was built by parsing scientific literature using VLM/LLM pipelines, and aims to support research on synthesisability prediction and planning. By analyzing over **80,000 open-access papers**, LeMat-Synth builds one of the first large-scale datasets of material synthesis recipes, covering 35 synthesis methods and 16 material classes.
    - [https://huggingface.co/datasets/LeMaterial/LeMat-Synth](https://huggingface.co/datasets/LeMaterial/LeMat-Synth)
    - [https://www.arxiv.org/pdf/2510.26824](https://www.arxiv.org/pdf/2510.26824)


 More datasets are under development, including surfaces, defects, and electron densities, and we welcome collaborators interested in extending or improving any of the above.
