---
title: "Overview"
description: "What LeMat-Synth extracts and how the pipeline works."
summary: ""
date: 2025-10-28T00:00:00+00:00
lastmod: 2025-10-28T00:00:00+00:00
draft: false
weight: 501
toc: true
seo:
  title: "LeMat-Synth Overview"
  description: "What LeMat-Synth extracts and how the pipeline works."
  canonical: ""
  noindex: false
---

**LeMat-Synth** is an open-source multi-modal toolbox for extracting structured synthesis procedures and performance data from materials science literature at scale. It was presented at [NeurIPS AI4Mat 2025](https://arxiv.org/abs/2510.26824).

- [GitHub](https://github.com/LeMaterial/lematerial-llm-synthesis)
- [arXiv paper](https://arxiv.org/abs/2510.26824)
- [HuggingFace dataset](https://huggingface.co/datasets/LeMaterial/LeMat-Synth)
{.no-bullets}

Tutorial and documentation were provided by [Stefano Ribes](https://www.linkedin.com/in/stefano-ribes-1379b1107?trk=feed-detail_main-feed-card-text)!

---

LeMat-Synth is an extensible pipeline that parses scientific PDFs using large language models (LLMs) and vision-language models (VLMs) to produce structured, machine-readable records of material synthesis and performance.

<img src="/images/publications/lemat-synth-figure1.png" alt="LeMat-Synth pipeline figure" style="max-width:100%">

## What you get

For each paper processed, LeMat-Synth outputs:

**Synthesised materials** — The actual materials fabricated in the paper, distinguished from materials merely cited or compared against.

**Structured procedures** — Step-by-step synthesis recipes including reagents with amounts and purity, temperatures, durations, and atmospheres.

**Performance data** — Quantitative values extracted from figures in the paper using VLM technology, including axis labels and plot coordinates.

**Quality scores** — An LLM-based rating across five dimensions to indicate extraction confidence.

A typical output record looks like this:

```json
{
  "material": "Ru/MgO(110)",
  "synthesis_method": "wet impregnation",
  "reagents": [
    {"name": "RuCl3", "amount": "0.15 g", "purity": "99.9%"}
  ],
  "steps": [
    {"description": "Dissolve precursor in water", "temperature": "25°C", "duration": "30 min"},
    {"description": "Impregnate support", "atmosphere": "N2"},
    {"description": "Calcine", "temperature": "500°C", "duration": "4 h", "atmosphere": "air"}
  ],
  "performance": {
    "metric": "NH3 conversion",
    "value": 78.5,
    "unit": "%",
    "conditions": "400°C"
  },
  "quality_scores": {
    "completeness": 0.9,
    "consistency": 0.85
  }
}
```

## How it works

The pipeline has four composable components:

1. **PlotFilterConfig** — Screens figures by axis labels and units to identify relevant performance plots (e.g., `"Temperature"` × `"Conversion %"`).
2. **Material Extraction Prompts** — LLM instructions for identifying domain-specific materials from text.
3. **Optional Metric Extractors** — Text-based or vision-based scalar extraction (e.g., bandgap, onset voltage, critical temperature).
4. **Output Writers** — JSON or CSV serialisation with custom schemas.

## Scale and coverage

The LeMat-Synth v1.0 dataset was built from over **80,000 open-access papers** and covers:

- **35 synthesis methods** — including wet impregnation, sol-gel, chemical vapour deposition, hydrothermal synthesis, and more
- **16 material classes** — oxides, nitrides, carbides, MOFs, zeolites, perovskites, and others
- Three fully worked case studies: thermocatalysis, superconductors, and porous materials

## Citation

If you use LeMat-Synth in your research, please cite:

```bibtex
@misc{lederbauer2025lematsynth,
  title={LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature},
  author={Magdalena Lederbauer et al.},
  year={2025},
  eprint={2510.26824},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
