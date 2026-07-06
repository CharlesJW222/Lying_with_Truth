# Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage (ACL 2026 Oral)

[![ACL 2026 Oral](https://img.shields.io/badge/ACL%202026-Oral-blue)](https://aclanthology.org/2026.acl-long.270/)

## Overview

This directory contains demo materials for our recent work on cognitive collusion attacks against LLM-based agents. We demonstrate how coordinated agents can manipulate victim beliefs using only truthful evidence fragments through strategic narrative construction.

## Repository Contents

```
.
├── README.md
├── dataset/
│   └── CoPHEME/              # Complete CoPHEME dataset
├── Montage/
│   └── attack_plans/         # Pre-generated attack sequences using GPT-4.1-mini
│       ├── charlie_hebdo/
│       ├── sydney_siege/
│       ├── ferguson/
│       ├── ottawa_shooting/
│       ├── germanwings/
│       └── putin_missing/
|   └──src/ Demo code for our framework
├── requirements.txt          # Python dependencies
└── environment.yml           # Conda environment file
```

## Quick Start

### Installation

**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

**Option 2: Using conda**
```bash
conda env create -f environment.yml
conda activate colusive_agent
```

### Running the Demo

```bash
./run_demo.sh
```

**Example output:**
```
[Event: Charlie Hebdo]
Target Hypothesis: Ahmed Merabet was the first victim
Attack Success Rate: 81.7%
Average Confidence: 0.83
High-Confidence ASR: 67.4%
```

## Dataset: CoPHEME

The **CoPHEME** dataset extends PHEME for cognitive collusion research:

- **6 rumor events**: Charlie Hebdo, Sydney Siege, Ferguson, Ottawa Shooting, Germanwings, Putin Missing
- **evidence fragments**: Verified truthful tweets satisfying Local Truth (LT=1)
- **target fabrications**: High-impact false narratives for attack simulation

See `CoPHEME/` for complete statistics.

## Pre-computed Attack Plans

To facilitate reproducibility without requiring extensive API calls, we provide pre-generated attack sequences produced by our **Generative Montage** framework using GPT-4.1-mini:

- **Writer-Editor-Director** optimized narratives
- **Validated** montage sequences (passed Director acceptance threshold τ=7.0)
- **Ready-to-use** for easy victim evaluation across different LLM families

## API Configuration

Set your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# For open-weights models (if using hosted APIs)
export HUGGINGFACE_API_KEY="your-key-here"
```
## Citation

If you find our idea is helpful, please cite:

```bibtex
@inproceedings{hu-etal-2026-lying,
    title = "Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage",
    author = "Hu, Jinwei  and
      Huang, Xinmiao  and
      Sun, Youcheng  and
      Dong, Yi  and
      Huang, Xiaowei",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.270/",
    doi = "10.18653/v1/2026.acl-long.270",
    pages = "5979--5996",
    ISBN = "979-8-89176-390-6",
    abstract = "As large language models (LLMs) transition to autonomous agents synthesizing real-time information, their reasoning capabilities introduce an unexpected attack surface. This paper introduces a novel threat where colluding agents steer victim beliefs using only truthful evidence fragments distributed through public channels, without relying on covert communications, backdoors, or falsified documents. By exploiting LLMs' overthinking tendency, we formalize the first **cognitive collusion attack** and propose **Generative Montage**: a Writer-Editor-Director framework that constructs deceptive narratives through adversarial debate and coordinated posting of evidence fragments, causing victims to internalize and propagate fabricated conclusions. To study this risk, we develop **CoPHEME**, a dataset derived from real-world rumor events, and simulate attacks across diverse LLM families. Our results show pervasive vulnerability across 14 LLM families: attack success rates reach 74.4{\%} for proprietary models and 70.6{\%} for open-weights models. Counterintuitively, stronger reasoning capabilities increase susceptibility, with reasoning-specialized models showing higher attack success than base models or prompts. Furthermore, these false beliefs then cascade to downstream judges, achieving over 60{\%} deception rates, highlighting a socio-technical vulnerability in how LLM-based agents interact with dynamic information environments."
}
```
