# ReLoRA: Paper Implementation and Experiments

## Project by:
- **Omar Zoloev**
- **Konstantin Zorin**
- **Alex Boriskin**
- **Ivan Lisitsyn**
- **Nikita Vakhrameev**

---

## Project Description

This repository contains the implementation and experiments for the ReLoRA paper. The project is structured to facilitate easy navigation and reproducibility of the experiments.

Read the paper: [ReLoRA: High-Rank Training Through Low-Rank Updates](https://arxiv.org/pdf/2307.05695)

## Directory Structure

- **Data:** `.../src/data`
  - Contains all the datasets used for the experiments.
  
- **Modules:** `.../src/modules`
  - Contains all the modules and scripts for the implementation of ReLoRA.

## Dataset and Metric

The dataset and metric used for this project were taken from a competition on Kaggle [Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data)

### Metric

```python
import sklearn
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(y_pred, y_true):
    return cohen_kappa_score(
        y_true.astype(int),
        y_pred.clip(0, 5).round(0),
        weights='quadratic',
    )
