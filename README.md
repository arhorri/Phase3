# Machine Learning-Assisted Microstructure Image Prediction
## Conditional Denoising Diffusion Probabilistic Model for the Cast-Forged AZ80 Microstructure

This is the implementation of conditional denoising diffusion probabilistic model to generate SEM microstructure images of seen and unseen cast-forged AZ80 magnesium alloy components. The model is conditioned based on the cast geometry, casting cooling rate, soaking process, pre-forging heat treatment, forging temperature, location of extracted metallography sample, and magnification. 


Please cite our paper as follows:
E. Azqadan, H. Jahed, A. Arami, Predictive microstructure image generation using denoising diffusion probabilistic models, Acta Mater. 261 (2023), 119406, https://doi.org/10.1016/j.actamat.2023.119406.


The following repositories were used as inspiration for this implementation:
https://github.com/dome272/Diffusion-Models-pytorch/tree/main
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/CompVis/latent-diffusion

## Training notebook (`az80-image-generation.ipynb`)

The notebook in this repo trains the conditional DDPM from randomly initialized weights, on Kaggle's free GPU tier, following the architecture and methodology in the paper (Section 2.2).

### Running it on Kaggle

1. Open a new Kaggle Notebook and enable **GPU** under *Settings → Accelerator*.
2. *File → Add Input → GitHub* and add this repository (`arhorri/Phase3`).
3. *File → Add Input → Datasets* and add the **`az80-microstructure-data`** dataset. The notebook expects it to mirror this repo's `data/` layout: `Training/<series>/*.jpg`, `Testing/<series>/*.jpg`, `Training Labels.xlsx`, `Testing Labels.xlsx`.
4. Run all cells. Checkpoints and the final model are written under the working directory (see Folder structure below).

Running locally instead (`git clone git@github.com:arhorri/Phase3.git`) works the same way — the notebook falls back to the `data/` folder at the repo root automatically.

### Main hyperparameters

| Parameter | Value |
|---|---|
| Image size | 512×512, single-channel (grayscale), normalized to [-1, 1] |
| Diffusion steps | 1000 |
| Beta schedule | linear, `1e-4` → `0.02` |
| Batch size | 4 |
| Optimizer | Adam, `lr = 3e-4` |
| Loss | MSE (true noise vs. predicted noise) |
| EMA | `beta = 0.995`, starts averaging after 2000 steps |
| Iterations | 3600 (paper default — "more than 130 h of computation"; configurable, with checkpoint resume for multi-session runs) |
| Train/test split | 87:13, leave-one-category-out (114 vs. 17 of 131 image classes) |

### Folder structure

```
data/                 Training/Testing SEM images + process-parameter labels (local only, see .gitignore)
checkpoints/           periodic training checkpoints (model + EMA model + optimizer state)
models/                final trained model, saved at the end of training
```

