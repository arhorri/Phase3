# Machine Learning-Assisted Microstructure Image Prediction
## Conditional Denoising Diffusion Probabilistic Model for the Cast-Forged AZ80 Microstructure

This is the implementation of conditional denoising diffusion probabilistic model to generate SEM microstructure images of seen and unseen cast-forged AZ80 magnesium alloy components. The model is conditioned based on the cast geometry, casting cooling rate, soaking process, pre-forging heat treatment, forging temperature, location of extracted metallography sample, and magnification. 


Please cite our paper as follows:
E. Azqadan, H. Jahed, A. Arami, Predictive microstructure image generation using denoising diffusion probabilistic models, Acta Mater. 261 (2023), 119406, https://doi.org/10.1016/j.actamat.2023.119406.


The following repositories were used as inspiration for this implementation:
https://github.com/dome272/Diffusion-Models-pytorch/tree/main
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/CompVis/latent-diffusion

## Training notebooks

Two notebooks train the same conditional DDPM from randomly initialized weights, following the architecture and methodology in the paper (Section 2.2) — pick whichever free GPU host you're using:

- **`az80-image-generation.ipynb`** — for Kaggle.
- **`az80-image-generation-colab.ipynb`** — for Google Colab (data + checkpoints on Google Drive instead of a Kaggle dataset input; auto-resumes from the latest Drive checkpoint across sessions).

### Running it on Kaggle

1. Open a new Kaggle Notebook and enable **GPU** under *Settings → Accelerator*.
2. *File → Add Input → GitHub* and add this repository (`arhorri/Phase3`).
3. *File → Add Input → Datasets* and add a dataset that mirrors this repo's `data/` layout: `Training/<series>/*.jpg`, `Testing/<series>/*.jpg`, `Training Labels.xlsx`, `Testing Labels.xlsx` (any dataset name works — the notebook searches Kaggle's mounted inputs structurally rather than by name).
4. Run all cells. Checkpoints and the final model are written under the working directory (see Folder structure below).

Running locally instead (`git clone git@github.com:arhorri/Phase3.git`) works the same way — the notebook falls back to the `data/` folder at the repo root automatically.

### Running it on Google Colab

1. Upload a dataset folder mirroring `data/`'s layout to Google Drive (any location under My Drive, any folder name).
2. Open `az80-image-generation-colab.ipynb` in Colab (or File → Open notebook → GitHub → `arhorri/Phase3`).
3. Runtime → Change runtime type → GPU.
4. Run the Setup cell and authorize Drive access when prompted, then run the rest of the notebook. Checkpoints/model/samples are written to Drive (`My Drive/az80_ddpm_outputs/` by default) and training auto-resumes from the latest checkpoint there on a fresh runtime.

### Reproducing the paper's figures (Fig. 2, 3, 4)

Section 9 of both notebooks redraws the paper's result figures from your trained model, laid out like the originals, and saves them as PNGs under `generated_samples/paper_figures/`:

- **9.1 — Fig. 2:** FID score vs. training iteration (with the real-vs-real baseline) plus one image sharpening from noise across checkpoints. Needs `pytorch-fid` (installed by the notebook; requires internet) and the checkpoints the training loop saves at iteration 30 and every 600 iterations.
- **9.2 — Fig. 3:** synthesized vs. real images for eight seen conditions, with process parameters and I-beam sample location under each pair.
- **9.3 — Fig. 4:** synthesized vs. real images for unseen conditions (one-I-beam-out, one-sample-out, one-magnification-out), all taken from `Testing/`.

Sampling is slow (about 1000 U-Net passes per image), so generated images are cached and interrupted runs resume. Before drawing anything, Section 9 frees the training-only tensors (`model`, optimizer state) and keeps only the EMA model, since at 512x512 each copy of this model is ~1.4 GiB and a 12.7 GB Colab runtime can't hold them on the CPU either; re-run Section 5 with `RESUME_FROM` before training further in the same session. With few images the FID's absolute values aren't comparable to the paper's; use the trend. The coloured Mg₁₇Al₁₂ phase highlights are an automatic approximation of the paper's hand annotations (`HIGHLIGHT_PHASES = False` turns them off).

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

Checkpoints are written at iteration 30 and every 600 iterations (the paper's Fig. 2 panels).

### Folder structure

```
data/                 Training/Testing SEM images + process-parameter labels (local only, see .gitignore)
checkpoints/           periodic training checkpoints (model + EMA model + optimizer state)
models/                final trained model, saved at the end of training
```

