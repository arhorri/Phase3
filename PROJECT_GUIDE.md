# Project Guide: Generating Microstructure Images with a Diffusion Model

A plain-language walkthrough of this repository: what it solves, what data it uses, how the data is prepared, how the model works, and how to train it yourself.

Based on the paper *"Predictive microstructure image generation using denoising diffusion probabilistic models"* (Azqadan, Jahed, Arami — Acta Materialia 261, 2023) and the code in this repo.

---

## 1. The problem this project solves

### The material science background
**AZ80** is a lightweight magnesium alloy. It is used for car and aircraft parts. To make a part, engineers **cast** the metal, optionally **heat-treat** it, then **forge** it (squeeze it hot into shape).

Each choice in that process (how fast it cooled, how long it was heated, what temperature it was forged at, and so on) changes the metal's **microstructure**: the tiny internal pattern of grains and particles. Microstructure decides strength and toughness.

Scientists look at microstructure with a **scanning electron microscope (SEM)**, which gives grayscale photos. Getting those photos is slow and expensive. You have to cast, forge, cut, polish, etch and image a real sample for every process combination you want to study.

### The goal
> **Given a set of process parameters, generate a realistic SEM image of what the microstructure would look like, without making the sample.**

The model should work for combinations it saw during training, and also for **unseen** combinations it was never shown. That second part is the real point: it acts as a "virtual microscope" that predicts microstructure for new process routes.

### The seven inputs (process parameters)

| # | Parameter | Possible values |
|---|---|---|
| 1 | Cast geometry (shape of the starting piece) | cylinder, pre-form (2 options) |
| 2 | Location the sample was cut from | tall flange, web, short flange (3) |
| 3 | Casting cooling rate | 1.5, 6, 10.4 °C/s (3) |
| 4 | Soaking process | normal, 1.5 h, 2 h (3) |
| 5 | Pre-forging heat treatment | none, homogenization (2) |
| 6 | Forging temperature | 250, 300, 350 °C (3) |
| 7 | Microscope magnification | 100×, 500×, 1000×, 1500×, 2000×, 3000× (6) |

Output: one 512×512 grayscale SEM image.

The paper checks quality two ways. One is a numeric image-similarity score (FID). The other is measuring real metallurgical features (grain size, fraction of the Mg₁₇Al₁₂ intermetallic phase, recrystallized area) on real versus generated images. Those measurements agreed to within about 6–7 %.

---

## 2. The dataset

### What it contains
- **434 grayscale SEM images** in total, roughly 1140×768 pixels each (the paper says 1024×768).
- Made from **27 metallographic samples** taken from 11 cast-forged I-beam components.
- Images are grouped into **series**. A series is one sample at one magnification, e.g. `CM01-0500`.
- Each series folder holds a few images (up to about 9).

### How to read a series name
`CM01-0500` breaks down as:
- `C` or `P`: cylinder or pre-form cast.
- `M` / `S` / `L`: web, short flange or tall flange (location).
- `01`: component number.
- `0500`: magnification (500×).

### Where it lives

```
data/
├── Training/            114 series folders, 396 images
├── Testing/              17 series folders,  38 images
├── Training Labels.xlsx  the 7 parameter values for each training series
├── Testing Labels.xlsx   the same for each testing series
└── data.zip              a zipped copy of the above
```

The labels files are spreadsheets. Each column is one series. The rows are `Label`, `Class`, `Shape`, `Location`, `Cooling rate`, `Soaking`, `Heat-treatment`, `Forging Temp` and `Magnification`. The values are small integers (0, 1, 2, …) that stand for the options in the table above.

### Which dataset should you use to retrain?
**Use `data/Training/` with `data/Training Labels.xlsx` to train.**
Keep `data/Testing/` (with `Testing Labels.xlsx`) out of training. It is the "unseen conditions" test set, used only to check how well the model predicts new combinations.

Why the split looks odd: it is **leave-one-category-out**, not a random split. The 17 test series are whole process conditions the model never sees, such as all samples of one I-beam. That makes the test a fair check of prediction. The split is 114 : 17 series, about 87 % : 13 %.

Notes:
- `data/` is in `.gitignore`. It is a local copy and is not pushed to GitHub. Do not regenerate or bulk-edit it.
- The paper says the data is available "on request", so treat it as reference data.
- The maintained notebooks read this folder layout directly. The older `ddpm.py` script instead expects a different folder, `All-Dataset/Whole Cropped/`, that this repo does not contain.

---

## 3. Data preprocessing

The steps, in order (in the notebook they live in the `AZ80SeriesDataset` class):

1. **Load and convert to grayscale.** Each image is opened and turned into a single channel. The model works with 1 channel, not 3.
2. **Random crop to 512×512.** The originals are about 1140×768. Every time an image is fetched, a random 512×512 window is cut out. Each visit to the same photo therefore gives a slightly different piece, which acts as a free form of data augmentation. (The notebook also pads with reflection if an image is ever smaller than 512.)
3. **Scale pixel values to [−1, 1].** Pixels start at 0–255, become 0–1, then `x * 2 − 1`. Diffusion models expect roughly centered values.
4. **Random flips (training only).** Horizontal and vertical flips, each with 50 % probability. This is safe because microstructure looks the same whichever way it is rotated or mirrored.
5. **Attach the 7 labels.** Each image is paired with its series' 7 integer parameters from the labels spreadsheet. These are used as conditions, not as a single "class number".

Batches are made of image tensors shaped `[batch, 1, 512, 512]` plus 7 label integers per image.

There is no manual cleaning or resizing step. Cropping and normalization happen on the fly.

---

## 4. The model, step by step

### The big idea: a diffusion model (DDPM)
1. **Forward process (training only).** Take a real image and gradually add random noise over **1000 steps** until only static remains. There is a formula to jump straight to any step `t`.
2. **Learning.** A neural network is shown a noisy image, the step number `t` and the 7 parameters. Its job is to **predict the noise that was added**. The training loss is simply the mean squared error between the true noise and the predicted noise.
3. **Generation (reverse process).** Start from pure random noise. Repeat 999 times: ask the network "what noise is in this?", subtract a bit of it, and add a little fresh randomness. After the loop, a clean, realistic image is left. The 7 parameters steer which kind of microstructure appears.

Noise schedule: linear, from `1e-4` to `0.02` over 1000 steps.

### Exact inputs and outputs of the network (`UNet_conditional`)

| | What | Shape / type |
|---|---|---|
| Input 1 | Noisy image `x` | `[B, 1, 512, 512]` floats in about [−1, 1] |
| Input 2 | Time step `t` | `[B]` integers, 1 to 999 |
| Inputs 3–9 | Shape, Location, Cooling rate, Soaking, Heat treatment, Forging temp, Magnification | 7 tensors, each `[B]` integers |
| **Output** | **Predicted noise** | `[B, 1, 512, 512]`, same shape as the input image |

For the whole pipeline, the input is the 7 parameters and the output is a 512×512 grayscale image. The network is called about 1000 times to produce it.

### Architecture: a conditional U-Net
A U-Net first shrinks the image while learning increasingly abstract features, then expands it back to full size. Skip connections copy details from the shrinking side to the growing side.

**Step 0: Input block.** Two 3×3 convolutions turn 1 channel into 16 channels at 512×512.

**Steps 1–6: Encoder (going down).** Each "Down" block does the following:
- halves the image with max-pooling;
- applies a residual double-convolution, then another double-convolution (four convolutions in total, with group normalization and GELU activation);
- adds the **time step** as a sinusoidal embedding, passed through a small linear layer;
- appends the **7 condition channels** (described below).

| Block | Resolution | Channels |
|---|---|---|
| Input | 512×512 | 16 |
| Down 1 | 256×256 | 32 |
| Down 2 | 128×128 | 64 |
| Down 3 | 64×64 | 128 |
| Down 4 | 32×32 | 256 |
| Down 5 | 16×16 | 512 |
| Down 6 | 8×8 | 512 |

Self-attention (which lets distant parts of the image "look at" each other) is applied after Down 3 through Down 6.

**Step 7: Bottleneck.** Three double-convolutions at 8×8 with 512 channels.

**Steps 8–13: Decoder (going up).** Each "Up" block does the following:
- upsamples ×2 (bilinear);
- concatenates the matching skip connection from the encoder;
- applies two double-convolutions;
- appends the 7 condition channels and adds the time embedding again.

It goes 16×16, 32×32, 64×64, 128×128, 256×256, 512×512, ending at 8 channels. Self-attention follows the four blocks from 16×16 up to 128×128.

**Step 14: Output.** A 1×1 convolution turns 8 channels into 1, the predicted noise image.

### How the 7 parameters are injected (the unusual part)
Most conditional diffusion models add one class embedding once. Here, **at every Down and Up block**:
- each of the 7 parameters has its own lookup table (`nn.Embedding`, 100 numbers per option);
- a linear layer stretches it to a full `H×W` map, where H×W is that block's resolution;
- the 7 maps are stacked onto the feature maps as **7 extra channels**.

So the network is reminded of the process conditions at every scale, going down and going up.

### Training details worth knowing
- **Optimizer:** Adam, learning rate `3e-4`. **Batch size:** 4.
- **EMA:** a second copy of the model keeps a slow-moving average of the weights (β = 0.995, starts after 2000 steps). Use the **EMA model** to generate the final images. It gives smoother results.
- **Mixed precision (AMP):** the notebooks use it so batch size 4 fits on a 15–16 GB GPU. The attention layer on the 128×128 map is very memory hungry.
- **Paper's schedule:** 3600 iterations (about 130 hours on a P100 GPU), stopped when quality stopped improving.
- **Unused feature:** the code supports classifier-free guidance (`cfg_scale`), but training never drops the labels, and sampling always uses `cfg_scale=0`, so it is effectively off.

---

## 5. How to train from scratch on a new dataset

The maintained way is the notebooks: `az80-image-generation.ipynb` (Kaggle or local) and `az80-image-generation-colab.ipynb` (Google Colab). They train from random weights and need no pre-existing checkpoint. The older `ddpm.py` script needs a checkpoint file and other paths that don't exist here, so the notebooks are the easier route.

You need a **GPU** with about 15 GB of memory (a T4 or P100 is enough). Training on CPU is impractical.

### Step 1: Prepare your images
- Use grayscale SEM-style images at least 512×512. Larger is fine, since they are randomly cropped. Smaller images get padded.
- Put them in **series folders**, one folder per unique combination of conditions:

```
my_dataset/
├── Training/
│   ├── SeriesA-0500/   img1.jpg  img2.jpg ...
│   └── SeriesB-1000/   ...
├── Testing/            (held-out conditions; same structure)
├── Training Labels.xlsx
└── Testing Labels.xlsx
```

### Step 2: Write the labels spreadsheets
Copy the layout of `data/Training Labels.xlsx`:
- One **column per series folder**. The column header must match the folder name exactly.
- The first column holds the row names, in this order: `Label`, `Class`, `Shape`, `Location`, `Cooling rate`, `Soaking`, `Heat-treatment`, `Forging Temp`, `Magnification`.
- The value in each cell is an integer starting at 0.

The notebook loader requires both `Training/`, `Testing/` and both xlsx files to exist, or it stops with an error.

### Step 3: Match the model to *your* parameters
If your new dataset has **different parameters** (other alloy, other options), edit the model's settings:
- **Vocabulary sizes.** Change `shapes`, `locations`, `cooling_rates`, `soaking_times`, `heat_treatments`, `forging_temps`, `magnifications` in the notebook's model cell so each equals the number of distinct values in your labels. These set the embedding table sizes.
- **Different number of parameters** (say 5 instead of 7). This is a bigger change. The `- 7` and the 7 embedding layers in `Down`/`Up`, the 7 arguments of `forward()`, and `CONDITION_COLUMNS` must all be updated together.
- **Different image size.** The `imsize` values passed to each block (256, 128, 64, …) and the attention `size` values assume 512×512. If you change the resolution, update all of them.

If your dataset uses the **same 7 parameters and value ranges**, you can skip this step.

### Step 4: Set up the environment
- **Kaggle:** create a notebook, enable a GPU, add this GitHub repo as an input, and add your dataset as a Kaggle Dataset (any name). The notebook searches for it automatically.
- **Colab:** upload the dataset folder to Google Drive, open the Colab notebook, choose a GPU runtime, run the setup cell and authorize Drive.
- **Local:** clone the repo and put your dataset in a `data/` folder at the repo root. You need Python with `torch`, `torchvision`, `numpy`, `pandas`, `openpyxl`, `pillow`, `matplotlib` and `tqdm`. There is no requirements file, so install these yourself.

### Step 5: Do a quick smoke test
Set `NUM_ITERATIONS` to something tiny (like 20) and run all cells. Confirm it loads the data, prints the series and image counts, trains without running out of memory, and saves a checkpoint. If you hit an out-of-memory error, lower `BATCH_SIZE`.

### Step 6: Run the full training
Set the settings in the "Training Configuration" cell:

| Setting | Default | Meaning |
|---|---|---|
| `NUM_ITERATIONS` | 3600 | Total training steps |
| `LEARNING_RATE` | 3e-4 | Adam step size |
| `BATCH_SIZE` | 4 | Images per step |
| `CHECKPOINT_EVERY` | 600 | Save a checkpoint this often (plus one at iteration 30, via `EXTRA_CHECKPOINT_ITERS`) |
| `SAMPLE_EVERY` | 500 | Draw progress images this often |
| `RESUME_FROM` | `None` | Path of a checkpoint to continue from |

Then run all cells. In each iteration the loop:
1. takes a batch of images and labels;
2. picks a random noise step `t` for each image;
3. adds that much noise to the images;
4. asks the model to predict the noise;
5. computes the MSE loss and updates the weights;
6. updates the EMA copy.

### Step 7: Watch progress
- Loss prints every 50 iterations.
- Every 500 iterations it draws sample images (from the EMA model) for a fixed set of held-out conditions, so you can see quality improve. They are also saved to `generated_samples/`.
- Checkpoints go to `checkpoints/` (model, EMA model, optimizer, iteration).
- Kaggle and Colab sessions time out. To continue, set `RESUME_FROM` to the latest checkpoint. The Colab notebook resumes automatically from Drive.

### Step 8: Save and evaluate
- The final model is saved to `models/az80_ddpm_final.pth.tar`.
- Run the sampling cells to generate images for any parameter combination. Try one seen condition and one unseen one, and compare with a real image.
- Section 9 of the notebooks redraws the paper's result figures: Fig. 2 (FID score vs. iteration plus generation progress), Fig. 3 (synthesized vs. real for seen conditions) and Fig. 4 (unseen conditions). The PNGs go to `generated_samples/paper_figures/`.
- FID needs the `pytorch-fid` package (installed by the notebook, needs internet). Generation is slow, so the results are cached.
- The paper's feature measurements (DRX grain size, Mg₁₇Al₁₂ area fraction) are still not included, so you would have to add those yourself.

### Step 9: Generate images later
Use `image_generator.py` or `az80-image-generation.ipynb`'s sampling cells. Set the 7 parameter variables at the top and point the checkpoint path (`load_dir`) at your `.pth.tar` file. The original file has a hardcoded Kaggle path that must be changed.

---

## Things to be aware of

- **The code exists in several copies** (`ddpm.py`+`modules.py`+`utils.py`, `image_generator.py`, and both notebooks). They have drifted apart. If you fix a bug, check every copy.
- **The original `ddpm.py` script is not ready to run.** It loads a checkpoint that must already exist, uses different paths, and reads a dataset folder this repo does not have. It also defines `class_table` for the 114 training series only.
- **There are no tests, linters or package files.** This is research code.
- **Training is slow** even on good hardware (the paper reports 130+ hours). Expect a long run, and use the checkpoint and resume feature.
- **Small dataset.** With 396 training images the model can only learn what those conditions show. Predictions for unseen conditions work best when they sit between conditions it has seen.
