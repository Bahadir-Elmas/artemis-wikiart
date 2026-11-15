# Artist-disjoint evaluation of frozen visual representations for art movement classification

Code, data splits and results for the paper

> B. Elmas, "Artist-Disjoint Evaluation of Frozen Visual Representations for Art Movement Classification:
> Pretraining Type, Artist Leakage, and Variability," manuscript submitted for publication, 2026.

Twenty painting movements of WikiArt (65,998 paintings, 936 artists) are classified from frozen features of
ten image encoders. A small classifier network is trained on the features. It is evaluated on splits in which no
artist appears in both the training and the test set: 10-fold cross-validation and 10 stratified repeated
splits (SSS), both grouped by artist.

[![DOI](https://zenodo.org/badge/1436305452.svg)](https://doi.org/10.5281/zenodo.23709917)

- Code, splits and results (this repository, archived on Zenodo): [10.5281/zenodo.23709917](https://doi.org/10.5281/zenodo.23709917)
- Extracted features: [10.5281/zenodo.23710694](https://doi.org/10.5281/zenodo.23710694)

## Contents

```
notebooks/
  01_dataset_preparation.ipynb   metadata table, the 20 movements, list of paintings
  02_resnet50_features.ipynb     ResNet50 features (TensorFlow/Keras)
  02B_gpu_features.ipynb         features of the nine PyTorch configurations (timm / OpenCLIP)
  03_classification.ipynb        classifier network, artist-disjoint splits, evaluation, split export
analysis/
  paired_tests.py                Nadeau–Bengio corrected t-tests, Wilcoxon, Holm (Table 6)
  per_class_metrics.py           per-movement recall / precision / F1 and confusions (Table 10)
  robustness_analyses.py         confidence intervals, selection check, confusion-matrix analyses,
                                 robustness experiments (Table 11)
  external_validation.py         external validation on National Gallery of Art paintings (Table 12)
  figure3.py, figure4.py         Figures 3 and 4
tools/
  clean_info_json.py             normalises feature description files written by older runs of 02B
  run_robustness_experiments.py  runs notebook 03 without class weights, with a linear classifier and with
                                 other initialisation seeds
  download_nga.py                downloads the open-access images of the external test set
  run_external_validation.py     extracts the features of the external images and applies classifiers
                                 trained on all WikiArt paintings to them
data/external_nga/
  nga_paintings.csv              the 1,223 National Gallery of Art paintings of the external test set
data/metadata/
  selected_movements.json        the 20 movements
  image_list.csv.gz              the 65,998 paintings (file name, movement, artist, WikiArt style list)
splits/
  splits_<split mode>_<level>.csv.gz   role of every painting (train / val / test) in each of the 20 splits
results/
  cv/                            one workbook per experiment: Folds, Summary, CM_* (pooled confusion matrices)
  analysis/                      outputs of the analysis scripts
  figures/                       Figures 3 and 4
```

The WikiArt images and the National Gallery of Art images are not redistributed. The trained classifiers are not part of this repository.

## Data

The images and the metadata files `classes.csv` and `wclasses.csv` come from the Kaggle copy of WikiArt
(`steubk/wikiart`). Place them under `data/wikiart/`, with one folder per style. `data/metadata/image_list.csv.gz`
lists the paintings used in the paper. Notebook 01 regenerates it, together with the metadata table that the
other notebooks read.

## Environments

| Step | Environment | File |
|---|---|---|
| 01, 02, 03, analysis | Python 3.13, TensorFlow 2.20 (CPU is sufficient), scikit-learn 1.7 | `requirements-tf.txt` |
| 02B | Python 3.11, PyTorch 2.14 (CUDA 12.6), timm 1.0.30, open_clip 3.3.0 | `requirements-gpu.txt` |

The notebooks take the repository folder as the project root. They are expected to run from `notebooks/`.
Set the environment variable `ARTEMIS_ROOT` if the data are stored elsewhere.

## Reproducing the experiments

1. `01_dataset_preparation` builds the metadata table and selects the 20 movements.
2. Feature extraction can be done in either of two ways:
   - run `02_resnet50_features` (ResNet50) and `02B_gpu_features` once for each configuration (`MODEL_NAME`, `VIEWS`, `CROP_MODE`);
   - or download the extracted features from Zenodo ([10.5281/zenodo.23710694](https://doi.org/10.5281/zenodo.23710694)) into `cache/embeddings/` and `cache/embeddings/gpu/`.
3. `03_classification` is run once for each row of the table below. It writes
   `results/cv/classification_cv_results_full_<SPLIT_MODE>_<TAG>.xlsx`.
4. `python analysis/paired_tests.py`, `python analysis/per_class_metrics.py`, `python analysis/robustness_analyses.py`,
   `python analysis/figure3.py` and `python analysis/figure4.py` are run from the repository root.

| Experiment (paper) | `FEATURE_SOURCE` | `FEATURE_VIEW` | Other settings |
|---|---|---|---|
| Backbone comparison (Tables 3 and 6) | `resnet50_nb02`, `wikiart_<model>_full` for the nine models of 02B | `full` | `LEVEL=20` |
| Movement merging (Tables 4–5) | ResNet50, CLIP ViT-L/14, DINOv2, SigLIP2 (CLIP ViT-L/14 also with crops) | `full`, `full+crops_mean`, `all_expand` | `LEVEL=16` or `12` |
| Multi-view inference (Table 7) | `wikiart_clip_vitl14_fullcrops6_f0.5`, `wikiart_clip_vitl14_fullcrops4_quad` | `full+crops_mean`, `all_expand` | |
| Feature concatenation (Table 7) | `["wikiart_clip_vitl14_fullcrops6_f0.5", "wikiart_dinov2_vitl14_full"]` | `full` | |
| Artist leakage (Table 8) | `resnet50_nb02`, `wikiart_clip_vitl14_full` | `full` | `SPLIT_MODE="random"` |
| Per-artist cap (Table 9) | `wikiart_siglip2_so400m_full` | `full` | `MAX_PER_ARTIST=150` or `SINGLE_STYLE_ONLY=True` |

### Robustness experiments (Table 11)

`python tools/run_robustness_experiments.py` runs notebook 03 again on the same 20 splits with three changes to
the classifier. The settings are passed through environment variables; with the default settings the notebook
reproduces the main experiments.

| Group | Setting | Configurations | Suffix of the workbook |
|---|---|---|---|
| A | no class weights (`ARTEMIS_BALANCE=none`) | all ten | `_nocw` |
| B | linear classifier, no hidden layer (`ARTEMIS_HEAD_HIDDEN=0`) | all ten | `_linear` |
| C | initialisation seeds 43–46 (`ARTEMIS_INIT_SEED`), 10-fold scheme | SigLIP2 So400m/16 (384) | `_init43` … `_init46` |

`python analysis/robustness_analyses.py` then writes `results/analysis/L20_robustness.xlsx`. The workbook also
holds the confidence intervals of the paired differences, the leave-one-fold-out repetition of the main-model
selection and the analyses of the pooled confusion matrices that are reported in the paper.

### External validation (Table 12)

The classifiers are trained on all WikiArt paintings and applied, without adaptation, to 1,223 open-access
paintings of the National Gallery of Art (NGA), Washington, DC, in nine of the 20 movements. The list
`data/external_nga/nga_paintings.csv` was built from the museum's open data
(https://github.com/NationalGalleryOfArt/opendata, CC0): works classified as paintings, with an open-access
image and exactly one style term that corresponds to one of the 20 movements.

1. `python tools/download_nga.py` downloads the images (long side at most 2048 px) and writes the metadata file.
2. `python tools/run_external_validation.py features` extracts the features with the nine PyTorch
   configurations (notebook 02B with `ARTEMIS_DATASET=nga`); `--pilot` first runs a short test that also
   compares newly extracted WikiArt features with the stored ones.
3. `python tools/run_external_validation.py classify` runs notebook 03 with `ARTEMIS_EXTERNAL_TEST=nga` for the
   ten configurations: five training runs each, with different inner validation sets and initialisation seeds.
   The workbooks `results/cv/external_nga_full_L20*_external-nga.xlsx` hold the runs, the prediction of every
   painting in every run and a confusion matrix.
4. `python analysis/external_validation.py` writes `results/analysis/external_nga.xlsx`.

`<model>` is one of `clip_vitl14`, `clip_vith14`, `siglip2_so400m`, `siglip2_so400m_224`, `dinov2_vitl14`,
`dinov3_vitl16`, `convnext_l_in22k`, `convnext_l_clip`, `convnext_l_clip_320`.

### Extracted features (Zenodo, [10.5281/zenodo.23710694](https://doi.org/10.5281/zenodo.23710694))

For every configuration there are three files: `<tag>.npy` (float16, L2-normalised, shape images × views ×
dimension), `<tag>_index.parquet` (file name, row, `load_ok`) and `<tag>_info.json` (extractor specification).

- ResNet50 (`cache/embeddings/`): `embeddings.npy` (float32, not normalised), `embedding_index.parquet`,
  `embedding_info.json`
- Full image (`cache/embeddings/gpu/`): `wikiart_<model>_full` for the nine models listed below
- Multi-view, CLIP ViT-L/14: `wikiart_clip_vitl14_fullcrops6_f0.5` (full image + 6 square crops) and
  `wikiart_clip_vitl14_fullcrops4_quad` (full image + 4 quadrants)

### Splits

The splits are generated deterministically with `StratifiedGroupKFold` (artists as groups). The 10-fold scheme
uses seed 42. The SSS scheme takes the first fold of a 10-fold split for each of seeds 42–51. The inner validation
set used for early stopping holds 10% of the training artists (`GroupShuffleSplit`, seed 42 + 1000·i). The files
in `splits/` follow the logic of the "Export the splits" cell of notebook 03. The training and test sizes of all
20 splits agree with those recorded in every results workbook of the same level. They can be used to check a
re-implementation without running the notebooks; `pandas.read_csv` reads the compressed files directly.

**The scikit-learn version matters.** The splits of the paper were generated with scikit-learn 1.7.2. Later
versions of `StratifiedGroupKFold` produce different artist-disjoint splits from the same seeds (for example
with 1.9.1). Use the pinned version in `requirements-tf.txt`, or read the splits from `splits/`.

## Main results (20 movements, artist-disjoint 10-fold cross-validation, mean of 10 folds)

| Feature extractor | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---|---|
| ResNet50 (ImageNet-1k) | 33.1 | 34.5 | 28.1 |
| DINOv2 ViT-L/14 | 41.7 | 43.7 | 36.1 |
| ConvNeXt-L (ImageNet-22k) | 42.7 | 44.5 | 36.4 |
| DINOv3 ViT-L/16 | 45.7 | 48.4 | 39.4 |
| CLIP ViT-L/14 | 48.4 | 51.0 | 42.2 |
| ConvNeXt-L (CLIP, 256) | 49.6 | 48.8 | 41.8 |
| ConvNeXt-L (CLIP, 320) | 50.1 | 49.5 | 42.4 |
| SigLIP2 So400m/14 (224) | 52.1 | 53.2 | 44.2 |
| OpenCLIP ViT-H/14 | 52.5 | 53.5 | 44.7 |
| SigLIP2 So400m/16 (384) | 53.4 | 53.3 | 45.6 |

The results of the SSS scheme, the confidence intervals and the paired tests are in `results/` and in the paper.

### Classifier configuration (accuracy / balanced accuracy, %, 10-fold)

| Feature extractor | Main classifier | No class weights | Linear classifier |
|---|---|---|---|
| ResNet50 (ImageNet-1k) | 33.1 / 34.5 | 40.9 / 32.4 | 35.6 / 35.4 |
| DINOv2 ViT-L/14 | 41.7 / 43.7 | 49.7 / 41.9 | 44.9 / 45.7 |
| ConvNeXt-L (ImageNet-22k) | 42.7 / 44.5 | 49.5 / 42.9 | 44.5 / 44.5 |
| DINOv3 ViT-L/16 | 45.7 / 48.4 | 54.2 / 45.9 | 49.4 / 48.8 |
| CLIP ViT-L/14 | 48.4 / 51.0 | 56.4 / 48.9 | 51.2 / 51.6 |
| ConvNeXt-L (CLIP, 256) | 49.6 / 48.8 | 56.1 / 47.4 | 50.1 / 49.8 |
| ConvNeXt-L (CLIP, 320) | 50.1 / 49.5 | 56.1 / 47.5 | 50.9 / 49.6 |
| SigLIP2 So400m/14 (224) | 52.1 / 53.2 | 58.1 / 50.3 | 52.9 / 52.7 |
| OpenCLIP ViT-H/14 | 52.5 / 53.5 | 58.0 / 50.5 | 53.0 / 52.3 |
| SigLIP2 So400m/16 (384) | 53.4 / 53.3 | 58.2 / 49.6 | 53.6 / 53.3 |

The main classifier is trained with inverse-frequency class weights and one hidden layer. With five
initialisation seeds, the 10-fold accuracy of SigLIP2 So400m/16 (384) ranges from 52.9% to 54.6%.

### External validation on NGA paintings (accuracy, %, mean of five training runs)

| Feature extractor | WikiArt, 10-fold | NGA (n = 1,223) | NGA, artists not in WikiArt (n = 693) |
|---|---|---|---|
| ResNet50 | 33.1 | 34.5 | 31.1 |
| DINOv2 ViT-L/14 | 41.7 | 37.7 | 33.0 |
| ConvNeXt-L (IN-22k) | 42.7 | 38.9 | 36.6 |
| DINOv3 ViT-L/16 | 45.7 | 44.6 | 40.7 |
| CLIP ViT-L/14 | 48.4 | 48.9 | 45.9 |
| ConvNeXt-L (CLIP, 256) | 49.6 | 47.5 | 43.5 |
| ConvNeXt-L (CLIP, 320) | 50.1 | 51.1 | 50.7 |
| SigLIP2 So400m/14 (224) | 52.1 | 50.8 | 47.2 |
| OpenCLIP ViT-H/14 | 52.5 | 47.2 | 41.7 |
| SigLIP2 So400m/16 (384) | 53.4 | 50.9 | 48.0 |

The external set covers nine movements; 98% of its paintings belong to five of them (Realism, Naive Art,
Impressionism, Romanticism and Post-Impressionism).

## Licence

Code: MIT (`LICENSE`). Split files, lists, features and result tables: CC BY 4.0 (`LICENSE-data.md`).
The WikiArt images remain subject to the terms of their original sources. The descriptive data in
`data/external_nga/nga_paintings.csv` come from the open data of the National Gallery of Art (CC0). The pretrained encoders are
subject to their own licences.

## Citation

See `CITATION.cff`. Please cite the paper and the Zenodo records: code, splits and results
(https://doi.org/10.5281/zenodo.23709917) and extracted features (https://doi.org/10.5281/zenodo.23710694).
