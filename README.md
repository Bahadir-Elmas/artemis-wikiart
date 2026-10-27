# Artist-disjoint evaluation of frozen visual representations for art movement classification

Code, data splits and results for the paper

> B. Elmas, "Artist-Disjoint Evaluation of Frozen Visual Representations for Art Movement Classification:
> Pretraining Type, Artist Leakage, and Variability," *IEEE Access*, under review.

Twenty painting movements of WikiArt (65,998 paintings, 936 artists) are classified from frozen features of
ten image encoders. A small classifier network is trained on the features. It is evaluated on splits in which no
artist appears in both the training and the test set: 10-fold cross-validation and 10 stratified repeated
splits (SSS), both grouped by artist.

Archived release and extracted features: **Zenodo, DOI [to be added]**.

## Contents

```
notebooks/
  01_dataset_preparation.ipynb   metadata table, the 20 movements, list of paintings
  02_resnet50_features.ipynb     ResNet50 features (TensorFlow/Keras)
  02B_gpu_features.ipynb         features of the nine PyTorch configurations (timm / OpenCLIP)
  03_classification.ipynb        classifier network, artist-disjoint splits, evaluation, split export
analysis/
  paired_tests.py                Nadeau–Bengio corrected t-tests, Wilcoxon, Holm (Table 5)
  per_class_metrics.py           per-movement recall / precision / F1 and confusions (Table 9)
  figure3.py, figure4.py         Figures 3 and 4
tools/
  clean_info_json.py             normalises feature description files written by older runs of 02B
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

The WikiArt images are not redistributed. The trained classifiers are not part of this repository.

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
   - or download the extracted features from Zenodo into `cache/embeddings/` and `cache/embeddings/gpu/`.
3. `03_classification` is run once for each row of the table below. It writes
   `results/cv/classification_cv_results_full_<SPLIT_MODE>_<TAG>.xlsx`.
4. `python analysis/paired_tests.py`, `python analysis/per_class_metrics.py`, `python analysis/figure3.py`
   and `python analysis/figure4.py` are run from the repository root.

| Experiment (paper) | `FEATURE_SOURCE` | `FEATURE_VIEW` | Other settings |
|---|---|---|---|
| Backbone comparison (Table 5) | `resnet50_nb02`, `wikiart_<model>_full` for the nine models of 02B | `full` | `LEVEL=20` |
| Movement merging (Tables 3–4) | ResNet50, CLIP ViT-L/14, DINOv2, SigLIP2 (CLIP ViT-L/14 also with crops) | `full`, `full+crops_mean`, `all_expand` | `LEVEL=16` or `12` |
| Multi-view inference (Table 6) | `wikiart_clip_vitl14_fullcrops6_f0.5`, `wikiart_clip_vitl14_fullcrops4_quad` | `full+crops_mean`, `all_expand` | |
| Feature concatenation (Table 6) | `["wikiart_clip_vitl14_fullcrops6_f0.5", "wikiart_dinov2_vitl14_full"]` | `full` | |
| Artist leakage (Table 7) | `resnet50_nb02`, `wikiart_clip_vitl14_full` | `full` | `SPLIT_MODE="random"` |
| Per-artist cap (Table 8) | `wikiart_siglip2_so400m_full` | `full` | `MAX_PER_ARTIST=150` or `SINGLE_STYLE_ONLY=True` |

`<model>` is one of `clip_vitl14`, `clip_vith14`, `siglip2_so400m`, `siglip2_so400m_224`, `dinov2_vitl14`,
`dinov3_vitl16`, `convnext_l_in22k`, `convnext_l_clip`, `convnext_l_clip_320`.

### Extracted features (Zenodo)

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

## Licence

Code: MIT (`LICENSE`). Split files, lists, features and result tables: CC BY 4.0 (`LICENSE-data.md`).
The WikiArt images remain subject to the terms of their original sources. The pretrained encoders are
subject to their own licences.

## Citation

See `CITATION.cff`. Please cite the paper and the Zenodo record.
