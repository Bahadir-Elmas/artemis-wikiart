# Split files

`splits_<split mode>_<level>.csv.gz` (gzip-compressed CSV; `pandas.read_csv` reads it directly) has one row per painting (`filename`, `movement`, `artist`) and one column per
split: `kfold_01` … `kfold_10` (10-fold cross-validation, seed 42) and `sss_01` … `sss_10` (SSS repetitions,
seeds 42–51). Each cell gives the role of the painting in that split:

- `train`: used to fit the classifier network,
- `val`: inner validation set for early stopping (10% of the training artists),
- `test`: test fold.

`movement` is the class label at the given level (L20, L16 or L12). `random` files are the
artist-overlapping splits used for the leakage measurement (Table 7).
