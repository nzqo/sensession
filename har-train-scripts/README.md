# HAR experiment scripts.

The scripts here are used to instrument the open-source models used in our paper to evaluate
receiver effects on deep learning-based Human Activity Recognition (HAR) sensing models.

We evaluate four models:

- [ARIL](https://doi.org/10.1109/ACCESS.2019.2923743): https://github.com/geekfeiw/ARIL
- [RGANet](https://doi.org/10.3390/s25030918): https://github.com/h334658994/RGANet
- [WiADN](https://doi.org/10.1109/JIOT.2024.3434738): https://github.com/jzhoujg/WiADN
- [Recurrent Conformer for WiFi HAR](https://doi.org/10.1109/JAS.2023.123291): https://github.com/infinite0522/Recurrent-ConFormer-for-WiFi-HAR

## Usage

We try to keep the model architecture and training details as close as possible to the above
reference implementations. To do so, we provide single-file modules meant to be dropped into
the reference implementation codebase root.

Changes are only introduced where required to run the models on our dataset and are minimal.
All model APIs support specify dynamic input data formats, so directly support our dataset.
Only WiADN requires a patch for the mask module to handle data of size 700.

So, to use the above scripts, simply drop them into the repo root of the respective model,
adjust the data path, and run.

## Experiments

We run two basic types of experiments:

- kfold: We use repeated stratified kfold training each classifier on data from one receiver to estimate
    its performance. The goal of this experiment is to see how well a receiver's data supports a HAR use-case
    over the others. In other words, we care about **statistically significant differences** in accuracy.
- cross/ablation: We train the classifier on data from one receiver (using a specified standardization) and
    test on data from another. The goal of this experiment is to uncover cross-receiver data incompatibilities,
    which we attribute to stem mostly from AGC in our paper.


### Deviations from the original ARIL implementation

- **Single-task objective.** We optimize only the activity head; the location
    loss is dropped even though the branch remains in the model.
- **Dummy location head.** `location_num=1` and the location classifier is
    instantiated but never used in the loss or evaluation.
- **Checkpoint selection.** We keep the checkpoint with the highest *training* activity accuracy.


### Deviations from the original WiADN implementation

- **Single-task HAR only.** We train WiADN using only activity labels (location head is unused) and apply
    `AutomaticWeightedLoss` to this single activity loss, instead of the original dual-task (activity + location)
    setup with two coupled losses.

- **Sequence-length-agnostic attention.** We patch `MaskGenerator.forward` to crop tensors along the time dimension
    before concatenation so the model supports arbitrary sequence lengths (e.g., 700) instead of relying on the
    original fixed-length assumption.


### Deviations from the original RGANet implementation

- **Standardization.**
  We drop the original min-max standardization as default since we ablate over
  different types in our experiments.

- **Mini-batch handling.**  
  We use `drop_last=False`, so the final short batch is kept and contributes to gradient updates;
  the original UT-HAR training loader uses `drop_last=True`. We want to make use of all data.

- **Random seeding.**  
  We fix seeds to enforce deterministic runs, the original doesn't seed at all.


### Deviations from Recurrent Conformer for WiFi HAR

Shouldn't be any. Our dataset is supported there without modification.
