# Data

The raw EuroSAT images are intentionally excluded from version control. The
dataset contains 27,000 RGB images and is available from the official EuroSAT
release (DOI: `10.5281/zenodo.7711097`) or through
`torchvision.datasets.EuroSAT(download=True)`.

The versioned files under `splits/` define the fixed stratified 70/15/15 split
used throughout this project. Do not regenerate these files when reproducing
the reported experiments.

- Random seed: `42`
- Split checksum: `f97c4ec9a27435a932662d5a8b707255`
- Training images: `18,900`
- Validation images: `4,050`
- Test images: `4,050`
