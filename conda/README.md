Conda packaging for seasmon_xr
==============================

To install in conda

```bash
conda activate myenv
conda install -c https://data.earthobservation.vam.wfp.org/hdc seasmon-xr
```

Or when defining environment

```yaml
name: myenv
channels:
  - https://conda.anaconda.org/conda-forge
  - https://data.earthobservation.vam.wfp.org/hdc
dependencies:
  - python=3.10
  - seasmon-xr
```

Or add it to channel list

```bash
conda config --append channels https://data.earthobservation.vam.wfp.org/hdc
```
