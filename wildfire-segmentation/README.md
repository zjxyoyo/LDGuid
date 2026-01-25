# wildfire-segmentation

Here is [Sample Jupyter Notebook](train.ipynb)

## Step 1: Install Dependencies

> Notes: Make sure you installed correct pytorch version that match your CUDA driver version

```bash
pip install -e . --upgrade
```
## Step 2: Download Dataset

```bash
export PYTHONPATH=./

python data/load_data.py
```

## Step 3: Train the Model

You can change the default configurations and parameters in `src/train/train.py`

```bash
python src/train/train.py
```

## Step 4: Visualization

You need install and enable [git lfs](https://git-lfs.com/) to download pre-trained model

```bash
python visualize_sample.py
```

