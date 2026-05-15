# How to Train on Google Colab

Since we finished preprocessing locally, we have thousands of images in `data/train`, `data/val`, and `data/test`.
Follow these steps to train your models using a free GPU on Colab:

## 1. Zip your data
Run this command in your project terminal:
```powershell
Compress-Archive -Path data -DestinationPath data.zip
```

## 2. Upload to Colab
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Create a new Notebook.
3. In the left sidebar, click the "Files" icon and upload `data.zip`.
4. Upload `src/train.py`, `src/train_baselines.py`, and `src/utils.py`.

## 3. Run these cells in Colab

### Cell 1: Setup
```python
!unzip data.zip
!pip install tensorflow==2.16.1 opencv-python pandas matplotlib seaborn scikit-learn
```

### Cell 2: Train Main Model (ResNet-50)
```python
!python src/train.py
```

### Cell 3: Train Baselines (VGG, MobileNet, Custom)
```python
!python src/train_baselines.py
```

## 4. Download results
Once finished, you will see new files in the `models/` and `results/` folders in Colab.
Download all `.h5` files and the `results/` folder back to your local computer.
- Put `.h5` files in `models/`
- Put the logs and figures in `results/`

## 5. Run local evaluation
Once you have the models locally, run:
```powershell
python src/evaluate.py
```
This will generate the final confusion matrices and metrics summary for your thesis.
