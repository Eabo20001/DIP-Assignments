# Assignment 2 - DIP with PyTorch

### This is Yibo Zhao's implementation of DIP assignment 2.

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Running

To run Poisson Image Editing, run:

```blend
cd 2.1_PoissonImageBlending
python run_blending_gradio.py
```

To run Pix2Pix training, run:

```learning
cd 2.2_Pix2Pix
python download_facades_dataset.py
python train.py
```

## Results 
### Poisson Image Editing

<img src = "./pics/result1.PNG" width = "80%">
<img src = "./pics/result2.PNG" width = "80%">


### Pix2Pix:
#### train_results

<img src = "./pics/train/result_1.png" width = "80%">
<img src = "./pics/train/result_2.png" width = "80%">
<img src = "./pics/train/result_3.png" width = "80%">
<img src = "./pics/train/result_4.png" width = "80%">
<img src = "./pics/train/result_5.png" width = "80%">

#### val_results
<img src = "./pics/val/result_1.png" width = "80%">
<img src = "./pics/val/result_2.png" width = "80%">
<img src = "./pics/val/result_3.png" width = "80%">
<img src = "./pics/val/result_4.png" width = "80%">
<img src = "./pics/val/result_5.png" width = "80%">

