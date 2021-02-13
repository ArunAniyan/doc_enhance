## Document Enhancement

This repo provides implementation of Zhang et. al 2017 method for image denoising and enhancement.

Before running the main code, perform the following steps.

 1. Clone this repo.
 2. Make python (>=3.6) virtual environment.
 3. pip install -r requirements.txt

The main code enhance.py can be run in single image mode as in the following example.

```python enhance.py -i test_images/test_2.png -m dncnn3```

To run the code in batch mode, follow the example below

```python enhance.py -b test_images-m dncnn3 ```

### Results  for test images
![Brisque Scores for test images with different models](https://github.com/ArunAniyan/doc_enhance/blob/main/table.png)


