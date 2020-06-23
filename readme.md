## Object Detection
This project lets you fine-tune Mask-RCNN on masks annotated using labelme, this allows you to train mask-rcnn on any categories you want to annotate! This project comes with several pretrained models trained on either custom datasets or on subsets of COCO.

## Getting Started
The pretrained models are stored in the repo with git-lfs, when you clone make sure you've pulled the files by calling, 

```bash
git lfs pull
```
 or by downloading them from github directly. This project uses conda to manage its enviroment; once conda is installed we create the enviroment and activate it, 
```bash
conda env create -f enviroment.yml
conda activate object_detection
```
. On windows powershell needs to be initialised and the execution policy needs to be modified. 
```bash
conda init powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Pre-Trained Projects
This project comes bundled with several pretrained models, which can be found in the `pretrained` directory. To infer objects and instance masks on your images run `evaluate_images`.
```bash
# to display the output
python evaluate_images.py --images ~/Pictures/ --model pretrained/model_mask_rcnn_skin_30.pth --display
# to save the output
python evaluate_images.py --images ~/Pictures/ --model pretrained/model_mask_rcnn_skin_30.pth --save
```

### Pizza Topping Segmentation
This was trained with a custom dataset of 89 images taken from COCO where pizza topping annotations were added. There's very few images for each type of topping so this model performs very badly and needs quite a few more images to behave well!

- 'chilli', 'ham', 'jalapenos', 'mozzarella', 'mushrooms', 'olive', 'pepperoni', 'pineapple', 'salad', 'tomato'

![Pizza Toppings](https://raw.githubusercontent.com/WillBrennan/ObjectDetection/master/pretrained/pizza_topping_example.jpg)

### Cat and Bird Detection
Annotated images of birds and cats were taken from COCO using the `extract_from_coco` script and then trained on. 

- cat, birds

![Demo on Cat & Birds](https://raw.githubusercontent.com/WillBrennan/ObjectDetection/master/pretrained/cat_examples.jpg)

### Cutlery Detection
Annotated images of knifes, forks, spoons and other cutlery were taken from COCO using the `extract_from_coco` script and then trained on. 

- knife, bowl, cup, bottle, wine glass, fork, spoon, dining table

![Demo on Cutlery](https://raw.githubusercontent.com/WillBrennan/ObjectDetection/master/pretrained/cutlery_example.jpg)

## Training New Projects
To train a new project you can either create new labelme annotations on your images, to launch labelme run, 

```bash
labelme
```
and start annotating your images! You'll need a couple of hundred. Alternatively if your category is already in COCO you can run the conversion tool to create labelme annotations from them. 

```bash
python extract_from_coco.py --images ~/datasets/coco/val2017 --annotations ~/datasets/coco/annotations/instances_val2017.json --output ~/datasets/my_cat_images_val --categories cat
```

Once you've got a directory of labelme annotations you can check how the images will be shown to the model during training by running, 

```bash
python check_dataset.py --dataset ~/datasets/my_cat_images_val
# to show our dataset with training augmentation
python check_dataset.py --dataset ~/datasets/my_cat_images_val --use-augmentation
```
. If your happy with the images and how they'll appear in training then train the model using, 

```bash
python train.py --train ~/datasets/my_cat_images_train --val ~/datasets/my_cat_images_val --model-tag mask_rcnn_cat
```
. This may take some time depending on how many images you have. Tensorboard logs are available in the `logs` directory. To run your trained model on a directory of images run

```bash
# to display the output
python evaluate_images.py --images ~/Pictures/my_cat_imgs --model models/model_mask_rcnn_cat_30.pth --display
# to save the output
python evaluate_images.py --images ~/Pictures/my_cat_imgs --model models/model_mask_rcnn_cat_30.pth --save
```
