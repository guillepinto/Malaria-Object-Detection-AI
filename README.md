<div align="center">
  
# Malaria Parasite Detection and Classification Using Blood Slide Images

<a href="#">
  <img src="https://pbs.twimg.com/media/GVVaDq7XEAA6sh_?format=jpg&name=medium" alt="challenge" style="max-width: 100%; height: 400px;">
</a>

[Guillermo Pinto](https://github.com/guillepinto), [Miguel Pimiento](https://github.com/pimientoyolo125), [Juan Diego Roa]()

Research Group: [Hands-on Computer Vision](https://github.com/semilleroCV)

> **Overview:** This project aims to detect and classify malaria parasites in blood slide images. Using advanced machine learning and deep learning techniques, we focus on detecting the parasite at the trophozoite stage and classifying infected and uninfected cells. The project is part of a [global challenge](https://zindi.africa/competitions/lacuna-malaria-detection-challenge) to improve medical diagnostics in resource-limited areas, with a special focus on Africa. This challenge is presented as an academic project for the subject Artificial Intelligence II, with the goal of participating and winning in this challenge.

</div> 

</br>

<p align="center"> <img src="" alt="pipeline" height='500'> </p>

## Dataset

The dataset used for this project can be found at the following link: [Malaria Blood Smear Dataset](https://drive.google.com/file/d/16T40TdpaB8VXohm50SySREwrzbuPcJBC/view). It includes images of blood slides with infected and uninfected cells.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
│
├── models             <- Trained and serialized models, model predictions, or model summaries    
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── scripts            <- Automates model training with preset configurations and batch scripts
│
├── make_submission.py <- Code to run model inference with trained models and make the submission file 
│
├── run_object_detection_no_trainer.py <- Code adapted from https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection
│                                       to train any model available at hugging face
│
├── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└──  src                <- Source code for use in this project.
    ├── config.py               <- Store useful variables and configuration
    ├── dataset.py              <- Scripts to download or generate data
    ├── __init__.py             <- Makes challenge_malaria a Python module
    └── utils.py                <- Utilities scripts
```

--------

## Prerequisites

1. **Create and activate the environment**:
   ```bash
    git clone https://github.com/semilleroCV/challenge-malaria
    cd challenge-malaria

    # if you have conda, run and activate the environment
    make create_environment

    # Install necessary dependencies using
    make requirements
    ```

2. **Log into Hugging Face and Weights & Biases (W&B)**:
   - Login to Hugging Face via the CLI:
     ```bash
     huggingface-cli login
     ```
   - Login to Weights & Biases:
     ```bash
     wandb login
     ```

3. **Using the dataset**:
   - When using the dataset from the organization, make sure to pass the full name of the dataset to the `--dataset_name` argument, e.g., `SemilleroCV/lacuna_malaria`.
   
## Config name style

```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_
```

- {training settings}: Information of training settings such as batch size, augmentations, loss, learning rate scheduler, and epochs/iterations. For example: `4xb4-ce-linearlr-40K` means using 4-gpus x 4-images-per-gpu, CrossEntropy loss, Linear learning rate scheduler, and train 40K iterations. Some abbreviations:

    {gpu x batch_per_gpu}: GPUs and samples per GPU. bN indicates N batch size per GPU. E.g. 8xb2 is the short term of 8-gpus x 2-images-per-gpu. And 4xb4 is used by default if not mentioned.

    {schedule}: training schedule, options are 20k, 40k, etc. 20k and 40k means 20000 iterations and 40000 iterations respectively.

- {training dataset information}: Training dataset names like cityscapes, ade20k, etc, and input resolutions. For example: `cityscapes-768x768` means training on cityscapes dataset and the input shape is 768x768. also the augmentations used in training.

full example: `deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024`

## Running `run_object_detection_no_trainer.py` from the Script:

To execute `run_object_detection_no_trainer.py` from a script located in the `scripts` folder while being in the root of the project, use the following command:

```bash
bash ./scripts/<your_script_name>.sh
```

Make sure your `.sh` script contains the correct parameters and paths relative to the root of the project. The script will automatically call `run_object_detection_no_trainer.py` with the necessary arguments for training.

## Important Notes:

- **Handling Negative Images (`NEG`)**:
   - The dataset is structured correctly. Negative images (those without objects) are represented by empty lists in all fields (`bbox`, `categories`, `id`, `area`), while images with objects contain the appropriate coordinates and categories.
   - For category mapping, only the actual classes `{0: 'Trophozoite', 1: 'WBC'}` are needed. The model will automatically learn when to predict objects and when not to during training.

- **Code Adjustments**:
   - **Line 136**: Changed the key for `objects` from `'category'` to `'categories'`.
   - **Line 477**: Manually passed the `categories` field.
   - **Continuing Training**: You can resume training from a previous checkpoint by using the `--resume_from_checkpoint` parameter. For example:
     ```bash
     --resume_from_checkpoint detr-resnet-50-finetuned/epoch_0
     ```
   - **Model Requirements**: You can use any object detection model from Hugging Face, as long as it is available in `.safetensors` format and has a properly configured `config.json`.

- **Mixed Precision Training**:
   - If you start training without mixed precision and later configure `accelerate` for mixed precision, it won't work because `accelerate` looks for a `scaler.pt` file. However, you can do the reverse (start with mixed precision and disable it later).

- **W&B Run Naming**:
   - To ensure W&B uses the same name as the `output_dir` for the run, add the following to line 621:
     ```python
     wandb.run.name = args.output_dir
     ```

- **Pushing the Model to Hugging Face**:
   - To upload the model privately to your organization, set the `--hub_model_id` parameter, where `MODEL_ID` can be:
     ```bash
     MODEL_ID="SemilleroCV/${OUTPUT_DIR}"
     ```
   - Also, on line 443, pass the `private=True` argument to `api.create_repo`.

## Making Inference and Submissions:

- To create the submission file (`submission.csv`), run `make_submission.py`. This script takes two parameters:
  - The name of the model on Hugging Face or the local directory where the model is stored.
  - The directory containing the test images.

Example usage:
```bash
python make_submission.py --model_name SemilleroCV/facebook-detr-resnet-50-finetuned-malaria --test_dir_path test
```
