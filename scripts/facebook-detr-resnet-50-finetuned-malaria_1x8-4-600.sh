#!/bin/bash

# Configuración de variables
MODEL_NAME="facebook/detr-resnet-50"
DATASET_NAME="SemilleroCV/lacuna_malaria"
OUTPUT_DIR="facebook-detr-resnet-50-finetuned-malaria"
NUM_TRAIN_EPOCHS=4
IMAGE_SQUARE_SIZE=600
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
CHECKPOINTING_STEPS="epoch"
LEARNING_RATE=5e-5
IGNORE_MISMATCHED_SIZES="--ignore_mismatched_sizes"
WITH_TRACKING="--with_tracking"
PUSH_TO_HUB="--push_to_hub"
REPORT_TO="wandb"
WANDB_PROJECT="challenge-malaria"
SEED=3407
# CHECKPOINT="detr-resnet-50-finetuned/epoch_1"
MODEL_ID="SemilleroCV/${OUTPUT_DIR}"

# Generación del nuevo nombre del archivo basado en la configuración
NEW_SCRIPT_NAME="${OUTPUT_DIR}_1x${TRAIN_BATCH_SIZE}-${NUM_TRAIN_EPOCHS}-${IMAGE_SQUARE_SIZE}.sh"

# Llamada al script de entrenamiento
accelerate launch run_object_detection_no_trainer.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --image_square_size $IMAGE_SQUARE_SIZE \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    --learning_rate $LEARNING_RATE \
    $IGNORE_MISMATCHED_SIZES \
    $WITH_TRACKING \
    --report_to $REPORT_TO \
    --wandb_project "$WANDB_PROJECT" \
    --seed $SEED \
    # $PUSH_TO_HUB \
    # --hub_model_id "$MODEL_ID"
    # --resume_from_checkpoint "$CHECKPOINT" \

# Renombrar el archivo script
mv "$0" "scripts/$NEW_SCRIPT_NAME"