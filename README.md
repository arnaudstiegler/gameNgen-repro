# gameNgen-repro

## Run training
```
python -m sd3.train_text_to_image --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --dataset_name arnaudstiegler/gameNgen_test_dataset --num_train_epochs 1 --train_batch_size 1 --gradient_checkpointing --max_train_samples 3 --push_to_hub --checkpointing_steps 2
```