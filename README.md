# gameNgen-repro

## Datasets

`arnaudstiegler/gameNgen_test_dataset`
`P-H-B-D-a16z/ViZDoom-Deathmatch-PPO`

## Run training
```
python train_text_to_image.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --dataset_name  --num_train_epochs 1 --train_batch_size 1 --gradient_checkpointing --max_train_samples 3 --push_to_hub --checkpointing_steps 2
```