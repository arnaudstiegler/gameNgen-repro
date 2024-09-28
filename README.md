# gameNgen-repro

## Datasets

`arnaudstiegler/gameNgen_test_dataset`
`P-H-B-D-a16z/ViZDoom-Deathmatch-PPO`

## Run training on gameNgen
```
python train_text_to_image.py  --dataset_name x --num_train_epochs 1 --train_batch_size 1 --gradient_checkpointing --max_train_samples 1 --checkpointing_steps 2 --report_to wandb
```

## Run training for original sd model
```
python compare_train.py --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --gradient_checkpointing --mixed_precision bf16 --train_batch_size 32 --learning_rate 1e-4 --num_train_epochs 2 --report_to wandb
```