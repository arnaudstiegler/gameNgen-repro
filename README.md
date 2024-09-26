# gameNgen-repro

## Datasets

`arnaudstiegler/gameNgen_test_dataset`
`P-H-B-D-a16z/ViZDoom-Deathmatch-PPO`

## Run training
```
python train_text_to_image.py  --dataset_name x --num_train_epochs 1 --train_batch_size 1 --gradient_checkpointing --max_train_samples 1 --checkpointing_steps 2 --report_to wandb
```