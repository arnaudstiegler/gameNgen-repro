# gameNgen-repro

## Datasets

`arnaudstiegler/gameNgen_test_dataset`
`P-H-B-D-a16z/ViZDoom-Deathmatch-PPO`

## Run training on gameNgen
```
python train_text_to_image.py  --dataset_name x --num_train_epochs 1 --train_batch_size 1 --gradient_checkpointing --max_train_samples 1 --checkpointing_steps 2 --report_to wandb
```

Debug for now:
```
python train_text_to_image.py  --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO --gradient_checkpointing --train_batch_size 8 --learning_rate 5e-5 --num_train_epochs 1500 --validation_epochs 250 --validation_prompt "doom image, high quality, 4k, high resolution" --skip_image_conditioning --skip_action_conditioning
```


## Run training for original sd model
```
python compare_train.py --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --gradient_checkpointing --train_batch_size 8 --learning_rate 5e-5 --num_train_epochs 1500 --validation_epochs 250 --validation_prompt "doom image, high quality, 4k, high resolution" --report_to wandb
```

Note: changing the image resolution really degrades the generation quality.