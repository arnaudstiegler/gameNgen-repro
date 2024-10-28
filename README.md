# gameNgen-repro

# TODO List
- [x] Add the conditioning image noise
- [x] Add the VAE decoder training script
- [] Put the buffer size as a parameter to the train script
- [] Clean up the inference script
- [] Try reducing the number of timesteps for inference
- [] Implement latent caching for the autoregressive inference
- [] Do some quick parameter sweep on LR, scheduler, etc.
- [x] Add the target image to the validation logging
- [x] Try some speed optimizations


## Datasets

Test dataset: `arnaudstiegler/gameNgen_test_dataset`
Training dataset (small): `P-H-B-D-a16z/ViZDoom-Deathmatch-PPO`
Training dataset (large): `P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-Lrg`


## Run training on gameNgen

Debug on a single sample
```
python train_text_to_image.py  \
    --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-Lrg  \
    --gradient_checkpointing  \
    --train_batch_size 12  \
    --learning_rate 5e-5  \
    --num_train_epochs 1500  \
    --validation_steps 250  \
    --dataloader_num_workers 18 \
    --max_train_samples 1 \
    --use_cfg \
    --report_to wandb
```

Full training
```
python train_text_to_image.py \
    --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-Lrg \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --train_batch_size 12 \
    --dataloader_num_workers 18 \
    --num_train_epochs 10 \
    --validation_steps 250 \
    --use_cfg \
    --output_dir sd-model-finetuned \
    --push_to_hub \
    --report_to wandb
```


## Run training for original sd model
```
python compare_train.py --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --gradient_checkpointing --train_batch_size 8 --learning_rate 5e-5 --num_train_epochs 1500 --validation_epochs 250 --validation_prompt "doom image, high quality, 4k, high resolution" --report_to wandb
```
