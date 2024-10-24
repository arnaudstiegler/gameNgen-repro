# gameNgen-repro

# TODO List
- [] Put the buffer size as a parameter to the train script
- [] Clean up the inference script
- []


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
    --train_batch_size 8  \
    --learning_rate 5e-5  \
    --num_train_epochs 1500  \
    --validation_steps 250  \
    --max_train_samples 1
```

Full training
```
python train_text_to_image.py  \
    --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-Lrg  \
    --gradient_checkpointing  \
    --train_batch_size 10  \
    --learning_rate 5e-5  \
    --num_train_epochs 10  \
    --validation_steps 250  \
    --lr_scheduler cosine  \
    --use_cfg  \
    --output_dir sd-model-finetune  \
    --push_to_hub  \
    --report_to wandb
```


## Run training for original sd model
```
python compare_train.py --dataset_name P-H-B-D-a16z/ViZDoom-Deathmatch-PPO --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --gradient_checkpointing --train_batch_size 8 --learning_rate 5e-5 --num_train_epochs 1500 --validation_epochs 250 --validation_prompt "doom image, high quality, 4k, high resolution" --report_to wandb
```
