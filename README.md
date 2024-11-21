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

See: `config_sd.py` for the datasets used in the experiments.


## Generate the training data

First, cd into ViZDoomPPO/ and generate the venv from the `requirements.txt` file.

Then, run the following command to train an agent on vizdoom:
```
python train_ppo_parallel.py
```

Once the agent is trained, generate episodes and upload them as a HF dataset using:
```
python load_model_generate_dataset.py --episodes {number of episodes} --output parquet --upload --hf_repo {name of the repo}
```


## Train the diffusion model

Debug on a single sample
```
python train_text_to_image.py  \
    --dataset_name arnaudstiegler/vizdoom-episode  \
    --gradient_checkpointing  \
    --train_batch_size 12  \
    --learning_rate 5e-5  \
    --num_train_epochs 1500  \
    --validation_steps 250  \
    --dataloader_num_workers 18 \
    --max_train_samples 2 \
    --use_cfg \
    --report_to wandb
```

Full training
```
python train_text_to_image.py \
    --dataset_name arnaudstiegler/vizdoom-episode-large \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --train_batch_size 12 \
    --dataloader_num_workers 18 \
    --num_train_epochs 10 \
    --validation_steps 1000 \
    --use_cfg \
    --output_dir sd-model-finetuned \
    --push_to_hub \
    --lr_scheduler cosine \
    --report_to wandb
```

## Run distributed trainng

```
torchrun --nproc_per_node=1 train_text_to_image.py \
    --dataset_name arnaudstiegler/vizdoom-episode \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --train_batch_size 12 \
    --dataloader_num_workers 18 \
    --num_train_epochs 10 \
    --validation_steps 1000 \
    --use_cfg \
    --output_dir sd-model-finetuned \
    --lr_scheduler cosine \
    --push_to_hub \
    --report_to wandb
```


## Run inference (generating a single image)

```
python run_inference.py --model_folder arnaudstiegler/gameNgen-baseline-20ksteps
```

## Run autoregressive inference

```
python run_autoregressive.py --model_folder arnaudstiegler/gameNgen-baseline-20ksteps
```
