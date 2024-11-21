#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo
from packaging import version
from safetensors.torch import load_file
from tqdm.auto import tqdm

import wandb
from config_sd import (
    BUFFER_SIZE,
    CFG_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    REPO_NAME,
    TRAINING_DATASET_DICT,
    VALIDATION_PROMPT,
    ZERO_OUT_ACTION_CONDITIONING_PROB,
)
from dataset import EpisodeDataset, get_dataloader
from model import get_model, save_and_maybe_upload_to_hub
from run_inference import run_inference_img_conditioning_with_params
from utils import add_conditioning_noise, get_conditioning_noise

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

logger = get_logger(__name__, log_level="INFO")

torch.set_float32_matmul_precision("high")


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    """
    Here, we wanna validate different actions based on the same sample (doesn't really matter for now testing different samples)
    """
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            images.append(
                pipeline(
                    args.validation_prompt, num_inference_steps=30, generator=generator
                ).images[0]
            )

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}")
                        for i, image in enumerate(images)
                    ]
                }
            )
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="arnaudstiegler/game-n-gen-finetuned-sd",
        help="The name of the model to use as a base model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=TRAINING_DATASET_DICT["small"],
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=VALIDATION_PROMPT,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run fine-tuning validation every 50 steps. The validation process consists of running the model on a batch of images"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--skip_action_conditioning",
        action="store_true",
        help="Whether or not to use action conditioning.",
    )
    parser.add_argument(
        "--skip_image_conditioning",
        action="store_true",
        help="Whether or not to use action conditioning.",
    )
    parser.add_argument(
        "--use_cfg",
        action="store_true",
        help="Whether or not to use classifier free guidance.",
    )
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default=None,
        help="Path to a directory containing a previously trained model to load.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.report_to == "wandb":
        import wandb

        run = wandb.init()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=REPO_NAME, exist_ok=True, token=args.hub_token
            ).repo_id

    # This is a bit wasteful
    dataset = EpisodeDataset(args.dataset_name)
    action_dim = dataset.get_action_dim()

    unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = get_model(
        action_dim, skip_image_conditioning=args.skip_image_conditioning
    )

    if args.load_pretrained:
        logger.info(f"Loading pretrained model from {args.load_pretrained}")
        unet.load_state_dict(
            load_file(
                os.path.join(
                    args.load_pretrained, "unet", "diffusion_pytorch_model.safetensors"
                )
            )
        )
        vae.load_state_dict(
            load_file(
                os.path.join(
                    args.load_pretrained, "vae", "diffusion_pytorch_model.safetensors"
                )
            )
        )

        # Load scheduler configuration
        with open(
            os.path.join(args.load_pretrained, "scheduler", "scheduler_config.json"),
            "r",
        ) as f:
            scheduler_config = json.load(f)
        noise_scheduler = DDIMScheduler.from_config(scheduler_config)

        action_embedding.load_state_dict(
            torch.load(os.path.join(args.load_pretrained, "action_embedding.pth"))
        )

        # Load embedding info
        embedding_info = torch.load(
            os.path.join(args.load_pretrained, "embedding_info.pth")
        )
        action_embedding.num_embeddings = embedding_info["num_embeddings"]
        action_embedding.embedding_dim = embedding_info["embedding_dim"]

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    action_embedding.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.skip_action_conditioning:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            list(unet.parameters()) + list(action_embedding.parameters()),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    train_dataloader = get_dataloader(
        dataset_name=args.dataset_name,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * num_update_steps_per_epoch
            * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = (
            args.max_train_steps * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if (
            num_training_steps_for_scheduler
            != args.max_train_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Dataset = {args.dataset_name}")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        f"  Zero out action conditioning probability = {ZERO_OUT_ACTION_CONDITIONING_PROB}"
    )
    logger.info(f"  Skip action conditioning = {args.skip_action_conditioning}")
    logger.info(f"  Skip image conditioning = {args.skip_image_conditioning}")
    logger.info(f"  Use CFG = {args.use_cfg}")
    logger.info(f"  Prediction type = {args.prediction_type}")
    logger.info(f"  SNR gamma = {args.snr_gamma}")
    logger.info(f"  Max grad norm = {args.max_grad_norm}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  Adam beta 1 = {args.adam_beta1}")
    logger.info(f"  Adam beta 2 = {args.adam_beta2}")
    logger.info(f"  Adam weight decay = {args.adam_weight_decay}")
    logger.info(f"  Adam epsilon = {args.adam_epsilon}")
    logger.info(f"  Lr scheduler = {args.lr_scheduler}")
    logger.info(f"  Lr warmup steps = {args.lr_warmup_steps}")
    logger.info(f"  Report to = {args.report_to}")
    logger.info(f"  Output dir = {args.output_dir}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.skip_image_conditioning:
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    # Just to keep the same var as with the image conditioning
                    concatenated_latents = noisy_latents
                else:
                    bs, buffer_len, channels, height, width = batch[
                        "pixel_values"
                    ].shape

                    # Fold buffer len in to batch for encoding in one go
                    folded_conditional_images = batch["pixel_values"].view(
                        bs * buffer_len, channels, height, width
                    )

                    latents = vae.encode(
                        folded_conditional_images.to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    _, latent_channels, latent_height, latent_width = latents.shape
                    # Separate back the conditioning frames
                    latents = latents.view(
                        bs, buffer_len, latent_channels, latent_height, latent_width
                    )

                    # Generate noise with the same shape as latents
                    # Careful with the indexing here
                    noise = torch.randn_like(latents[:, -1:, :, :, :])

                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise[:, -1, :, :, :] = args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[2], 1, 1),
                            device=latents.device,
                        )
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bs,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # Note that the noise is only added to the target frame (last frame)
                    noisy_latents = latents.clone()
                    noisy_latents[:, -1:, :, :, :] = noise_scheduler.add_noise(
                        latents[:, -1:, :, :, :], noise, timesteps
                    )

                    # Generate noise for the conditioning frames with a corresponding discrete noise level
                    noise_level, discretized_noise_level = get_conditioning_noise(
                        latents[:, :-1, :, :, :]
                    )
                    # Add noise to the conditioning frames only
                    noisy_latents[:, :-1, :, :, :] = add_conditioning_noise(
                        latents[:, :-1, :, :, :], noise_level
                    )

                    # We collapse the frame conditioning into the channel dimension
                    concatenated_latents = noisy_latents.view(
                        bs, buffer_len * latent_channels, latent_height, latent_width
                    )

                # Get the text embedding for conditioning
                if args.skip_action_conditioning:
                    encoder_hidden_states = text_encoder(
                        batch["input_ids"], return_dict=False
                    )[0]
                else:
                    encoder_hidden_states = action_embedding(batch["input_ids"])

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                # Predict the noise residual and compute loss
                model_pred = unet(
                    concatenated_latents,
                    timesteps,
                    class_labels=discretized_noise_level,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

                if not args.skip_image_conditioning:
                    # Only predict the last frame – empirically verified that it is frame 0 and not frame -1.
                    target_last_frame = target[:, -1, :, :, :]
                else:
                    target_last_frame = target

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target_last_frame.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Log the loss
                if accelerator.is_main_process and args.report_to == "wandb":
                    run.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[
                                0
                            ],  # Add this line
                        },
                        step=global_step,
                    )

                # Gather the losses across all pro`cesses for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = trainable_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],  # Add this line
                    },
                    step=global_step,
                )
                train_loss = 0.0

                validation_images = []
                context_images = []  # To store context images
                target_images = []
                if global_step % args.validation_steps == 0:
                    accelerator.print("Generating validation image")
                    unet.eval()
                    if accelerator.is_main_process:
                        save_and_maybe_upload_to_hub(
                            repo_id=REPO_NAME,
                            output_dir=args.output_dir,
                            unet=unet,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            action_embedding=action_embedding,
                            should_upload_to_hub=args.push_to_hub,
                            images=validation_images,
                            dataset_name=args.dataset_name,
                        )

                        # Use the current batch for inference
                        # Generate 2 images
                        for i in range(2):
                            single_sample_batch = {
                                "pixel_values": batch["pixel_values"][i].unsqueeze(0),
                                "input_ids": batch["input_ids"][i].unsqueeze(0),
                            }
                            with torch.no_grad():
                                if args.skip_image_conditioning:
                                    raise NotImplementedError("Not supported anymore")
                                else:
                                    generated_image = run_inference_img_conditioning_with_params(
                                        unet=accelerator.unwrap_model(unet),
                                        vae=vae,
                                        noise_scheduler=noise_scheduler,
                                        action_embedding=action_embedding,
                                        tokenizer=tokenizer,
                                        text_encoder=text_encoder,
                                        batch=single_sample_batch,
                                        device=accelerator.device,
                                        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                                        do_classifier_free_guidance=args.use_cfg,
                                        guidance_scale=CFG_GUIDANCE_SCALE,
                                        skip_action_conditioning=args.skip_action_conditioning,
                                    )
                                validation_images.append(generated_image)

                                # Extract and store context images
                                context_images.append(
                                    single_sample_batch["pixel_values"][0][:BUFFER_SIZE]
                                )
                                target_images.append(
                                    single_sample_batch["pixel_values"][0][-1]
                                )

                        if args.report_to == "wandb":
                            wandb.log(
                                {
                                    # The int index is used to order the images in the wandb dashboard
                                    "1_validation_images": [
                                        wandb.Image(img, caption=f"Generated Image {i}")
                                        for i, img in enumerate(validation_images)
                                    ],
                                    "2_target_images": [
                                        wandb.Image(
                                            target_img, caption=f"Target Image {i}"
                                        )
                                        for i, target_img in enumerate(target_images)
                                    ],
                                    "3_context_images": [
                                        wandb.Image(
                                            context_img, caption=f"Context Image {i}"
                                        )
                                        for i, context_img in enumerate(context_images)
                                    ],
                                },
                                step=global_step,
                            )
                        unet.train()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the model
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_and_maybe_upload_to_hub(
            repo_id=REPO_NAME,
            output_dir=args.output_dir,
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            action_embedding=action_embedding,
            should_upload_to_hub=args.push_to_hub,
            images=validation_images,
            dataset_name=args.dataset_name,
        )

    accelerator.end_training()

    # At the end of your script
    if accelerator.is_main_process:
        if args.report_to == "wandb":
            wandb.finish()


if __name__ == "__main__":
    main()
