from pathlib import Path

import click
import hydra
import numpy as np
import pyrootutils
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
import time
import math

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model_vqgan(config_name, checkpoint_path, device="cuda"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path, map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model: {result}")
    return model


@torch.no_grad()
def vqgan_inference(indices: torch.Tensor, endpoint: int, model, device="cuda"):

    print(f"decoding start")
    start = time.time()
    feature_lengths = torch.tensor([indices.shape[1]], device=device)
    fake_audios, _ = model.decode(
        indices=indices[None], feature_lengths=feature_lengths
    )
    audio_time = fake_audios.shape[-1] / model.spec_transform.sample_rate
    end = time.time()
    print(f"decoding end: time taken: {end - start}")

    print(f"Indices shape of: {indices.shape}")
    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.10f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    startpoint = math.ceil(endpoint / 10) * 10 - 10
    print(startpoint, endpoint)

    output_path = f"./temp/cum-{endpoint}.wav"
    print(f"saving cum to {output_path}")
    sf.write(output_path, fake_audio, model.spec_transform.sample_rate)
    logger.info(f"Saved audio to {output_path}")

    output_path = f"./temp/new-{endpoint}.wav"
    print(f"saving new portion to {output_path}")
    sf.write(output_path, fake_audio[startpoint*2048:], model.spec_transform.sample_rate)
    logger.info(f"Saved audio to {output_path}")


    # going to check to see if the without context the codebook generation worsens
    indices = indices[:, startpoint:]
    print(indices.shape)
    feature_lengths = torch.tensor([indices.shape[1]], device=device)
    fake_audios, _ = model.decode(
        indices=indices[None], feature_lengths=feature_lengths
    )
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    output_path = f"./temp/ONLYnew-{endpoint}.wav"
    print(f"saving ONLYnew portion to {output_path}")
    sf.write(output_path, fake_audio, model.spec_transform.sample_rate)
    logger.info(f"Saved audio to {output_path}")




if __name__ == "__main__":
    main_vqgan()
