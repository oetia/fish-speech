import argparse
parser = argparse.ArgumentParser(description="Run something mediocre.")
parser.add_argument("--checkpoint-path", type=str)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--compile", dest="compile", action="store_true")
parser.add_argument("--no-compile", dest="compile", action="store_false")
parser.add_argument("--half", dest="compile", action="store_true")
parser.add_argument("--output-base-path", type=str, default="./output")
parser.set_defaults(half=False)
args = parser.parse_args()
print(args)
print("sanity check")


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.exceptions import HTTPException
from typing import Literal
from pydantic import BaseModel, Field
from io import BytesIO
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://127.0.0.1:7997"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)







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
from pydub import AudioSegment
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model_vqgan(config_name: str, checkpoint_path: str, device="cuda"):
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


logger.info("Loading VQGAN ...")
t0 = time.time()
model_vqgan = load_model_vqgan(
    "firefly_gan_vq", 
    os.path.join(args.checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"), 
    device=args.device
)
logger.info(f"Loaded VQGAN in {time.time() - t0:2f}")


@torch.no_grad()
def vqgan_inference(
    indices: torch.Tensor, # grouped finite scalar vector quantization encoded (8,n)
):
    logger.info(f"Decoding start")
    start = time.time()
    feature_lengths = torch.tensor([indices.shape[1]], device=args.device)
    fake_audios, _ = model_vqgan.decode(
        indices=indices[None], feature_lengths=feature_lengths
    )
    audio_time = fake_audios.shape[-1] / model_vqgan.spec_transform.sample_rate
    end = time.time()
    logger.info(f"Decoding end: time taken: {end - start:3f} seconds")
    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.10f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    return fake_audio


def inference_save_get_stream(
    indices: torch.Tensor,
    splice_start: int, # inclusive
    splice_end: int, # exclusive
    output_dir: str,
    upsample_factor: int = 2048,
):
    cum_audio = vqgan_inference(indices)
    cum_output_path = os.path.join(output_dir, f"cum-{splice_start}-{splice_end}.wav")
    sf.write(cum_output_path, cum_audio, model_vqgan.spec_transform.sample_rate)
    logger.info(f"Saved CUM audio to {cum_output_path}")

    cumnew_audio = cum_audio[splice_start*upsample_factor:splice_end*upsample_factor]
    cumnew_output_path = os.path.join(output_dir, f"cumnew-{splice_start}-{splice_end}.wav")
    sf.write(cumnew_output_path, cumnew_audio, model_vqgan.spec_transform.sample_rate)
    logger.info(f"Saved CUMNEW audio to {cumnew_output_path}")

    onlynew_audio = vqgan_inference(indices[splice_start:splice_end])
    onlynew_output_path = os.path.join(output_dir, f"onlynew-{splice_start}-{splice_end}.wav")
    sf.write(onlynew_output_path, onlynew_audio, model_vqgan.spec_transform.sample_rate)
    logger.info(f"Saved ONLYNEW audio to {onlynew_output_path}")

    return cumnew_audio


def combine_outputs(output_dir: str) -> None:
    # check quality of stitching together vqgan outputs without postprocessing
    # compare between providing past (already streamed) tokens and only new tokens
    contents = os.listdir(output_dir)
    
    cumnew_wavs = list(filter(lambda x: x.startswith("cumnew"), contents))
    sorted_cumnew = sorted(cumnew_wavs, key=lambda x: int(x.split(".")[0].split("-")[1]))
    logger.info(f"Combining wav files: {sorted_cumnew}")
    cumnew_combined = AudioSegment.empty()
    for wav in sorted_cumnew:
        audio = AudioSegment.from_wav(wav)
        cumnew_combined += audio  # Append
    cumnew_output_path = os.path.join(output_dir, "_cumnew-combined.wav")
    cumnew_combined.export(cumnew_output_path, format="wav")
    logger.info(f"Combined wav saved to {cumnew_output_path}")

    onlynew_wavs = list(filter(lambda x: x.startswith("onlynew"), contents))
    sorted_onlynew = sorted(onlynew_wavs, key=lambda x: int(x.split(".")[0].split("-")[1]))
    logger.info(f"Combining wav files: {sorted_onlynew}")
    onlynew_combined = AudioSegment.empty()
    for wav in sorted_onlynew:
        audio = AudioSegment.from_wav(wav)
        onlynew_combined += audio  # Append
    onlynew_output_path = os.path.join(output_dir, "_cumnew-combined.wav")
    onlynew_combined.export(onlynew_output_path, format="wav")
    logger.info(f"Combined wav saved to {onlynew_output_path}")






import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from fish_speech.models.text2semantic.llama import BaseModelArgs
from fish_speech.text import clean_text, split_text
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync_agent(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def sample_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs_agent(
        logits=logits[:, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync_agent(probs)
    return idx_next, probs


def decode_one_token_ar_agent(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # print(x, input_pos)
    x = model.forward_generate(x, input_pos)
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    sampling_kwargs_main = sampling_kwargs.copy()
    sampling_kwargs_main["temperature"] = 0.1
    sampling_kwargs_main["top_p"] = 0.1
    sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample_agent(
            logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample_agent(
            logits,
            previous_tokens=(
                previous_tokens[:, codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~torch.isin(codebooks[:, :1, :], semantic_ids_tensor),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks


def decode_one_token_naive_agent(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    codebooks = [
        sample(
            x.token_logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample_agent(
                x.codebook_logits[:, :, i],
                previous_tokens=(
                    previous_tokens[:, i + 1] if previous_tokens is not None else None
                ),
                **sampling_kwargs,
            )[0]
        )

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~torch.isin(codebooks[:, :1, :], semantic_ids_tensor),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    sampling_kwargs_main = sampling_kwargs.copy()
    # sampling_kwargs_main["temperature"] = 0.1
    # sampling_kwargs_main["top_p"] = 0.1
    # sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample(
            x.logits,
            previous_tokens=(
                previous_tokens[0] if previous_tokens is not None else None
            ),  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    hidden_states = x.hidden_states

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=0)
    # semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    # codebooks[1:, :] = torch.masked_fill(
    #     codebooks[1:, :], ~torch.isin(codebooks[:1, :], semantic_ids_tensor), CODEBOOK_PAD_TOKEN_ID
    # )

    # each generation is 9 dimensional. since codebook is based off of grouped finite scalar quantization. 
    # normally you'd have just prob over all tokens for a single codebook
    # but in this case, since you have 9 codebooks, you need to have 9 prob distributions
    # how does input work then? input still has to be a latent vector. latent vector assembled through concatenating 9 codebooks?
    # yeah. guessing forward_generate creates concatenates sublatents into full latent vector
    # print("codebook shape", codebooks.shape)
    return codebooks


def decode_one_token_naive(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    sampling_kwargs_main = sampling_kwargs.copy()
    sampling_kwargs_main["temperature"] = 0.1
    sampling_kwargs_main["top_p"] = 0.1
    sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample(
            x.logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample(
                x.codebook_logits[:, :, i],
                previous_tokens=(
                    previous_tokens[i + 1] if previous_tokens is not None else None
                ),
                **sampling_kwargs,
            )[0]
        )

    return torch.stack(codebooks, dim=0)


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # print("generating token ", i)
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
            if torch.cuda.is_available()
            else nullcontext()
        ):  # Actually better for Inductor to codegen attention here
            # print(cur_token.shape)
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            )
            # print("decoded a token:", next_token.shape)
            # technically this has 9 shape atm. the 1st dim is in the 10k's so i think it can probably be removed
            # the inference_cum for vqgan is 8 dim. there's code in generate_long(), where the indexing of y is y[1:, ...]
            # the first codebook is tossed
            yield ("next_token", next_token)

        input_pos += 1
        # print("input_pos", input_pos)
        # print("previous tokens[0]", previous_tokens[:, 0])
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break
    
    print("previous tokens", previous_tokens.shape)
    # return previous_tokens[:, : i + 1]
    yield ("previous_token", previous_tokens[:, : i + 1])


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: NaiveTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    # semantic_id = model.tokenizer.convert_tokens_to_ids("<|semantic|>")
    semantic_ids = [
        model.tokenizer.get_token_id(f"<|semantic:{i}|>") for i in range(1024)
    ]

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    codebook_dim = 1 + model.config.num_codebooks
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (
        decode_one_token_naive
        if isinstance(model, NaiveTransformer)
        else decode_one_token_ar
    )

    next_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )

    seq[:, T : T + 1] = next_token
    print(next_token.shape)
    print(seq.shape)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    # x = decode_n_tokens(
    x_generator = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        decode_one_token=decode_one_token,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )

    for x_gen in x_generator:
        print("xgen sanitycheck")
        if x_gen[0] == "previous_token": # default functionality
            seq = seq[:, : T + 1 + x_gen[1].size(1)]
            seq[:, T + 1 :] = x_gen[1]
            yield ("seq", seq)
        elif x_gen[0] == "next_token": # keep passing it up
            yield x_gen

    # # x = torch.cat(generated_tokens, dim=1)
    # seq = seq[:, : T + 1 + x.size(1)]
    # seq[:, T + 1 :] = x
    # return seq



def decode_n_tokens_agent(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    batch_size = cur_token.size(0)
    previous_tokens = torch.zeros(
        (batch_size, model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=cur_token.device)
    finished = finished | (cur_token[:, 0, -1] == im_end_id)
    start_time = time.time()

    for i in tqdm(range(num_new_tokens), desc="Decoding: ", total=num_new_tokens):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :, :win_size]
        else:
            window = previous_tokens[:, :, i - win_size : i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(batch_size, model.config.num_codebooks + 1, -1)
        previous_tokens[:, :, i : i + 1] = next_token.view(
            batch_size, model.config.num_codebooks + 1, -1
        )

        yield cur_token.cpu()

        finished = finished | (cur_token[:, 0, -1] == im_end_id)
        if finished.all() or (
            0 < early_stop_threshold < 1
            and finished.sum() >= round(batch_size * early_stop_threshold)
        ):
            break

    total_time = time.time() - start_time
    generated_tokens = i + 1
    tokens_per_second = (generated_tokens / total_time) * batch_size
    logger.info(
        f"Decoded {generated_tokens} x {batch_size} tokens in {total_time:.2f}s ({tokens_per_second:.2f} tokens/s)"
    )


@torch.no_grad()
@torch.inference_mode()
def generate_agent(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    num_samples: int = 1,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (
        decode_one_token_naive_agent
        if isinstance(model, NaiveTransformer)
        else decode_one_token_ar_agent
    )
    next_token = prefill_decode(
        model,
        prompt,
        input_pos,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    ).view(num_samples, codebook_dim, -1)
    yield next_token.cpu()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    yield from decode_n_tokens_agent(
        model,
        next_token,
        input_pos,
        max_new_tokens - 1,
        im_end_id=im_end_id,
        semantic_ids=semantic_ids,
        decode_one_token=decode_one_token,
        early_stop_threshold=early_stop_threshold,
        **sampling_kwargs,
    )


def encode_tokens(
    tokenizer,
    string,
    device="cuda",
    prompt_tokens=None,
    num_codebooks=4,
):
    string = clean_text(string)

    messages = []
    messages.append(
        Message(
            role="user",
            parts=[TextPart(text=string)],
            cal_loss=False,
        )
    )

    if prompt_tokens is not None:
        if prompt_tokens.ndim == 3:
            assert (
                prompt_tokens.shape[0] == 1
            ), "3D prompt tokens should have shape (1, num_codebooks, seq_len)"
            prompt_tokens = prompt_tokens[0]

        assert prompt_tokens.ndim == 2, "Prompt tokens should be 2D tensor"

        if prompt_tokens.shape[0] > num_codebooks:
            logger.warning(
                f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
            )
            prompt_tokens = prompt_tokens[:num_codebooks]

        vq_part = VQPart(codes=prompt_tokens.to(device))

        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>"), vq_part],
                cal_loss=False,
            )
        )
    else:
        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>")],
                cal_loss=False,
                add_im_end=False,
            )
        )

    conversation = Conversation(messages=messages)
    # conversation.visualize(tokenizer)
    encoded = conversation.encode_for_inference(
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
    )

    return encoded.to(device)


def load_model_llm(checkpoint_path, device, precision, compile=False, is_agent=False):
    model: Union[NaiveTransformer, DualARTransformer] = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = (
            decode_one_token_ar_agent if is_agent else decode_one_token_ar
        )
        logger.info("Using DualARTransformer")
    else:
        decode_one_token = (
            decode_one_token_naive_agent if is_agent else decode_one_token_naive
        )
        logger.info("Using NaiveTransformer")

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    iterative_prompt: bool = True,
    max_length: int = 2048,
    chunk_length: int = 150,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    im_end_id = tokenizer.get_token_id("<|im_end|>")

    encoded = []
    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    encoded_prompts = [
        Conversation(
            messages=[
                Message(
                    role="system",
                    parts=[TextPart(text="Speak out the provided text.")],
                    cal_loss=False,
                )
            ]
        )
        .encode_for_inference(
            tokenizer=tokenizer,
            num_codebooks=model.config.num_codebooks,
        )
        .to(device)
    ]

    if use_prompt:
        for idx, (t, c) in enumerate(zip(prompt_text, prompt_tokens)):
            encoded_prompts.append(
                encode_tokens(
                    tokenizer,
                    string=t,
                    device=device,
                    prompt_tokens=c,
                    num_codebooks=model.config.num_codebooks,
                )
            )

    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                device=device,
                num_codebooks=model.config.num_codebooks,
            )
        )
        logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        print(sample_idx)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        seg_idx = 0

        while seg_idx < len(encoded):
            logger.info(
                f"Generating sentence {seg_idx + 1}/{len(encoded)} of sample {sample_idx + 1}/{num_samples}"
            )

            seg = encoded[seg_idx]
            global_encoded.append(seg)

            lengths = reversed([seg.size(1) for seg in global_encoded])

            # Pick last 2000 tokens
            count = 0
            for i, length in enumerate(lengths):
                count += length
                if count + length > max_length - 1024 - sum(
                    t.shape[1] for t in encoded_prompts
                ):
                    break

            if i != 0 and i % 2 == 0:
                i -= 1

            # Rotate the list, always make sure first segment is included to avoid drift
            if i < len(global_encoded) - 2:
                partial_encoded = global_encoded[:2] + global_encoded[-i:]
            else:
                partial_encoded = global_encoded

            if use_prompt:
                partial_encoded = encoded_prompts + partial_encoded

            cat_encoded = torch.cat(partial_encoded, dim=1)
            prompt_length = cat_encoded.size(1)

            t0 = time.perf_counter()
            # y = generate(
            y_generator = generate(
                model=model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            for y_gen in y_generator:
                print("ygen sanity check")
                if y_gen[0] == "seq":
                    y = y_gen[1]
                    if sample_idx == 0 and seg_idx == 0 and compile:
                        logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    t = time.perf_counter() - t0

                    tokens_generated = y.size(1) - prompt_length
                    tokens_sec = tokens_generated / t
                    logger.info(
                        f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
                    )
                    logger.info(
                        f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
                    )

                    if torch.cuda.is_available():
                        logger.info(
                            f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
                        )

                    # Put the generated tokens
                    # since there is <im_end>, we remove last token
                    codes = y[1:, prompt_length + 1 :].clone()
                    assert (codes >= 0).all(), f"Negative code found"

                    decoded = y[:, prompt_length:].clone()
                    # But for global encoding, we should keep the <im_end> token

                    global_encoded.append(decoded)
                    assert (codes >= 0).all(), f"Negative code found: {codes}"
                    yield GenerateResponse(action="sample", codes=codes, text=texts[seg_idx])
                    seg_idx += 1
                elif y_gen[0] == "next_token":
                    print("ygen - yielded next token")
                    yield y_gen
        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = load_model_llm(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


def launch_thread_safe_queue_agent(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = BaseModelArgs.from_pretrained(checkpoint_path)

    def worker():
        model, decode_one_token = load_model_llm(
            checkpoint_path, device, precision, compile=compile, is_agent=True
        )

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for token in generate_agent(
                    model=model,
                    decode_one_token=decode_one_token,
                    **kwargs,
                ):
                    response_queue.put(token)

                response_queue.put("stop")
            except Exception as e:
                import traceback

                logger.exception(f"Error in worker: {traceback.format_exc()}")
                response_queue.put("error")

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue, tokenizer, config


logger.info("Loading LLM ...")
t0 = time.time()
model_llm, decode_one_token = load_model_llm(
    args.checkpoint_path, args.device, torch.half if args.half else torch.bfloat16, compile=args.compile
)
with torch.device(args.device):
    model_llm.setup_caches(
        max_batch_size=1,
        max_seq_len=model_llm.config.max_seq_len,
        dtype=next(model_llm.parameters()).dtype,
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()
logger.info(f"Time to load LLM: {time.time() - t0:.02f} seconds")


def generate_audio_chunks(
    text: str,
    prompt_text: Optional[list[str]], # TODO: replace with prompt_dir in the future
    prompt_tokens: Optional[list[Path]],
    output_dir: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    device: str = args.device,
    compile: bool = args.compile,
    seed: int = 16,
    iterative_prompt: bool = True,
    chunk_length: int = 100,
):
    print(text)
    os.makedirs(output_dir, exist_ok=True)

    if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
        raise ValueError(f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same")
    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    generator = generate_long(
        model=model_llm,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

def generate_audio_chunks2(
    text: str,
    prompt_text: Optional[list[str]], # TODO: replace with prompt_dir in the future
    prompt_tokens: Optional[list[Path]],
    output_dir: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    seed: int = 16,
    iterative_prompt: bool = True,
    chunk_length: int = 100,
):
    print("1")
    os.makedirs(output_dir, exist_ok=True)

    if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )

    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model_llm,
        device=args.device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=args.compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    idx = 0
    codes = [] # contains token chunks - default behavior
    tokens = [] # contains individual tokens - custom behavior w. streaming
    last_stream_end = 0 # token the last stream to client ended on

    print("starting generation")
    start = time.time()
    for response in generator:
        # print("response sanitycheck", response)
        # print(response)
        if isinstance(response, GenerateResponse):
            if response.action == "sample":
                codes.append(response.codes)
                logger.info(f"Sampled text: {response.text}")
            elif response.action == "next":
                if codes:
                    codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                    np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                    logger.info(f"Saved codes to {codes_npy_path}")
                    print("torch final dim: ", torch.cat(codes, dim=1).shape)

                    tokens_array = torch.cat(tokens, dim=1)
                    print(tokens_array.shape)
                logger.info(f"Next sample")
                codes = []
                idx += 1
            else:
                logger.error(f"Error: {response}")
        else:
            if response[0] == "next_token":
                tokens.append(response[1][1:, :])
                if len(tokens) % 10 == 0: # on every tenth token do a generation
                    # tokens_array = np.concatenate(tokens, axis=1)
                    tokens_array = torch.cat(tokens, dim=1)
                    print(tokens_array.shape)

                    new_audio = inference_save_get_stream(
                        tokens_array,
                        last_stream_end, len(tokens),
                        "./temp"
                    )
                    buffer = BytesIO()
                    sf.write(buffer, new_audio, model_vqgan.spec_transform.sample_rate, format="WAV")
                    buffer.seek(0)
                    yield buffer.read()
                    last_stream_end = len(tokens)

    combine_outputs(output_dir)

    end = time.time()
    print(f"finished generation: time taken: {end - start}")


# class FishRequest(BaseModel):
#     name: str = Field(
#         ...,
#         max_length=50,
#         pattern=r'^[a-zA-Z0-9]+$'
#     )
#     tts_text: str
#     personality: Literal["ling", "chen", "surtr"] = "ling"
# @app.post("/tts/stream")
# async def tts_stream(request: FishRequest):
#     text = "他人の指導役はもうごめんだ。一般人たちと雁首揃えてアーツごっこするなんて興味ない。"
#     prompt_text = "あんた、自分の仕事も全うできないからって、私に助けろっていうの？"
#     prompt_tokens = "surtr.npy"

#     output_dir = os.path.join(args.output_base_path, request.name)
#     os.makedirs(output_dir)

#     try:
#         return StreamingResponse(
#             generate_audio_chunks(text, prompt_text, prompt_tokens, output_dir),
#             media_type="audio/wav"
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


text = "他人の指導役はもうごめんだ。一般人たちと雁首揃えてアーツごっこするなんて興味ない。"
prompt_text = ["あんた、自分の仕事も全うできないからって、私に助けろっていうの？"]
prompt_tokens = ["surtr.npy"]

print("0")

generate_audio_chunks(text, prompt_text, prompt_tokens, output_dir="./output/temp")

print("2")
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting server... Imports loaded.")
#     uvicorn.run(app, host="0.0.0.0", port=8000)