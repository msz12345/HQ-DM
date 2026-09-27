"""Synchronized, warm benchmark for FP32, fake-quant HQ-DM, and INT HQ-DM."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image


from tools._bootstrap import PROJECT

from omegaconf import OmegaConf

from hqdm_int.cuda_graph import GraphUNetRunner
from hqdm_int.hq_int import replace_single_h_modules, reset_timestep_scales
from tools.validate_checkpoint import load_quantized_model
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from hqdm.quantization.single_hadamard import SingleH32QuantModule


def load_fp32_model():
    config = OmegaConf.load(PROJECT / "configs/latent-diffusion/cin256-v2.yaml")
    checkpoint = torch.load(
        PROJECT / "models/ldm/cin256-v2/model.ckpt", map_location="cpu"
    )
    model = instantiate_from_config(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    del checkpoint
    return model.cuda().eval()


def reset_fake_quant_steps(model, steps):
    for module in model.modules():
        if isinstance(module, SingleH32QuantModule):
            module.act_quantizer.current_step = steps - 1


def prepare_model(mode, steps, channels_last=False):
    prepack_seconds = 0.0
    conversion = None
    if mode in ("fp32", "fp16"):
        model = load_fp32_model()
    else:
        model = load_quantized_model(steps=steps)
    if mode in ("fp32", "fp16") and channels_last:
        model.model.diffusion_model.to(memory_format=torch.channels_last)
    if mode in ("int", "int_amp"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        conversion = replace_single_h_modules(
            model.model.diffusion_model,
            channels_last_outputs=channels_last,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        prepack_seconds = time.perf_counter() - start
    return model, conversion, prepack_seconds


@torch.inference_mode()
def run_once(model, sampler, conditioning, unconditional, x_t, args):
    if hasattr(model.apply_model, "reset"):
        model.apply_model.reset()
    if args.mode in ("int", "int_amp"):
        reset_timestep_scales(model, args.steps - 1)
    elif args.mode == "fake":
        reset_fake_quant_steps(model, args.steps)

    torch.cuda.synchronize()
    start = time.perf_counter()
    latent, _ = sampler.sample(
        S=args.steps,
        conditioning=conditioning,
        batch_size=args.batch_size,
        shape=[3, 64, 64],
        verbose=False,
        unconditional_guidance_scale=args.guidance,
        unconditional_conditioning=unconditional,
        eta=0.0,
        x_T=x_t,
    )
    torch.cuda.synchronize()
    denoise_seconds = time.perf_counter() - start

    start = time.perf_counter()
    # The channels-last integer path may propagate a non-default latent stride.
    # Normalize the tiny 3x64x64 tensor here so VAE timing is comparable across
    # modes and does not depend on the final UNet output layout.
    image = model.decode_first_stage(latent.contiguous())
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - start
    return image, denoise_seconds, decode_seconds


def save_image(image, path):
    array = 255.0 * rearrange(image[0], "c h w -> h w c").cpu().numpy()
    Image.fromarray(array.astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fp32", "fp16", "fake", "int", "int_amp"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--class-label", type=int, default=333)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="keep quantized convolution outputs in NCHW channels-last storage",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    args = parser.parse_args()
    if args.channels_last and args.mode not in (
        "int",
        "int_amp",
        "fp16",
        "fp32",
    ):
        raise ValueError(
            "--channels-last is only supported by integer and float modes"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, conversion, prepack_seconds = prepare_model(
        args.mode, args.steps, channels_last=args.channels_last
    )

    if args.mode in ("fp16", "int_amp"):
        eager_apply_model = model.apply_model

        def amp_unet_apply(image, timestep, context):
            # Keep DDIM state and parameter storage in FP32, while allowing the
            # remaining floating-point convolutions/linears to use FP16 Tensor
            # Cores.  Custom integer operators retain their explicit dtypes.
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if args.channels_last and args.mode == "fp16":
                    image = image.contiguous(memory_format=torch.channels_last)
                return eager_apply_model(image, timestep, context).float()

        model.apply_model = amp_unet_apply
    sampler = DDIMSampler(model)

    labels = torch.full(
        (args.batch_size,), args.class_label, dtype=torch.long, device=model.device
    )
    unconditional_labels = torch.full_like(labels, 1000)
    conditioning = model.get_learned_conditioning({model.cond_stage_key: labels})
    unconditional = model.get_learned_conditioning(
        {model.cond_stage_key: unconditional_labels}
    )
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    x_t = torch.randn(
        (args.batch_size, 3, 64, 64), device=model.device, generator=generator
    )

    graph_capture_seconds = 0.0
    if args.cuda_graphs:
        if args.mode not in ("int", "int_amp", "fp16", "fp32"):
            raise ValueError(
                "the graph runner targets --mode int, int_amp, fp16, or fp32"
            )
        example_image = torch.cat((x_t, x_t), dim=0)
        if args.channels_last and args.mode in ("fp16", "fp32"):
            example_image = example_image.contiguous(
                memory_format=torch.channels_last
            )
        example_timestep = torch.full(
            (2 * args.batch_size,), 951, dtype=torch.long, device=model.device
        )
        example_context = torch.cat((unconditional, conditioning), dim=0)
        torch.cuda.synchronize()
        capture_start = time.perf_counter()
        runner = GraphUNetRunner(
            model,
            example_image,
            example_timestep,
            example_context,
            scale_steps=args.steps,
            compute_dtype=None,
        )
        model.apply_model = runner
        torch.cuda.synchronize()
        graph_capture_seconds = time.perf_counter() - capture_start

    torch.cuda.reset_peak_memory_stats()
    last_image = None
    for _ in range(args.warmup):
        last_image, _, _ = run_once(
            model, sampler, conditioning, unconditional, x_t, args
        )

    measurements = []
    for repeat in range(args.repeats):
        last_image, denoise, decode = run_once(
            model, sampler, conditioning, unconditional, x_t, args
        )
        item = {
            "repeat": repeat,
            "denoise_seconds": denoise,
            "decode_seconds": decode,
            "total_seconds": denoise + decode,
        }
        measurements.append(item)
        print(json.dumps(item, sort_keys=True))

    denoise_values = [item["denoise_seconds"] for item in measurements]
    decode_values = [item["decode_seconds"] for item in measurements]
    total_values = [item["total_seconds"] for item in measurements]
    summary = {
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "class_label": args.class_label,
        "guidance": args.guidance,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "denoise_seconds_mean": float(np.mean(denoise_values)),
        "denoise_seconds_p50": float(np.median(denoise_values)),
        "decode_seconds_mean": float(np.mean(decode_values)),
        "total_seconds_mean": float(np.mean(total_values)),
        "images_per_second": args.batch_size / float(np.mean(total_values)),
        "prepack_seconds_excluded": prepack_seconds,
        "cuda_graphs": args.cuda_graphs,
        "channels_last": args.channels_last,
        "graph_capture_seconds_excluded": graph_capture_seconds,
        "conversion": conversion,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "image_sum": float(last_image.sum().item()),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "measurements": measurements,
    }
    result_path = args.output_dir / f"benchmark_{args.mode}.json"
    image_path = args.output_dir / f"sample_{args.mode}.png"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    save_image(last_image, image_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved {result_path}")
    print(f"saved {image_path}")


if __name__ == "__main__":
    main()
