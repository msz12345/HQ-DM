"""Validate the integer HQ-DM operator against the checkpoint's fake-quant path."""

import gc

import torch
import torch.nn as nn

from tools._bootstrap import PROJECT

from omegaconf import OmegaConf

from hqdm_int.hq_int import HQIntModule, load_extension
from ldm.util import instantiate_from_config
from hqdm.quantization.single_hadamard import SingleH32QuantModule, SimpleDequantizer
from hqdm.quantization.model import SingleH32QuantModel


def load_quantized_model(steps=20):
    config = OmegaConf.load(PROJECT / "configs/latent-diffusion/cin256-v2.yaml")
    base_checkpoint = torch.load(
        PROJECT / "models/ldm/cin256-v2/model.ckpt", map_location="cpu"
    )
    model = instantiate_from_config(config.model)
    model.load_state_dict(base_checkpoint["state_dict"], strict=False)
    del base_checkpoint

    qnn = SingleH32QuantModel(
        model=model.model.diffusion_model,
        weight_quant_params={
            "n_bits": 4,
            "channel_wise": True,
            "scale_method": "mse",
        },
        act_quant_params={
            "n_bits": 4,
            "symmetric": True,
            "scale_method": "max",
            "leaf_param": True,
        },
        need_init=True,
        num_steps=steps,
    )
    qnn.set_first_last_layer_to_8bit()
    qnn.set_quant_state(True, True)
    for module in qnn.modules():
        if isinstance(module, SingleH32QuantModule):
            module.intn_dequantizer = SimpleDequantizer(
                uaq=module.weight_quantizer, weight=module.weight
            )
            module.weight.data = module.weight.data.byte()
            if not hasattr(module.act_quantizer, "delta_list"):
                module.act_quantizer.delta_list = nn.Parameter(
                    torch.full((steps,), 0.005, dtype=torch.float32)
                )
            module.act_quantizer.inited = True

    setattr(model.model, "diffusion_model", qnn)
    checkpoint = torch.load(
        PROJECT / f"quantw4a4_{steps}steps_SingleH32.pth", map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    del checkpoint
    model.cuda().eval()
    return model


def compare(name, source, input_tensor, step=19):
    source.act_quantizer.current_step = step
    with torch.inference_mode():
        expected = source(input_tensor)
        integer = HQIntModule(source, name)
        integer.reset_step(step)
        actual = integer(input_tensor)
    error = (actual - expected).abs()
    scale = expected.abs().clamp_min(1e-5)
    print(
        f"{name:28s} shape={tuple(input_tensor.shape)!s:18s} "
        f"digits={2 if integer.uses_high_digit else 1} "
        f"max_abs={error.max().item():.7g} "
        f"mean_abs={error.mean().item():.7g} "
        f"max_rel={(error / scale).max().item():.7g}"
    )
    # The fake-quant reference finishes in TF32/FP32 cuBLAS/cuDNN, while the
    # integer path has exact INT32 accumulation and one FP32 epilogue.  The
    # resulting reassociation noise is typically around 1e-4 and below 2e-3.
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)


def main():
    torch.manual_seed(23)
    load_extension(verbose=False)
    model = load_quantized_model(steps=20)
    modules = [
        (name, module)
        for name, module in model.model.diffusion_model.named_modules()
        if isinstance(module, SingleH32QuantModule)
    ]

    conv3_name, conv3 = next(
        (name, module)
        for name, module in modules
        if len(module.ori_shape) == 4
        and tuple(module.ori_shape[2:]) == (3, 3)
        and module.ori_shape[1] == 192
        and module.ori_shape[0] == 192
    )
    conv1_name, conv1 = next(
        (name, module)
        for name, module in modules
        if len(module.ori_shape) == 4 and tuple(module.ori_shape[2:]) == (1, 1)
    )
    linear_name, linear = next(
        (name, module)
        for name, module in modules
        if len(module.ori_shape) == 2
        and module.ori_shape[0] == 384
        and module.ori_shape[1] == 384
    )

    compare(conv3_name, conv3, torch.randn(2, 192, 16, 16, device="cuda"))
    compare(
        conv1_name,
        conv1,
        torch.randn(2, int(conv1.ori_shape[1]), 16, 16, device="cuda"),
    )
    compare(
        linear_name + " [rank2]",
        linear,
        torch.randn(64, int(linear.ori_shape[1]), device="cuda"),
    )
    compare(
        linear_name + " [rank3]",
        linear,
        torch.randn(2, 64, int(linear.ori_shape[1]), device="cuda"),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("checkpoint operator validation passed")


if __name__ == "__main__":
    main()
