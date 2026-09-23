# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shape-aware policy selection for the FlyDSL A16W16 GEMM."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes

from .kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    GEMM_A16W16_DTYPE_FP32,
    make_gemm_a16w16_param_and_validate,
)

__all__ = [
    "BASE_SELECTIONS",
    "EXTENDED_VALUES",
    "FLYDSL_A16W16_SPACES",
    "GemmConfigPruner",
    "get_flydsl_a16w16_configs",
    "parse_hipblaslt_kernel",
    "subspace_from_hipblaslt_kernel",
]

FLYDSL_A16W16_SPACES = ("base", "extended", "subspace")

# Original gfx950 policy lists (before hipBLASLt-driven tile expansion).
BASE_SELECTIONS = {
    "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
    "block_n": [16, 32, 64, 80, 96, 128, 256],
    "block_k": [64, 128, 256],
    "stages": list(range(2, 10)),
    "m_waves": [1, 2, 4],
    "n_waves": [1, 2, 4],
    "k_waves": [1, 2],
    "group_m": [0, 4],
    "use_half_tile_interleaved": [False, True],
}

# Extra values that explode the cartesian product. Opt in via space=extended
# or a hipBLASLt-derived subspace.
EXTENDED_VALUES = {
    "block_m": [160, 192, 224],
    "block_n": [160, 192, 224],
    "block_k": [512],
    "k_waves": [4],
}

_BASE_MMA_ITER_CAP = 4
_EXTENDED_MMA_ITER_CAP = 12


@dataclass(frozen=True)
class GemmConfigPruner:
    """Prune A16W16 policies using tile efficiency and estimated occupancy."""

    m: int
    n: int
    k: int
    device_props: object
    element_bytes: int
    target_waves_per_cu: int = 8
    max_split_grid_rounds: int = 2
    max_tile_grid_ratio: int = 4

    @staticmethod
    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    def _tiles(self, config):
        return self._ceil_div(self.m, config["block_m"]) * self._ceil_div(
            self.n, config["block_n"]
        )

    def _iou(self, config):
        split_k = config["split_k"]
        padded_m = self._ceil_div(self.m, config["block_m"]) * config["block_m"]
        padded_n = self._ceil_div(self.n, config["block_n"]) * config["block_n"]
        part_k = self.k // split_k
        padded_k = (
            self._ceil_div(part_k, config["block_k"]) * config["block_k"] * split_k
        )
        return self.m * self.n * self.k / (padded_m * padded_n * padded_k)

    def _occupancy(self, config):
        props = self.device_props
        waves = config["m_waves"] * config["n_waves"] * config["k_waves"]
        lds = max(
            config["stages"]
            * (config["block_m"] + config["block_n"])
            * config["block_k"]
            * self.element_bytes,
            config["k_waves"]
            * config["block_m"]
            * config["block_n"]
            * self.element_bytes,
        )
        # A workgroup may occupy the whole per-CU LDS here, so the capacity the
        # chip table reports per workgroup is also the per-CU figure.
        lds_per_cu = get_lds_capacity_bytes()
        resident = (
            min(
                props.max_threads_per_multi_processor // props.warp_size // waves,
                lds_per_cu // lds,
            )
            * waves
        )
        grid = (
            self._tiles(config)
            * config["split_k"]
            * waves
            / props.multi_processor_count
        )
        return min(self.target_waves_per_cu, resident, grid)

    def prune(self, configs):
        if not configs:
            return configs
        ious = [self._iou(config) for config in configs]
        keep_ratio = max(0.625, 1 - self.m / 160, 1 - 120 / self.m)
        min_iou = max(ious) * keep_ratio
        num_cus = self.device_props.multi_processor_count
        max_tiles = (
            max(num_cus, min(self._tiles(config) for config in configs))
            * self.max_tile_grid_ratio
        )
        kept = []
        for config, iou in zip(configs, ious):
            tiles = self._tiles(config)
            max_split_k = (
                1
                if tiles >= num_cus
                else self._ceil_div(self.max_split_grid_rounds * num_cus, tiles)
            )
            if (
                iou >= min_iou
                and tiles <= max_tiles
                and config["split_k"] <= max_split_k
                and (config["group_m"] == 0 or (tiles >= num_cus and tiles % 8 == 0))
            ):
                kept.append(config)

        best = {}
        keep = set()
        for index, config in sorted(
            enumerate(kept), key=lambda item: (item[1]["k_waves"], item[0])
        ):
            key = tuple(
                (name, value) for name, value in config.items() if name != "k_waves"
            )
            occupancy = self._occupancy(config)
            if occupancy > best.get(key, -1.0) + 1e-9:
                best[key] = occupancy
                keep.add(index)
        return [config for index, config in enumerate(kept) if index in keep]


def parse_hipblaslt_kernel(name: str) -> dict:
    """Decode TensileLite name tokens used by hipBLASLt winners."""
    parsed: dict = {"name": name, "custom": name.startswith("Custom_")}
    match = re.search(r"_MT(\d+)x(\d+)x(\d+)_", name)
    if match:
        parsed["block_m"], parsed["block_n"], parsed["block_k"] = map(
            int, match.groups()
        )
    match = re.search(r"_MI(\d+)x(\d+)x(\d+)_", name)
    if match:
        parsed["mi"] = tuple(map(int, match.groups()))
    match = re.search(r"_MIWT(\d+)_(\d+)_", name)
    if match:
        parsed["miwt"] = tuple(map(int, match.groups()))
    match = re.search(r"_WG(\d+)_(\d+)_(\d+)", name)
    if match:
        parsed["wg"] = tuple(map(int, match.groups()))
        parsed["k_waves"] = parsed["wg"][2]
    if "block_m" in parsed and "miwt" in parsed:
        wave_tile_m = parsed["miwt"][0] * 16
        wave_tile_n = parsed["miwt"][1] * 16
        if wave_tile_m and parsed["block_m"] % wave_tile_m == 0:
            parsed["m_waves"] = parsed["block_m"] // wave_tile_m
        if wave_tile_n and parsed["block_n"] % wave_tile_n == 0:
            parsed["n_waves"] = parsed["block_n"] // wave_tile_n
    return parsed


def subspace_from_hipblaslt_kernel(name: str) -> dict[str, list]:
    """Extended-selection extras implied by a hipBLASLt/Tensile kernel name."""
    parsed = parse_hipblaslt_kernel(name)
    extra: dict[str, list] = {}
    for key in ("block_m", "block_n", "block_k", "k_waves"):
        value = parsed.get(key)
        if value in EXTENDED_VALUES.get(key, ()):
            extra.setdefault(key, []).append(value)
    return extra


def _merge_unique(base, extra):
    merged = list(base)
    for value in extra:
        if value not in merged:
            merged.append(value)
    return merged


def _merge_extra_maps(*maps: dict[str, list] | None) -> dict[str, list]:
    merged: dict[str, list] = {}
    for extra in maps:
        if not extra:
            continue
        for key, values in extra.items():
            merged[key] = _merge_unique(merged.get(key, []), values)
    return merged


def _build_selections(k: int, extra: dict[str, list]) -> dict[str, list]:
    split_k_candidates = [1]
    split_k_candidates.extend(split_k for split_k in range(2, 10) if k % split_k == 0)
    selections = {
        name: _merge_unique(values, extra.get(name, []))
        for name, values in BASE_SELECTIONS.items()
    }
    selections["split_k"] = split_k_candidates
    return selections


def _neighborhood_overrides(parsed: dict, extra: dict[str, list]) -> dict[str, list]:
    overrides = dict(extra)
    for key in ("block_m", "block_n", "block_k", "m_waves", "n_waves", "k_waves"):
        if key in parsed:
            overrides[key] = [parsed[key]]
    return {key: values for key, values in overrides.items() if values}


def _config_uses_extended(config) -> bool:
    return any(
        config.get(name) in extra_values
        for name, extra_values in EXTENDED_VALUES.items()
    )


def _is_256x256_pht(config, n_waves) -> bool:
    return (
        config["use_half_tile_interleaved"]
        and config["block_m"] == 256
        and config["block_n"] == 256
        and config["block_k"] == 64
        and config["stages"] == 2
        and config["split_k"] == 1
        and config["m_waves"] == 2
        and config["n_waves"] == n_waves
        and config["k_waves"] == 1
    )


def _is_256x224_pft(config) -> bool:
    return (
        not config["use_half_tile_interleaved"]
        and config["block_m"] == 256
        and config["block_n"] == 224
        and config["block_k"] == 64
        and config["stages"] == 2
        and config["split_k"] == 1
        and config["m_waves"] == 2
        and config["n_waves"] == 2
        and config["k_waves"] == 1
    )


def _is_known_llvm_abort(config) -> bool:
    # gfx950 FlyDSL codegen: "Virtual register defs don't dominate all uses"
    # on hgemm_bf16_t160x128x64x3_ksd_w1x1x1. LLVM abort() kills the tuner
    # worker; drop this neighborhood rather than dumping machine code.
    return (
        config["block_m"] == 160
        and config["block_n"] == 128
        and config["block_k"] == 64
        and config["stages"] == 3
    )


def get_flydsl_a16w16_configs(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
    has_bias: bool,
    *,
    space: str = "base",
    subspace: dict[str, list] | None = None,
    hipblaslt_kernel: str | None = None,
):
    """Generate and validate the shape-aware policy catalog used by tuning.

    space:
      - ``base``: original selection lists (mma-iter cap 4, large-GEMM
        ``256x256 w2x4x1 pht`` only).
      - ``extended``: base union all ``EXTENDED_VALUES`` (mma-iter cap 12,
        large-GEMM also ``w2x2x1 pht`` and ``256x224 pft``).
      - ``subspace``: base catalog plus a hipBLASLt neighborhood (the
        mapped MT/waves, not the full extra cartesian). Pass ``subspace``
        and/or ``hipblaslt_kernel``.
    """

    if make_gemm_a16w16_param_and_validate is None:
        return []
    if get_gfx() != "gfx950":
        return []
    if dtype not in (torch.float16, torch.bfloat16):
        return []
    if out_dtype not in (dtype, torch.float32):
        return []
    if space not in FLYDSL_A16W16_SPACES:
        raise ValueError(f"space must be one of {FLYDSL_A16W16_SPACES}, got {space!r}")
    if space == "subspace" and not subspace and not hipblaslt_kernel:
        raise ValueError("space='subspace' requires subspace=... or hipblaslt_kernel=...")

    parsed = parse_hipblaslt_kernel(hipblaslt_kernel) if hipblaslt_kernel else {}
    if space == "extended":
        extra = dict(EXTENDED_VALUES)
        selections = _build_selections(k, extra)
    elif space == "subspace":
        extra = _merge_extra_maps(
            subspace,
            subspace_from_hipblaslt_kernel(hipblaslt_kernel) if hipblaslt_kernel else None,
        )
        selections = _build_selections(k, extra={})
    else:
        extra = {}
        selections = _build_selections(k, extra)

    configs = [
        dict(zip(selections, combo))
        for combo in itertools.product(*selections.values())
    ]
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    configs = GemmConfigPruner(
        m,
        n,
        k,
        device_props,
        2,
    ).prune(configs)

    if space == "subspace":
        overrides = _neighborhood_overrides(parsed, extra)
        if overrides:
            neigh_sel = _build_selections(k, extra={})
            for name, values in overrides.items():
                if name in neigh_sel:
                    neigh_sel[name] = list(values)
            existing = {tuple(sorted(config.items())) for config in configs}
            for combo in itertools.product(*neigh_sel.values()):
                seed = dict(zip(neigh_sel, combo))
                key = tuple(sorted(seed.items()))
                if key not in existing:
                    configs.append(seed)
                    existing.add(key)

    is_large_gemm = m >= 4096 and n >= 4096 and k >= 4096
    allow_w22 = space == "extended" or parsed.get("n_waves") == 2
    allow_224 = space == "extended" or 224 in extra.get("block_n", ())
    if is_large_gemm and allow_224:
        existing = {tuple(sorted(config.items())) for config in configs}
        for group_m in (0, 4):
            seed = {
                "block_m": 256,
                "block_n": 224,
                "block_k": 64,
                "stages": 2,
                "split_k": 1,
                "m_waves": 2,
                "n_waves": 2,
                "k_waves": 1,
                "group_m": group_m,
                "use_half_tile_interleaved": False,
            }
            if tuple(sorted(seed.items())) not in existing:
                configs.append(seed)

    in_dtype_id = (
        GEMM_A16W16_DTYPE_FP16 if dtype == torch.float16 else GEMM_A16W16_DTYPE_BF16
    )
    out_dtype_id = GEMM_A16W16_DTYPE_FP32 if out_dtype == torch.float32 else in_dtype_id
    valid_configs = []
    for config in configs:
        if _is_known_llvm_abort(config):
            continue
        if is_large_gemm:
            keep = _is_256x256_pht(config, 4)
            if allow_w22:
                keep = keep or _is_256x256_pht(config, 2)
            if allow_224:
                keep = keep or _is_256x224_pft(config)
            if not keep:
                continue
        elif not config["use_half_tile_interleaved"]:
            mma_m_iters = config["block_m"] // config["m_waves"] // 16
            mma_n_iters = config["block_n"] // config["n_waves"] // 16
            mma_cap = (
                _EXTENDED_MMA_ITER_CAP
                if space == "extended" or _config_uses_extended(config)
                else _BASE_MMA_ITER_CAP
            )
            if mma_m_iters > mma_cap or mma_n_iters > mma_cap:
                continue

        validation_config = {
            **config,
            "in_dtype_id": in_dtype_id,
            "out_dtype_id": out_dtype_id,
            "a_is_transposed": False,
            "b_is_transposed": True,
            "has_bias": has_bias,
        }
        if (
            make_gemm_a16w16_param_and_validate(
                m,
                n,
                k,
                validation_config,
            )
            is not None
        ):
            valid_configs.append(config)
    return valid_configs
