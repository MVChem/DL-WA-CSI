"""Neural reconstruction models for DL-WA-CSI.

The public model in this module follows the data layout used by the paper:
dynamic frames form a sequence and FID/spectral samples form channels.  The
implementation deliberately depends only on PyTorch so it can be trained or
deployed without the diffusion-model stack used by the original prototype.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["PriorInformedUNet3D"]


def _normalization_groups(channels: int, requested_groups: int) -> int:
    """Choose a valid GroupNorm count with at least two channels per group."""

    upper_bound = min(requested_groups, max(1, channels // 2))
    for groups in range(upper_bound, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _as_positive_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        pair = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pair = tuple(value)
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly two integers")
    else:
        raise TypeError(f"{name} must be an integer or a pair of integers")

    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in pair):
        raise ValueError(f"{name} values must be positive integers")
    return pair


def _expand_heads(value: int | Sequence[int], levels: int) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        heads = (value,) * levels
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        heads = tuple(value)
        if len(heads) != levels:
            raise ValueError(
                "attention_heads must be an integer or contain one value per U-Net level"
            )
    else:
        raise TypeError("attention_heads must be an integer or a sequence of integers")

    if any(not isinstance(head, int) or isinstance(head, bool) or head <= 0 for head in heads):
        raise ValueError("attention head counts must be positive integers")
    return heads


class _ResidualBlock3D(nn.Module):
    """Pre-activation residual block that preserves temporal and spatial size."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            _normalization_groups(in_channels, norm_groups), in_channels
        )
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            _normalization_groups(out_channels, norm_groups), out_channels
        )
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.skip(inputs)
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + residual


class _ResidualBlock2D(nn.Module):
    """2D residual block used by the multiscale anatomical encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            _normalization_groups(in_channels, norm_groups), in_channels
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            _normalization_groups(out_channels, norm_groups), out_channels
        )
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.skip(inputs)
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + residual


class _SpectralChannelAttention(nn.Module):
    """Early frequency-channel attention with residual propagation.

    Pooling over dynamic and spatial axes leaves one descriptor per FID
    channel.  A small MLP learns channel gates and the gated signal is added to
    the original input, matching the residual FCA use described in the paper.
    """

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        descriptors = inputs.mean(dim=(2, 3, 4))
        weights = self.gate(descriptors).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)
        return inputs + inputs * weights


class _AnatomyEncoder(nn.Module):
    """Create one co-registered anatomy feature map per U-Net resolution."""

    def __init__(
        self,
        channels: tuple[int, ...],
        blocks_per_stage: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.downsamplers = nn.ModuleList(
            nn.Conv2d(
                channels[level],
                channels[level + 1],
                kernel_size=3,
                stride=2,
                padding=1,
            )
            for level in range(len(channels) - 1)
        )
        self.stages = nn.ModuleList()
        for stage_channels in channels:
            self.stages.append(
                nn.Sequential(
                    *(
                        _ResidualBlock2D(
                            stage_channels,
                            stage_channels,
                            norm_groups,
                            dropout,
                        )
                        for _ in range(blocks_per_stage)
                    )
                )
            )

    def forward(self, anatomy: Tensor) -> list[Tensor]:
        hidden = self.stem(anatomy)
        features: list[Tensor] = []
        for level, stage in enumerate(self.stages):
            if level:
                hidden = self.downsamplers[level - 1](hidden)
            hidden = stage(hidden)
            features.append(hidden)
        return features


class _EfficientAnatomicalCrossAttention(nn.Module):
    """Cross-attend DMI queries to bounded anatomical key/value tokens.

    Anatomical tokens are adaptively pooled and DMI queries are processed in
    chunks, so peak attention memory is bounded by ``query_chunk_size`` times
    the configured anatomy token count rather than the full 3D volume.  A
    separately projected, spatially aligned anatomical residual is also added;
    this preserves exact local structure that token-only attention could lose
    through permutation-invariant pooling.
    """

    def __init__(
        self,
        dmi_channels: int,
        anatomy_channels: int,
        num_heads: int,
        pool_size: tuple[int, int],
        query_chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.query_chunk_size = query_chunk_size
        self.query_norm = nn.LayerNorm(dmi_channels)
        self.context_norm = nn.LayerNorm(anatomy_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=dmi_channels,
            num_heads=num_heads,
            dropout=dropout,
            kdim=anatomy_channels,
            vdim=anatomy_channels,
            batch_first=True,
        )
        self.attention_projection = nn.Conv3d(dmi_channels, dmi_channels, kernel_size=1)
        self.aligned_projection = nn.Conv2d(
            anatomy_channels, dmi_channels, kernel_size=1
        )
        self.attention_gain = nn.Parameter(torch.ones(()))
        self.aligned_gain = nn.Parameter(torch.ones(()))

    def forward(self, dmi: Tensor, anatomy: Tensor) -> Tensor:
        batch, channels, frames, height, width = dmi.shape
        pooled_height = min(height, self.pool_size[0])
        pooled_width = min(width, self.pool_size[1])
        pooled_anatomy = F.adaptive_avg_pool2d(
            anatomy, output_size=(pooled_height, pooled_width)
        )

        queries = dmi.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        queries = self.query_norm(queries)
        context = pooled_anatomy.flatten(2).transpose(1, 2)
        context = self.context_norm(context)

        attended_chunks: list[Tensor] = []
        for start in range(0, queries.shape[1], self.query_chunk_size):
            query_chunk = queries[:, start : start + self.query_chunk_size]
            attended, _ = self.attention(
                query_chunk,
                context,
                context,
                need_weights=False,
            )
            attended_chunks.append(attended)
        attended = torch.cat(attended_chunks, dim=1)
        attended = attended.reshape(batch, frames, height, width, channels)
        attended = attended.permute(0, 4, 1, 2, 3).contiguous()
        attended = self.attention_projection(attended)

        if anatomy.shape[-2:] != (height, width):
            anatomy = F.interpolate(
                anatomy, size=(height, width), mode="bilinear", align_corners=False
            )
        aligned = self.aligned_projection(anatomy).unsqueeze(2)
        return dmi + self.attention_gain * attended + self.aligned_gain * aligned


def _sinusoidal_positions(length: int, channels: int, reference: Tensor) -> Tensor:
    """Create an arbitrary-length temporal positional encoding."""

    positions = torch.arange(length, device=reference.device, dtype=torch.float32).unsqueeze(1)
    even_channels = torch.arange(
        0, channels, 2, device=reference.device, dtype=torch.float32
    )
    frequencies = torch.exp(-math.log(10_000.0) * even_channels / max(channels, 1))
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(
        length, channels, device=reference.device, dtype=torch.float32
    )
    encoding[:, 0::2] = torch.sin(angles)
    if channels > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=reference.dtype)


class _TemporalTransformer(nn.Module):
    """Model the dynamic-frame axis as a sequence at every spatial location."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=max(channels, int(round(channels * mlp_ratio))),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(channels),
            enable_nested_tensor=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, frames, height, width = inputs.shape
        sequences = inputs.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, frames, channels
        )
        positions = _sinusoidal_positions(frames, channels, sequences).unsqueeze(0)
        sequences = self.encoder(sequences + positions)
        return sequences.reshape(batch, height, width, frames, channels).permute(
            0, 4, 3, 1, 2
        ).contiguous()


class PriorInformedUNet3D(nn.Module):
    """Anatomy- and spectral-prior-informed 3D U-Net for dynamic DMI.

    Parameters
    ----------
    in_channels:
        Number of FID/spectral channels.  The paper uses 72.
    channels:
        Feature widths for successive spatial U-Net levels.  Its length sets
        the number of levels; the default is a four-level implementation choice
        aligned to the manuscript schematic, while
        short tuples such as ``(8, 16)`` are useful for CPU tests.
    blocks_per_stage:
        Residual blocks in every DMI and anatomy encoder stage.  Decoder stages
        contain the same number, including their skip-fusion block.
    attention_heads:
        Cross-attention heads, either shared or specified once per level.
    attention_pool_size:
        Maximum height and width of anatomical K/V token grids.
    query_chunk_size:
        Maximum DMI query tokens materialized in any attention operation.
    temporal_heads, temporal_layers, temporal_mlp_ratio:
        Bottleneck temporal Transformer configuration.
    residual_output:
        Learn a correction to the degraded DMI input when true (the default).

    Notes
    -----
    ``forward`` accepts DMI data shaped ``[B, T, C, H, W]`` and anatomy shaped
    ``[B, H, W]`` or ``[B, 1, H, W]``.  It always returns the original DMI
    shape.  Spatial downsampling uses stride ``(1, 2, 2)`` and therefore never
    reduces the dynamic-frame axis.
    """

    config_name = "model_config.json"

    def __init__(
        self,
        in_channels: int = 72,
        channels: Sequence[int] = (72, 72, 72, 72),
        blocks_per_stage: int = 2,
        attention_heads: int | Sequence[int] = 4,
        attention_pool_size: int | Sequence[int] = 8,
        query_chunk_size: int = 1024,
        temporal_heads: int = 8,
        temporal_layers: int = 2,
        temporal_mlp_ratio: float = 4.0,
        spectral_reduction: int = 8,
        norm_groups: int = 8,
        dropout: float = 0.0,
        residual_output: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
            raise TypeError("channels must be a sequence of integers")
        feature_channels = tuple(channels)
        if not feature_channels:
            raise ValueError("channels must define at least one U-Net level")
        if any(
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or channel < 2
            for channel in feature_channels
        ):
            raise ValueError("all feature channel widths must be integers of at least two")
        if (
            not isinstance(blocks_per_stage, int)
            or isinstance(blocks_per_stage, bool)
            or blocks_per_stage <= 0
        ):
            raise ValueError("blocks_per_stage must be a positive integer")
        if (
            not isinstance(query_chunk_size, int)
            or isinstance(query_chunk_size, bool)
            or query_chunk_size <= 0
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        if (
            not isinstance(temporal_heads, int)
            or isinstance(temporal_heads, bool)
            or temporal_heads <= 0
        ):
            raise ValueError("temporal_heads must be a positive integer")
        if (
            not isinstance(temporal_layers, int)
            or isinstance(temporal_layers, bool)
            or temporal_layers <= 0
        ):
            raise ValueError("temporal_layers must be a positive integer")
        if not isinstance(temporal_mlp_ratio, (int, float)) or temporal_mlp_ratio <= 0:
            raise ValueError("temporal_mlp_ratio must be positive")
        if (
            not isinstance(spectral_reduction, int)
            or isinstance(spectral_reduction, bool)
            or spectral_reduction <= 0
        ):
            raise ValueError("spectral_reduction must be a positive integer")
        if (
            not isinstance(norm_groups, int)
            or isinstance(norm_groups, bool)
            or norm_groups <= 0
        ):
            raise ValueError("norm_groups must be a positive integer")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(residual_output, bool):
            raise TypeError("residual_output must be a boolean")

        pool_size = _as_positive_pair(attention_pool_size, "attention_pool_size")
        cross_attention_heads = _expand_heads(attention_heads, len(feature_channels))
        for level, (width, heads) in enumerate(
            zip(feature_channels, cross_attention_heads, strict=True)
        ):
            if width % heads:
                raise ValueError(
                    f"channels[{level}] ({width}) must be divisible by "
                    f"attention_heads[{level}] ({heads})"
                )
        if feature_channels[-1] % temporal_heads:
            raise ValueError(
                "the bottleneck channel width must be divisible by temporal_heads"
            )

        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.residual_output = residual_output
        self._minimum_spatial_size = 2 ** (len(feature_channels) - 1)
        self._config: dict[str, Any] = {
            "in_channels": in_channels,
            "channels": list(feature_channels),
            "blocks_per_stage": blocks_per_stage,
            "attention_heads": list(cross_attention_heads),
            "attention_pool_size": list(pool_size),
            "query_chunk_size": query_chunk_size,
            "temporal_heads": temporal_heads,
            "temporal_layers": temporal_layers,
            "temporal_mlp_ratio": float(temporal_mlp_ratio),
            "spectral_reduction": spectral_reduction,
            "norm_groups": norm_groups,
            "dropout": float(dropout),
            "residual_output": residual_output,
        }

        self.spectral_attention = _SpectralChannelAttention(
            in_channels, spectral_reduction
        )
        self.input_projection = nn.Conv3d(
            in_channels, feature_channels[0], kernel_size=3, padding=1
        )
        self.anatomy_encoder = _AnatomyEncoder(
            feature_channels,
            blocks_per_stage,
            norm_groups,
            float(dropout),
        )

        self.encoder_stages = nn.ModuleList()
        self.encoder_attention = nn.ModuleList()
        for width, heads in zip(feature_channels, cross_attention_heads, strict=True):
            self.encoder_stages.append(
                nn.Sequential(
                    *(
                        _ResidualBlock3D(width, width, norm_groups, float(dropout))
                        for _ in range(blocks_per_stage)
                    )
                )
            )
            self.encoder_attention.append(
                _EfficientAnatomicalCrossAttention(
                    width,
                    width,
                    heads,
                    pool_size,
                    query_chunk_size,
                    float(dropout),
                )
            )

        self.downsamplers = nn.ModuleList(
            nn.Conv3d(
                feature_channels[level],
                feature_channels[level + 1],
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            )
            for level in range(len(feature_channels) - 1)
        )

        bottleneck_width = feature_channels[-1]
        self.bottleneck_before = _ResidualBlock3D(
            bottleneck_width, bottleneck_width, norm_groups, float(dropout)
        )
        self.temporal_transformer = _TemporalTransformer(
            bottleneck_width,
            temporal_heads,
            temporal_layers,
            float(temporal_mlp_ratio),
            float(dropout),
        )
        self.bottleneck_after = _ResidualBlock3D(
            bottleneck_width, bottleneck_width, norm_groups, float(dropout)
        )

        self.up_projections = nn.ModuleList(
            nn.Conv3d(
                feature_channels[level + 1], feature_channels[level], kernel_size=1
            )
            for level in range(len(feature_channels) - 1)
        )
        self.decoder_stages = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        for width, heads in zip(feature_channels, cross_attention_heads, strict=True):
            decoder_blocks: list[nn.Module] = [
                _ResidualBlock3D(width * 2, width, norm_groups, float(dropout))
            ]
            decoder_blocks.extend(
                _ResidualBlock3D(width, width, norm_groups, float(dropout))
                for _ in range(blocks_per_stage - 1)
            )
            self.decoder_stages.append(nn.Sequential(*decoder_blocks))
            self.decoder_attention.append(
                _EfficientAnatomicalCrossAttention(
                    width,
                    width,
                    heads,
                    pool_size,
                    query_chunk_size,
                    float(dropout),
                )
            )

        self.output_norm = nn.GroupNorm(
            _normalization_groups(feature_channels[0], norm_groups),
            feature_channels[0],
        )
        self.output_projection = nn.Conv3d(
            feature_channels[0], in_channels, kernel_size=3, padding=1
        )

    @property
    def config(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the construction config."""

        return self.get_config()

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the construction config."""

        # JSON round-tripping gives callers a deep copy without another public
        # dependency and guarantees the returned object remains serializable.
        return json.loads(json.dumps(self._config))

    def save_config(self, path: str | Path) -> Path:
        """Save construction parameters and return the written JSON path.

        ``path`` may be either a JSON filename or a directory, in which case
        :attr:`config_name` is created inside it.
        """

        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination / self.config_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.get_config(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | str | Path
    ) -> PriorInformedUNet3D:
        """Construct a model from a mapping, JSON file, or config directory."""

        if isinstance(config, (str, Path)):
            source = Path(config)
            if source.is_dir() or source.suffix.lower() != ".json":
                source = source / cls.config_name
            loaded = json.loads(source.read_text(encoding="utf-8"))
        elif isinstance(config, Mapping):
            loaded = dict(config)
        else:
            raise TypeError("config must be a mapping, JSON path, or config directory")
        return cls(**loaded)

    def _validate_inputs(self, dmi: Tensor, anatomy: Tensor) -> Tensor:
        if not isinstance(dmi, Tensor):
            raise TypeError("dmi must be a torch.Tensor")
        if not isinstance(anatomy, Tensor):
            raise TypeError("anatomy must be a torch.Tensor")
        if dmi.ndim != 5:
            raise ValueError(
                f"dmi must have shape [B, T, C, H, W], received {tuple(dmi.shape)}"
            )
        if anatomy.ndim == 3:
            anatomy = anatomy.unsqueeze(1)
        elif anatomy.ndim != 4:
            raise ValueError(
                "anatomy must have shape [B, H, W] or [B, 1, H, W], "
                f"received {tuple(anatomy.shape)}"
            )
        if anatomy.shape[1] != 1:
            raise ValueError(
                f"anatomy must have exactly one channel, received {anatomy.shape[1]}"
            )

        batch, frames, spectral_channels, height, width = dmi.shape
        if spectral_channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} DMI spectral channels, "
                f"received {spectral_channels}"
            )
        if batch <= 0 or frames <= 0:
            raise ValueError("DMI batch and dynamic-frame dimensions must be non-empty")
        if height < self._minimum_spatial_size or width < self._minimum_spatial_size:
            raise ValueError(
                f"DMI spatial dimensions must each be at least "
                f"{self._minimum_spatial_size} for {len(self.feature_channels)} levels"
            )
        if anatomy.shape[0] != batch:
            raise ValueError("DMI and anatomy batch sizes must match")
        anatomy_too_small = (
            anatomy.shape[-2] < self._minimum_spatial_size
            or anatomy.shape[-1] < self._minimum_spatial_size
        )
        if anatomy_too_small:
            raise ValueError(
                "anatomy spatial dimensions must support every encoder level"
            )
        if not dmi.is_floating_point() or not anatomy.is_floating_point():
            raise TypeError("dmi and anatomy must use floating-point dtypes")
        if anatomy.device != dmi.device:
            raise ValueError("dmi and anatomy must be on the same device")
        if anatomy.dtype != dmi.dtype:
            raise ValueError("dmi and anatomy must have the same dtype")
        return anatomy

    def forward(self, dmi: Tensor, anatomy: Tensor) -> Tensor:
        anatomy = self._validate_inputs(dmi, anatomy)

        # Conv3d uses [B, C, D, H, W]; dynamic time is the depth/sequence axis.
        input_residual = dmi.permute(0, 2, 1, 3, 4).contiguous()
        hidden = self.input_projection(self.spectral_attention(input_residual))
        anatomy_features = self.anatomy_encoder(anatomy)

        skips: list[Tensor] = []
        for level, (stage, cross_attention) in enumerate(
            zip(self.encoder_stages, self.encoder_attention, strict=True)
        ):
            hidden = stage(hidden)
            hidden = cross_attention(hidden, anatomy_features[level])
            skips.append(hidden)
            if level < len(self.downsamplers):
                hidden = self.downsamplers[level](hidden)

        hidden = self.bottleneck_before(hidden)
        hidden = self.temporal_transformer(hidden)
        hidden = self.bottleneck_after(hidden)

        for level in reversed(range(len(self.feature_channels))):
            skip = skips[level]
            if level < len(self.feature_channels) - 1:
                hidden = F.interpolate(
                    hidden,
                    size=(hidden.shape[2], skip.shape[-2], skip.shape[-1]),
                    mode="trilinear",
                    align_corners=False,
                )
                hidden = self.up_projections[level](hidden)
            hidden = torch.cat((hidden, skip), dim=1)
            hidden = self.decoder_stages[level](hidden)
            hidden = self.decoder_attention[level](hidden, anatomy_features[level])

        reconstruction = self.output_projection(F.silu(self.output_norm(hidden)))
        if self.residual_output:
            reconstruction = reconstruction + input_residual
        return reconstruction.permute(0, 2, 1, 3, 4).contiguous()
