"""Legacy import-path alias; the historical constructor/checkpoints differ."""

from dlwa_csi.models import PriorInformedUNet3D

UNet = PriorInformedUNet3D

__all__ = ["PriorInformedUNet3D", "UNet"]
