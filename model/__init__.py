"""Legacy import path for the new API; historical constructor weights differ."""

from dlwa_csi.models import PriorInformedUNet3D

UNet = PriorInformedUNet3D

__all__ = ["PriorInformedUNet3D", "UNet"]
