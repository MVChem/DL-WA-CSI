"""Legacy import-path alias for the new model API.

The historical Diffusers-derived constructor and raw checkpoints are not
compatible. New code should import
:class:`dlwa_csi.models.PriorInformedUNet3D` directly.
"""

from dlwa_csi.models import PriorInformedUNet3D

UNet = PriorInformedUNet3D

__all__ = ["PriorInformedUNet3D", "UNet"]
