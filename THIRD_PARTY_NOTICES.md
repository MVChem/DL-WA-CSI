# Third-party notices

This file records third-party source provenance in the repository's Git
history. It is not a license for the original DL-WA-CSI code. No project-wide
license has been declared; only the copyright holders can choose one.

## Hugging Face Diffusers

Historical revisions of `model/unet.py` and `model/fca_unet.py` were adapted
from Hugging Face Diffusers' conditional 3D U-Net and copied several methods
from its conditional 2D U-Net. Those historical files have been replaced by
the current implementation, but remain reachable through Git history.

- Upstream: <https://github.com/huggingface/diffusers>
- Source lineage verified against Diffusers v0.34.0:
  [`unet_3d_condition.py`](https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/models/unets/unet_3d_condition.py)
  and
  [`unet_2d_condition.py`](https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/models/unets/unet_2d_condition.py)
- License: Apache License 2.0
- Copyright notice in the upstream 3D U-Net: Copyright 2025 Alibaba DAMO-VILAB
  and The HuggingFace Team; Copyright 2025 The ModelScope Team.
- Copyright notice in the upstream 2D U-Net: Copyright 2025 The HuggingFace
  Team.

A copy of the Apache License 2.0 is distributed at
[`third_party_licenses/Apache-2.0.txt`](third_party_licenses/Apache-2.0.txt).
The verified upstream copy is at
<https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/LICENSE>.
The historical adaptations were modified for DL-WA-CSI; they are not endorsed
by the upstream authors.

## FcaNet

Historical revision `model/fca.py` was adapted from FcaNet's
`model/layer.py`. The local revision changed the accepted frequency-count
names; the rest of that historical file matches the upstream implementation.
The file has been removed from the current tree but remains reachable through
Git history.

- Upstream: <https://github.com/cfzd/FcaNet>
- Source verified against
  [`model/layer.py`](https://github.com/cfzd/FcaNet/blob/aa5fb63505575bb4e4e094613565379c3f6ada33/model/layer.py)
- License: MIT
- Copyright (c) 2021 cfzd

The MIT license requires the following notice to accompany copies or
substantial portions of that software:

> MIT License
>
> Copyright (c) 2021 cfzd
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The verified upstream license is available at
<https://github.com/cfzd/FcaNet/blob/aa5fb63505575bb4e4e094613565379c3f6ada33/LICENSE>.
