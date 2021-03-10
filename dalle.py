# Modified MIT License
# 
# Software Copyright (c) 2021 OpenAI
# 
# We don’t claim ownership of the content you create with the DALL-E discrete VAE, so it is yours to
# do with as you please. We only ask that you use the model responsibly and clearly indicate that it
# was used.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
#
# single unfied file for dalle discrete VAE
# why pip when you can wget
#
# original repo: https://github.com/openai/DALL-E
# credits: @adityaramesh

import attr
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import io
from collections  import OrderedDict
from functools    import partial

logit_laplace_eps: float = 0.1

# ------ Convolution function

@attr.s(eq=False)
class Conv2d(nn.Module):
  n_in:  int = attr.ib(validator=lambda i, a, x: x >= 1)
  n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
  kw:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)

  use_float16:   bool         = attr.ib(default=True)
  device:        torch.device = attr.ib(default=torch.device('cpu'))
  requires_grad: bool         = attr.ib(default=False)

  def __attrs_post_init__(self) -> None:
    super().__init__()
    size = (self.n_out, self.n_in, self.kw, self.kw)
    w = torch.empty(size=size, dtype=torch.float32, device=self.device)
    w.normal_(std=1 / math.sqrt(self.n_in * self.kw ** 2))
    
    # move requires_grad after filling values using normal_
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    w.requires_grad = self.requires_grad

    b = torch.zeros((self.n_out,), dtype=torch.float32, device=self.device,
      requires_grad=self.requires_grad)
    self.w, self.b = nn.Parameter(w), nn.Parameter(b)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.use_float16 and 'cuda' in self.w.device.type:
      if x.dtype != torch.float16:
        x = x.half()

      w, b = self.w.half(), self.b.half()
    else:
      if x.dtype != torch.float32:
        x = x.float()

      w, b = self.w, self.b

    return F.conv2d(x, w, b, padding=(self.kw - 1) // 2)

# ------ Decoder block

@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Module):
  n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
  n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
  n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

  device:        torch.device = attr.ib(default=None)
  requires_grad: bool         = attr.ib(default=False)

  def __attrs_post_init__(self) -> None:
    super().__init__()
    self.n_hid = self.n_out // 4
    self.post_gain = 1 / (self.n_layers ** 2)

    make_conv     = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
    self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
    self.res_path = nn.Sequential(OrderedDict([
        ('relu_1', nn.ReLU()),
        ('conv_1', make_conv(self.n_in,  self.n_hid, 1)),
        ('relu_2', nn.ReLU()),
        ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
        ('relu_3', nn.ReLU()),
        ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
        ('relu_4', nn.ReLU()),
        ('conv_4', make_conv(self.n_hid, self.n_out, 3)),]))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.id_path(x) + self.post_gain * self.res_path(x)

@attr.s(eq=False, repr=False)
class Decoder(nn.Module):
  group_count:     int = 4
  n_init:          int = attr.ib(default=128,  validator=lambda i, a, x: x >= 8)
  n_hid:           int = attr.ib(default=256,  validator=lambda i, a, x: x >= 64)
  n_blk_per_group: int = attr.ib(default=2,    validator=lambda i, a, x: x >= 1)
  output_channels: int = attr.ib(default=3,    validator=lambda i, a, x: x >= 1)
  vocab_size:      int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

  device:              torch.device = attr.ib(default=torch.device('cpu'))
  requires_grad:       bool         = attr.ib(default=False)
  use_mixed_precision: bool         = attr.ib(default=True)

  def __attrs_post_init__(self) -> None:
    super().__init__()

    blk_range  = range(self.n_blk_per_group)
    n_layers   = self.group_count * self.n_blk_per_group
    make_conv  = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
    make_blk   = partial(DecoderBlock, n_layers=n_layers, device=self.device,
        requires_grad=self.requires_grad)

    self.blocks = nn.Sequential(OrderedDict([
      ('input', make_conv(self.vocab_size, self.n_init, 1, use_float16=False)),
      ('group_1', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(self.n_init if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
        ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
      ]))),
      ('group_2', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(8 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
        ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
      ]))),
      ('group_3', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
        ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
      ]))),
      ('group_4', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
      ]))),
      ('output', nn.Sequential(OrderedDict([
        ('relu', nn.ReLU()),
        ('conv', make_conv(1 * self.n_hid, 2 * self.output_channels, 1)),
      ]))),
    ]))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) != 4:
      raise ValueError(f'input shape {x.shape} is not 4d')
    if x.shape[1] != self.vocab_size:
      raise ValueError(f'input has {x.shape[1]} channels but model built for {self.vocab_size}')
    if x.dtype != torch.float32:
      raise ValueError('input must have dtype torch.float32')

    return self.blocks(x)

# ------ Encoder block

@attr.s(eq=False, repr=False)
class EncoderBlock(nn.Module):
  n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
  n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
  n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

  device:        torch.device = attr.ib(default=None)
  requires_grad: bool         = attr.ib(default=False)

  def __attrs_post_init__(self) -> None:
    super().__init__()
    self.n_hid = self.n_out // 4
    self.post_gain = 1 / (self.n_layers ** 2)

    make_conv     = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
    self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
    self.res_path = nn.Sequential(OrderedDict([
        ('relu_1', nn.ReLU()),
        ('conv_1', make_conv(self.n_in,  self.n_hid, 3)),
        ('relu_2', nn.ReLU()),
        ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
        ('relu_3', nn.ReLU()),
        ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
        ('relu_4', nn.ReLU()),
        ('conv_4', make_conv(self.n_hid, self.n_out, 1)),]))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.id_path(x) + self.post_gain * self.res_path(x)

@attr.s(eq=False, repr=False)
class Encoder(nn.Module):
  group_count:     int = 4
  n_hid:           int = attr.ib(default=256,  validator=lambda i, a, x: x >= 64)
  n_blk_per_group: int = attr.ib(default=2,    validator=lambda i, a, x: x >= 1)
  input_channels:  int = attr.ib(default=3,    validator=lambda i, a, x: x >= 1)
  vocab_size:      int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

  device:              torch.device = attr.ib(default=torch.device('cpu'))
  requires_grad:       bool         = attr.ib(default=False)
  use_mixed_precision: bool         = attr.ib(default=True)

  def __attrs_post_init__(self) -> None:
    super().__init__()

    blk_range  = range(self.n_blk_per_group)
    n_layers   = self.group_count * self.n_blk_per_group
    make_conv  = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
    make_blk   = partial(EncoderBlock, n_layers=n_layers, device=self.device,
        requires_grad=self.requires_grad)

    self.blocks = nn.Sequential(OrderedDict([
      ('input', make_conv(self.input_channels, 1 * self.n_hid, 7)),
      ('group_1', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
        ('pool', nn.MaxPool2d(kernel_size=2)),
      ]))),
      ('group_2', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(1 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
        ('pool', nn.MaxPool2d(kernel_size=2)),
      ]))),
      ('group_3', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
        ('pool', nn.MaxPool2d(kernel_size=2)),
      ]))),
      ('group_4', nn.Sequential(OrderedDict([
        *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
      ]))),
      ('output', nn.Sequential(OrderedDict([
        ('relu', nn.ReLU()),
        ('conv', make_conv(8 * self.n_hid, self.vocab_size, 1, use_float16=False)),
      ]))),
    ]))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) != 4:
      raise ValueError(f'input shape {x.shape} is not 4d')
    if x.shape[1] != self.input_channels:
      raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
    if x.dtype != torch.float32:
      raise ValueError('input must have dtype torch.float32')

    return self.blocks(x)

# ------ helper functions

def map_pixels(x: torch.Tensor) -> torch.Tensor:
  if len(x.shape) != 4:
    raise ValueError('expected input to be 4d')
  if x.dtype != torch.float:
    raise ValueError('expected input to have type float')

  return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def unmap_pixels(x: torch.Tensor) -> torch.Tensor:
  if len(x.shape) != 4:
    raise ValueError('expected input to be 4d')
  if x.dtype != torch.float:
    raise ValueError('expected input to have type float')

  return torch.clamp((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)

def load_model(path: str, device: torch.device = None) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()
            
        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device)
