import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation as A
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single


class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  def forward(self, input):
    return input


class SlidingWindowMeanNorm(nn.Module):
  def __init__(self, swLen, padtype="reflect"):
    """ Input  must be (N, C, W_in) 
        Output must be (N, C, W_in_padded)
        (The last dim is assumed to be the time-frame axis)
    """
    super(SlidingWindowMeanNorm, self).__init__()
    padLen = int(swLen/2)
    self.padder = nn.ReflectionPad1d((padLen, padLen-1))
    self.swLen = swLen
    self.time_axis = -1

  def forward(self, x):
    if x.size(self.time_axis) > self.swLen:
      swMean = F.avg_pool1d(self.padder(x), 
                            kernel_size=self.swLen, 
                            stride=1)
    else:
      swMean = torch.mean(x, dim=self.time_axis, keepdim=True)
    return x - swMean

class Tdnn(torch.nn.modules.conv._ConvNd):
  def __init__(self, in_channels, out_channels, splice, stride=1,
               padding=0, bias=True):

    ## Weight mask options
    bound = max(abs(splice[0]), splice[-1])
    n_range = 2*bound + 1

    kernel_size = _single(n_range)
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(1)
    groups = 1
    padding_mode='zeros'
    super(Tdnn, self).__init__(
          in_channels, out_channels, kernel_size, stride, padding, dilation,
          False, _single(0), groups, bias, padding_mode)

    ## Weight mask
    tt_idx = [tt + bound for tt in splice]
    self.w_mask = torch.zeros(out_channels, in_channels, n_range).float()
    self.w_mask[..., tt_idx] = 1.0
    # self.register_buffer('w_mask', self.w_mask)
    self.w_mask = torch.nn.Parameter(self.w_mask, requires_grad=False)

  def forward(self, input):
    if self.padding_mode == 'circular':
        expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
        return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                        self.w_mask*self.weight, self.bias, self.stride,
                        _single(0), self.dilation, self.groups)
    return F.conv1d(input, self.w_mask*self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
  

class Dense(nn.Module):
  def __init__(self, in_dim, out_dim, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-06, 
               activ=None, kwargs={}, w_init_gain='linear', 
               as_class=False):
    super(Dense, self).__init__()
    self.activ, self.weights = [], []

    ## Affine transform
    affine = nn.Linear(in_dim, out_dim, bias=use_bias)
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    nn.init.xavier_normal_(affine.weight, gain=w_init_gain)
    if use_bias:
      nn.init.zeros_(affine.bias)
    self.affine = affine
    self.activ.append(affine)
    self.weights.append(affine.weight)

    ## Batch normalization
    if apply_bn:
      batch_norm = nn.BatchNorm1d(
        out_dim, eps=bn_eps, momentum=bn_mom, 
        affine=True, track_running_stats=True)
      self.bn = nn.Sequential(affine, batch_norm)
      self.activ.append(batch_norm)

    ## Activation function
    if activ is not None:
      activ_func = getattr(A, activ)(**kwargs)
      self.activ.append(activ_func)
    self.activ = nn.Sequential(*self.activ)

    if as_class:
      self.weight, self.bias = affine.weight, affine.bias
      if apply_bn:
        self.bn_gamma, self.bn_beta = batch_norm.weight, batch_norm.bias
        self.bn_mean, self.bn_var = batch_norm.running_mean, batch_norm.running_var
        self.bn_eps = batch_norm.eps

  def forward(self, x, drop=0.0):
    x = self.activ(x)
    if drop:
      x = F.dropout(x, drop, self.training)
    return x


class Tdense(nn.Module):
  def __init__(self, in_dim, out_dim, splice, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv1d', 
               as_class=False):
    super(Tdense, self).__init__()
    self.activ, self.weights = [], []

    bound = max(abs(splice[0]), splice[-1])
    n_range = 2*bound + 1
    padding = (n_range-1) // 2

    ## Convolution
    conv = Tdnn(
      in_dim, out_dim, splice=splice, stride=1, 
      padding=padding, bias=use_bias)
    # conv = nn.Conv1d(
    #   in_dim, out_dim, kernel_size=n_range, stride=1, 
    #   padding=padding, dilation=1, bias=use_bias)
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    nn.init.xavier_normal_(conv.weight, gain=w_init_gain)
    if use_bias:
      nn.init.zeros_(conv.bias)
    self.conv = conv
    self.activ.append(conv)
    self.weights.append(conv.weight)

    ## Batch normalization
    if apply_bn:
      batch_norm = nn.BatchNorm1d(
        out_dim, eps=bn_eps, momentum=bn_mom, 
        affine=True, track_running_stats=True)
      self.bn = nn.Sequential(conv, batch_norm)
      self.activ.append(batch_norm)

    ## Activation function
    if activ is not None:
      activ_func = getattr(A, activ)(**kwargs)
      self.activ.append(activ_func)
    self.activ = nn.Sequential(*self.activ)

    if as_class:
      self.weight, self.bias = conv.weight, conv.bias
      if apply_bn:
        self.bn_gamma, self.bn_beta = batch_norm.weight, batch_norm.bias
        self.bn_mean, self.bn_var = batch_norm.running_mean, batch_norm.running_var
        self.bn_eps = batch_norm.eps

  def forward(self, x, drop=0.0):
    x = self.activ(x)
    if drop:
      x = F.dropout(x, drop, self.training)
    return x


class GatedTdense(nn.Module):
  def __init__(self, in_dim, out_dim, splice, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv1d', 
               as_class=False):
    super(GatedTdense, self).__init__()
    self.activ, self.weights = [], []

    bound = max(abs(splice[0]), splice[-1])
    n_range = 2*bound + 1
    padding = (n_range-1) // 2

    ## Convolution
    conv = Tdnn(
      in_dim, in_dim+2*out_dim, splice=splice, stride=1, 
      padding=padding, bias=use_bias)
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    nn.init.xavier_normal_(conv.weight, gain=w_init_gain)
    if use_bias:
      nn.init.zeros_(conv.bias)
    self.conv = conv
    self.activ.append(conv)
    self.weights.append(conv.weight)

    ## Batch normalization
    if apply_bn:
      batch_norm = nn.BatchNorm1d(
        in_dim+2*out_dim, eps=bn_eps, momentum=bn_mom, 
        affine=True, track_running_stats=True)
      self.bn = nn.Sequential(conv, batch_norm)
      self.activ.append(batch_norm)

    ## Activatin function
    self.activ = nn.Sequential(*self.activ)

    ## Channel dimension resizer
    self.in_dim, self.out_dim = in_dim, out_dim
    self.resize = False if in_dim == out_dim else True
    if self.resize:
      self.resizer = Tdense(in_dim, out_dim, [0], use_bias=False, 
                            apply_bn=False, w_init_gain=w_init_gain, 
                            as_class=True)
      self.weights.append(self.resizer.weight)

    if as_class:
      self.weight, self.bias = conv.weight, conv.bias
      if apply_bn:
        self.bn_gamma, self.bn_beta = batch_norm.weight, batch_norm.bias
        self.bn_mean, self.bn_var = batch_norm.running_mean, batch_norm.running_var
        self.bn_eps = batch_norm.eps

  def forward(self, x, memory, drop=0.0):
    fog = self.activ(x)
    f = torch.sigmoid(fog[:,:self.in_dim,:])
    o = torch.sigmoid(fog[:,self.in_dim:self.in_dim+self.out_dim,:])
    g = torch.tanh(fog[:,self.in_dim+self.out_dim:,:])
    if drop:
      g = F.dropout(g, drop, self.training)
    memory = torch.mul(f, memory) + torch.mul(1.0-f, x)
    if self.resize:
      memory = self.resizer(memory)
    x = torch.mul(o, g) + memory
    return x, memory


class Conv1d(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=1, 
               padding='same', dilation=1, groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv1d', 
               as_class=False):
    super(Conv1d, self).__init__()
    self.activ, self.weights = [], []

    if isinstance(padding, str):
      padding = (kernel_size-1) * dilation // 2 if padding == 'same' else 0

    ## Convolution
    conv = nn.Conv1d(
      in_dim, out_dim, kernel_size, stride=stride, 
      padding=padding, dilation=dilation, groups=groups, 
      bias=use_bias)
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    nn.init.xavier_normal_(conv.weight, gain=w_init_gain)
    if use_bias:
      nn.init.zeros_(conv.bias)
    self.conv = conv
    self.activ.append(conv)
    self.weights.append(conv.weight)

    ## Batch normalization
    if apply_bn:
      batch_norm = nn.BatchNorm1d(
        out_dim, eps=bn_eps, momentum=bn_mom, 
        affine=True, track_running_stats=True)
      self.bn = nn.Sequential(conv, batch_norm)
      self.activ.append(batch_norm)

    ## Activation function
    if activ is not None:
      activ_func = getattr(A, activ)(**kwargs)
      self.activ.append(activ_func)
    self.activ = nn.Sequential(*self.activ)

    if as_class:
      self.weight, self.bias = conv.weight, conv.bias
      if apply_bn:
        self.bn_gamma, self.bn_beta = batch_norm.weight, batch_norm.bias
        self.bn_mean, self.bn_var = batch_norm.running_mean, batch_norm.running_var
        self.bn_eps = batch_norm.eps

  def forward(self, x, drop=0.0):
    x = self.activ(x)
    if drop:
      x = F.dropout(x, drop, self.training)
    return x


class Conv2d(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               as_class=False):
    super(Conv2d, self).__init__()
    self.activ, self.weights = [], []

    assert len(kernel_size) == 2
    if isinstance(padding, (str, int)):
      padding = (padding, padding)
    if isinstance(padding[0], str):
      padding_h = (kernel_size[0]-1) * dilation[0] // 2 if padding[0] == 'same' else 0
      padding_w = (kernel_size[1]-1) * dilation[1] // 2 if padding[1] == 'same' else 0
      padding = (padding_h, padding_w)

    ## Convolution
    conv = nn.Conv2d(
      in_dim, out_dim, kernel_size, stride=stride, 
      padding=padding, dilation=dilation, groups=groups, 
      bias=use_bias)
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    nn.init.xavier_normal_(conv.weight, gain=w_init_gain)
    if use_bias:
      nn.init.zeros_(conv.bias)
    self.conv = conv
    self.activ.append(conv)
    self.weights.append(conv.weight)

    ## Batch normalization
    if apply_bn:
      batch_norm = nn.BatchNorm2d(
        out_dim, eps=bn_eps, momentum=bn_mom, 
        affine=True, track_running_stats=True)
      self.bn = nn.Sequential(conv, batch_norm)
      self.activ.append(batch_norm)

    ## Activation function
    if activ is not None:
      activ_func = getattr(A, activ)(**kwargs)
      self.activ.append(activ_func)
    self.activ = nn.Sequential(*self.activ)

    if as_class:
      self.weight, self.bias = conv.weight, conv.bias
      if apply_bn:
        self.bn_gamma, self.bn_beta = batch_norm.weight, batch_norm.bias
        self.bn_mean, self.bn_var = batch_norm.running_mean, batch_norm.running_var
        self.bn_eps = batch_norm.eps

  def forward(self, x, drop=0.0):
    x = self.activ(x)
    if drop:
      x = F.dropout(x, drop, self.training)
    return x


class Conv2dCbr(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False):
    super(Conv2dCbr, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d + BatchNorm
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv + BatchNorm
    x = self.conv2(x) # no dropout
    ## ReLU
    return self.activ_func(x + residual)


class Conv2dBrc(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False):
    super(Conv2dBrc, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout
    return x + residual

#####################
## JeongHwan wrote ##
#####################

class Dense_eye(nn.Module):
  def __init__(self, in_dim, out_dim, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-06, 
               activ=None, kwargs={}, w_init_gain='linear', 
               as_class=False):
    super(Dense_eye, self).__init__()
    self.activ, self.weights = [], []

    ## Affine transform
    affine = nn.Linear(in_dim, out_dim, bias=use_bias)
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    nn.init.eye_(affine.weight)
    if use_bias:
      nn.init.zeros_(affine.bias)
    self.affine = affine
    self.activ.append(affine)
    self.weights.append(affine.weight)

    ## Batch normalization
    if apply_bn:
      batch_norm = nn.BatchNorm1d(
        out_dim, eps=bn_eps, momentum=bn_mom, 
        affine=True, track_running_stats=True)
      self.bn = nn.Sequential(affine, batch_norm)
      self.activ.append(batch_norm)

    ## Activation function
    if activ is not None:
      activ_func = getattr(A, activ)(**kwargs)
      self.activ.append(activ_func)
    self.activ = nn.Sequential(*self.activ)

    if as_class:
      self.weight, self.bias = affine.weight, affine.bias
      if apply_bn:
        self.bn_gamma, self.bn_beta = batch_norm.weight, batch_norm.bias
        self.bn_mean, self.bn_var = batch_norm.running_mean, batch_norm.running_var
        self.bn_eps = batch_norm.eps

  def forward(self, x, drop=0.0):
    x = self.activ(x)
    if drop:
      x = F.dropout(x, drop, self.training)
    return x

class Unet1D_down(nn.Module):
# """Downscaling with maxpool then double conv"""

  def __init__(self, in_dim, out_dim, kernel_size, stride=1, 
               padding='same', dilation=1, groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, activ=None,
               kwargs={}, w_init_gain='conv1d', as_class=False):

    super(Unet1D_down,self).__init__()
    self.weights = []
    # self.maxpool = nn.MaxPool1d(2)
    self.conv1 = Conv1d(
      in_dim, out_dim, kernel_size, stride, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv1d(
      out_dim, out_dim, kernel_size, 1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

  def forward(self, x):
    # x = self.maxpool(x)
    ## Conv + BatchNorm + ReLU
    # print(x.size()) 
    x = self.conv1(x)
       
    ## Conv + BatchNorm + ReLU
    # print(x.size()) 
    x = self.conv2(x) 
    # print(x.size()) 
    return x

class Unet1D_up(nn.Module):
#  """Upscaling then double conv"""

  def __init__(self, in_dim, out_dim, kernel_size, stride=1, 
               padding='same', dilation=1, groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, activ=None,
               kwargs={}, w_init_gain='conv1d', as_class=False):

    super(Unet1D_up,self).__init__()
    self.weights = []
    self.up_sample = nn.Upsample(scale_factor=2)
    # self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) ##2D
    self.conv1 = Conv1d(
      in_dim, out_dim, kernel_size, stride, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv1d(
      out_dim, out_dim, kernel_size, 1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

  def forward(self, x1, x2):
    x1 = self.up_sample(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]

    x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    ## Conv + BatchNorm + ReLU
    # print(x.size())
    x = self.conv1(x)
    ## Conv + BatchNorm + ReLU
    # print(x.size())
    x = self.conv2(x)    
    # print(x.size()) 
    return x

class Unet2D_down(nn.Module):
# """Downscaling with maxpool then double conv"""

  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, activ=None,
               kwargs={}, w_init_gain='conv2d', as_class=False):

    super(Unet2D_down,self).__init__()
    self.weights = []
    # self.maxpool = nn.MaxPool2d(2)
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, (1, 1), 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

  def forward(self, x):
    # x = self.maxpool(x)
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x)
    ## Conv + BatchNorm + ReLU
    x = self.conv2(x) 
    return x

class Unet2D_up(nn.Module):
#  """Upscaling then double conv"""

  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, activ=None,
               kwargs={}, w_init_gain='conv2d', as_class=False):

    super(Unet2D_up,self).__init__()
    self.weights = []
    # self.up_sample = nn.Upsample(scale_factor=2)
    self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) ##2D
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, (1, 1), 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

  def forward(self, x1, x2):
    x1 = self.up_sample(x1)

    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x)
    ## Conv + BatchNorm + ReLU
    x = self.conv2(x)       
    return x

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
  def __init__(self, channel_dims, reduction_ratio, pool_types=['avg', 'max']):
    super(ChannelGate, self).__init__()
    self.weights = []
    self.channel_dims = channel_dims
    self.linear1 = nn.Linear(channel_dims, channel_dims // reduction_ratio)
    self.linear2 = nn.Linear(channel_dims // reduction_ratio, channel_dims)
    
    self.mlp = nn.Sequential(Flatten(),self.linear1, nn.ReLU(), self.linear2)
    self.pool_types = pool_types

    self.weights.append(self.linear1.weight)
    self.weights.append(self.linear2.weight)

  def forward(self, x):
    for pool_type in self.pool_types:
      if pool_type=='avg':
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_raw = self.mlp( avg_pool )
      elif pool_type=='max':
        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_raw = self.mlp( max_pool )
      elif pool_type=='lp':
        lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_raw = self.mlp( lp_pool )
      elif pool_type=='lse': # LSE pool only
        lse_pool = logsumexp_2d(x)
        channel_att_raw = self.mlp( lse_pool )

    scale = torch.sigmoid( channel_att_raw ).unsqueeze(2).unsqueeze(3).expand_as(x)
    return x * scale

# class AvgMaxPool(nn.Module):
#   def __init__(self, axis=1):
#     super(AvgMaxPool, self).__init__()
#     self.axis = axis
#     ## axis 2:frq, 3:time
#   def forward(self, x):
#     if self.axis == 2:
#       x = x.permute(0,2,1,3)
#       avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#       max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#       out = torch.cat( (avg_pool, max_pool), dim=2 )
#       out = out.permute(0,2,1,3)    
#     elif self.axis == 3:
#       x = x.permute(0,3,2,1)
#       avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#       max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))    
#       out = torch.cat( (avg_pool, max_pool), dim=3 )
#       out = out.permute(0,3,2,1)     
#     else:
#       out = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
#     return out

class AvgMaxPool(nn.Module):
  def __init__(self, axis=1):
    super(AvgMaxPool, self).__init__()
    self.axis = axis
    ## axis 2:frq, 3:time
  def forward(self, x):
    if self.axis == 2:
      x = torch.mean(x,dim=3).unsqueeze(3)
      avg_pool = torch.mean(x,dim=1).unsqueeze(1)
      max_pool = torch.max(x,dim=1)[0].unsqueeze(1)
      out = torch.cat( (max_pool, avg_pool), dim=1)
    elif self.axis == 3:
      x = torch.mean(x,dim=(2)).unsqueeze(2)
      avg_pool = torch.mean(x,dim=1).unsqueeze(1)
      max_pool = torch.max(x,dim=1)[0].unsqueeze(1)
      out = torch.cat( (max_pool, avg_pool), dim=1)
    else:
      out = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    return out

class CBAMGate(nn.Module):
  def __init__(self, type_CBAM='ft-CBAM', kernal_CBAM=7):
    super(CBAMGate, self).__init__()
    self.weights = []
    self.type_CBAM = type_CBAM
    ## f-CBAM
    self.f_compress = AvgMaxPool(axis=2)
    self.f_spatial = Conv2d(2, 1, kernel_size=(kernal_CBAM,1), stride=1, padding="same",
      dilation=(1,1), groups=1, use_bias=False, apply_bn=True, bn_mom=0.99, bn_eps=1e-6, 
      activ=None, kwargs={}, w_init_gain='conv2d', as_class=False)

    ## t-CBAM
    self.t_compress = AvgMaxPool(axis=3)
    self.t_spatial = Conv2d(2, 1, kernel_size=(1,kernal_CBAM), stride=1, padding="same",
      dilation=(1,1), groups=1, use_bias=False, apply_bn=True, bn_mom=0.99, bn_eps=1e-6, 
      activ=None, kwargs={}, w_init_gain='conv2d', as_class=False)

    if self.type_CBAM == 'f-CBAM':
      self.weights.extend(self.f_spatial.weights)
    elif self.type_CBAM == 't-CBAM':
      self.weights.extend(self.t_spatial.weights)
    else:
      self.weights.extend(self.f_spatial.weights)
      self.weights.extend(self.t_spatial.weights)

  def forward(self, x):
    x_f_compress = self.f_compress(x)
    x_f_out = self.f_spatial(x_f_compress)
    scale_f = torch.sigmoid(x_f_out) # broadcasting
    
    x_t_compress = self.t_compress(x)
    x_t_out = self.t_spatial(x_t_compress)
    scale_t = torch.sigmoid(x_t_out) # broadcasting

    out_f = x * scale_f
    out_t = x * scale_t

    if self.type_CBAM == 'f-CBAM':
      out = out_f ## f-CBAM
    elif self.type_CBAM == 't-CBAM':
      out = out_t ## t-CBAM
    else:
      out = (out_f + out_t)/2 ## ft-CBAM

    return out

class CBAM(nn.Module):
  def __init__(self, channel_dims, type_CBAM, kernal_CBAM,
               reduction_ratio=8, pool_types=['avg', 'max'], no_channel_att=False):
    super(CBAM, self).__init__()
    self.weights = []
    self.CBAMGate = CBAMGate(type_CBAM, kernal_CBAM)
    self.weights.extend(self.CBAMGate.weights)
    self.no_channel_att = no_channel_att
    if not self.no_channel_att:  
      self.ChannelGate = ChannelGate(channel_dims, reduction_ratio, pool_types)
      self.weights.extend(self.ChannelGate.weights)
  def forward(self, x):
    if not self.no_channel_att:
      x = self.ChannelGate(x)
    x_out = self.CBAMGate(x)
    return x_out

class Conv2dBrc_CBAM(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               type_CBAM='ft-CBAM', kernal_CBAM=7,
               reduction_ratio=8, pool_types=['avg', 'max'], no_channel_att=False):
    super(Conv2dBrc_CBAM, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    self.cbam = CBAM(out_dim, type_CBAM, kernal_CBAM, reduction_ratio, pool_types, no_channel_att)
    self.weights.extend(self.cbam.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout
    ## cbam
    x = self.cbam(x)

    return x + residual

class eca_layer(nn.Module):
  """Constructs a ECA module.
  Args:
      channel: Number of channels of the input feature map
      k_size: Adaptive selection of kernel size
  """
  def __init__(self, channel, type_ECA='c-ECA', kernal_ECA=3):
    super(eca_layer, self).__init__()
    self.weights = []
    self.type_ECA = type_ECA
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.sigmoid = nn.Sigmoid()

    if self.type_ECA == 'f-ECA':
      ## f-eca
      self.conv_f = Conv1d(
        1, 1, kernel_size=kernal_ECA, stride=1, padding=(kernal_ECA - 1)//2, dilation=1, 
        groups=1, use_bias=False, apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
        activ=None, kwargs={}, w_init_gain='conv1d', as_class=False)      
      self.weights.extend(self.conv_f.weights)
    elif self.type_ECA == 't-ECA':      
      ## t-eca
      self.conv_t = Conv1d(
        1, 1, kernel_size=kernal_ECA, stride=1, padding=(kernal_ECA - 1)//2, dilation=1, 
        groups=1, use_bias=False, apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
        activ=None, kwargs={}, w_init_gain='conv1d', as_class=False)
      self.weights.extend(self.conv_t.weights)
    else:
      ## c-eca (orginal)
      self.conv_c = Conv1d(
        1, 1, kernel_size=kernal_ECA, stride=1, padding=(kernal_ECA - 1)//2, dilation=1, 
        groups=1, use_bias=False, apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
        activ=None, kwargs={}, w_init_gain='conv1d', as_class=False)
      self.weights.extend(self.conv_c.weights)

    
  def forward(self, x):
    # x: input features with shape [b, c, h, w]
    # x: input features with shape [b, c, f, t]

    if self.type_ECA == 'f-ECA':
      x_f = x.transpose(-2,-3)
      y_f = self.avg_pool(x_f)
      y_f = self.conv_f(y_f.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
      y_f = self.sigmoid(y_f)
      out = x_f * y_f.expand_as(x_f)
      out = out.transpose(-2,-3) 
    elif self.type_ECA == 't-ECA':
      x_t = x.transpose(-1,-3)
      y_t = self.avg_pool(x_t)
      y_t = self.conv_t(y_t.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
      y_t = self.sigmoid(y_t)
      out = x_t * y_t.expand_as(x_t)
      out = out.transpose(-1,-3)       
    else:
      # feature descriptor on the global spatial information
      y_c = self.avg_pool(x)
      # Two different branches of ECA module
      y_c = self.conv_c(y_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
      # Multi-scale information fusion
      y_c = self.sigmoid(y_c)
      
      out = x * y_c.expand_as(x)  

    return out

class Statspooling(nn.Module):
  def __init__(self, pool_axis=-1):
    super(Statspooling, self).__init__()
    self.pool_axis = pool_axis 
  def forward(self, x):
    pool_var, pool_mean = torch.var_mean(
      x, dim=self.pool_axis, keepdim=True, unbiased=False)
    pool_var = torch.max(pool_var, 1e-12*torch.ones_like(pool_var))
    pool_std = torch.sqrt(pool_var)

    return pool_mean, pool_std
        
class DSP_pool(nn.Module):
  def __init__(self):
    super(DSP_pool, self).__init__()
    self.statspool_dim2 = Statspooling(pool_axis=-2)
    self.statspool_dim1 = Statspooling(pool_axis=-1)
  def forward(self, x):    
    x_mean, x_std = self.statspool_dim2(x)
    x_mean_mean, x_mean_std = self.statspool_dim1(x_mean)
    x_std_mean, x_std_std = self.statspool_dim1(x_std)
    x_out = torch.cat((x_mean_mean, x_mean_std, x_std_mean, x_std_std), dim=-2)
   
    return x_out

class dsp_eca_layer(nn.Module):
  def __init__(self, channel, type_ECA='c-ECA', kernal_ECA=3):
    super(dsp_eca_layer, self).__init__()
    self.weights = []
    self.type_ECA = type_ECA
    self.sigmoid = nn.Sigmoid()

    self.dsp_pool = DSP_pool()

    if self.type_ECA == 'f-ECA':
      ## f-eca
      self.conv_f = Conv2d(4, 1, kernel_size=(1,kernal_ECA), stride=1, padding="same",
      dilation=(1,1), groups=1, use_bias=False, apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
      activ=None, kwargs={}, w_init_gain='conv2d', as_class=False)      
      self.weights.extend(self.conv_f.weights)
    elif self.type_ECA == 't-ECA':      
      ## t-eca
      self.conv_t = Conv2d(4, 1, kernel_size=(1,kernal_ECA), stride=1, padding="same",
      dilation=(1,1), groups=1, use_bias=False, apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
      activ=None, kwargs={}, w_init_gain='conv2d', as_class=False)
      self.weights.extend(self.conv_t.weights)
    else:
      ## c-eca (orginal)
      self.conv_c = Conv2d(4, 1, kernel_size=(1,kernal_ECA), stride=1, padding="same",
      dilation=(1,1), groups=1, use_bias=False, apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
      activ=None, kwargs={}, w_init_gain='conv2d', as_class=False)
      self.weights.extend(self.conv_c.weights)

  def forward(self, x):
    # x: input features with shape [b, c, f, t]

    if self.type_ECA == 'f-ECA':
      x_f = x.transpose(-2,-3)
      y_f = self.dsp_pool(x_f)
      y_f = self.conv_c(y_f.transpose(-2, -3)).transpose(-2, -3)
      y_f = self.sigmoid(y_f)
      out = x_f * y_f.expand_as(x_f)
      out = out.transpose(-2,-3) 
    elif self.type_ECA == 't-ECA':
      x_t = x.transpose(-1,-3)
      y_t = self.dsp_pool(x_t)
      y_t = self.conv_c(y_t.transpose(-2, -3)).transpose(-2, -3)
      y_t = self.sigmoid(y_t)
      out = x_t * y_t.expand_as(x_t)
      out = out.transpose(-1,-3)       
    else:
      # feature descriptor on the global spatial information
      y_c = self.dsp_pool(x)
      # Two different branches of ECA module
      # y_c = self.conv_c(y_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
      y_c = self.conv_c(y_c.transpose(-2, -3)).transpose(-2, -3)

      
      # Multi-scale information fusion
      y_c = self.sigmoid(y_c)
      
      out = x * y_c.expand_as(x)  

    return out

class fc_eca_layer(nn.Module):
  def __init__(self, channel, type_ECA='c-ECA', kernal_ECA=3):
    super(fc_eca_layer, self).__init__()
    self.weights = []
    self.type_ECA = type_ECA
    self.sigmoid = nn.Sigmoid()

    # # BatchNorm
    # self.bn = nn.BatchNorm2d(
    #   channel, eps=0.99, momentum=1e-6, 
    #   affine=True, track_running_stats=True)

    # ## ReLU
    # self.activ_func = getattr(A, "ReLU")(**{})

    ## fc-eca (orginal)
    self.conv_c = Conv2d(2, 1, kernel_size=(kernal_ECA,kernal_ECA), stride=(1, 1), padding="same",
      dilation=(1,1), groups=1, use_bias=False, apply_bn=True, bn_mom=0.99, bn_eps=1e-6, 
      activ=None, kwargs={}, w_init_gain='conv2d', as_class=False)
    self.weights.extend(self.conv_c.weights)

  def forward(self, x):
    # x = self.activ_func(self.bn(x)) ##com1
    # x = self.bn(x)  ##com0

    # x: input features with shape [b, c, h, w]
    # x: input features with shape [b, c, f, t]
    # feature descriptor on the global spatial information
    # print(x.size())


    y_c_var, y_c_mean = torch.var_mean(
      x, dim=-1, keepdim=True, unbiased=False)
    y_c_var = torch.max(y_c_var, 1e-12*torch.ones_like(y_c_var))
    y_c_std = torch.sqrt(y_c_var)
    y_c = torch.cat((y_c_std,y_c_mean),dim=-1)    

    # y_c = torch.mean(x, dim=-1, keepdim=True)

    # print(y_c.size())
    # exit()
    y_c = y_c.transpose(-1,-3)
    # Two different branches of ECA module
    y_c = self.conv_c(y_c)
    y_c = y_c.transpose(-1,-3)
    # Multi-scale information fusion
    y_c = self.sigmoid(y_c)
    
    out = x * y_c.expand_as(x)  
    # print(out.size())
    return out    

class Conv2dBrc_ECA(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               type_ECA='c-ECA', kernal_ECA=3):
    super(Conv2dBrc_ECA, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    self.eca = eca_layer(out_dim, type_ECA, kernal_ECA)
    self.weights.extend(self.eca.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout
    ## eca
    x = self.eca(x)

    return x + residual

class Conv2dBrc_dspECA(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               type_ECA='c-ECA', kernal_ECA=3):
    super(Conv2dBrc_dspECA, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    self.eca = fc_eca_layer(out_dim, type_ECA, kernal_ECA)
    self.weights.extend(self.eca.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout
    ## eca
    x = self.eca(x)

    return x + residual

class Conv2dBrc_fcECA(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               type_ECA='c-ECA', kernal_ECA=3):
    super(Conv2dBrc_fcECA, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    self.eca = fc_eca_layer(in_dim, type_ECA, kernal_ECA)
    self.weights.extend(self.eca.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    ## eca

    x = self.eca(x)
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout

    return x + residual


class Conv2dBrc_fffECA(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               type_ECA='c-ECA', kernal_ECA=3):
    super(Conv2dBrc_fffECA, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    self.eca = eca_layer(out_dim, type_ECA, kernal_ECA)
    self.weights.extend(self.eca.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    ## eca
    x = self.eca(x)
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout

    return x + residual


class Conv2dBrc_ECA_CBAM(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               type_ECA='c-ECA', kernal_ECA=3,
               type_CBAM='ft-CBAM', kernal_CBAM=7):
    super(Conv2dBrc_ECA_CBAM, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    self.eca = eca_layer(out_dim, type_ECA, kernal_ECA)
    self.weights.extend(self.eca.weights)

    self.cbam = CBAM(out_dim, type_CBAM, kernal_CBAM, no_channel_att=True)
    self.weights.extend(self.cbam.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout
    ## eca
    x = self.eca(x)
    ## cbam
    x = self.cbam(x)

    return x + residual

class Conv2dBrc_NECA_CBAM(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               type_ECA='c-ECA', kernal_ECA=3,
               type_CBAM='ft-CBAM', kernal_CBAM=7):
    super(Conv2dBrc_NECA_CBAM, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv2d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    self.eca = eca_layer(out_dim, type_ECA, kernal_ECA)
    self.weights.extend(self.eca.weights)

    self.fcbam = CBAM(out_dim, type_CBAM='f-CBAM', kernal_CBAM=kernal_CBAM, no_channel_att=True)
    self.weights.extend(self.fcbam.weights)

    self.tcbam = CBAM(out_dim, type_CBAM='t-CBAM', kernal_CBAM=kernal_CBAM, no_channel_att=True)
    self.weights.extend(self.tcbam.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout
    ## eca
    x_c = self.eca(x)
    ## cbam
    x_f = self.fcbam(x)
    x_t = self.tcbam(x)
    
    c = int(x.size(-3)); f= int(x.size(-2)); t= int(x.size(-1))
    # k=c+f+t

    # x = c/k*x_c + f/k*x_f + t/k*x_t ##gpu_id = 0
    # x = (f+t)/k/2*x_c + (c+t)/k/2*x_f + (f+c)/k/2*x_t ##gpu_id = 1
    k = c+f
    x = c/k*x_c + f/k/2*x_f + f/k/2*x_t ##gpu_id = 0

    return x + residual

# class Conv2dBrc_FT_ECA(nn.Module):
#   def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
#                padding='same', dilation=(1, 1), groups=1, use_bias=True, 
#                apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
#                activ=None, kwargs={}, w_init_gain='conv2d', 
#                shortcut_bn=False, as_class=False, 
#                kernal_ECA=3):
#     super(Conv2dBrc_FT_ECA, self).__init__()
#     self.activ, self.weights = [], []

#     assert activ is not None and apply_bn is True

#     if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
#       stride1, stride2 = stride
#       stride1, stride2 = tuple(stride1), tuple(stride2)
#       if stride1 == (1, 1):
#         assert stride2 != (1, 1)
#         stride_rsz = stride2
#       else:
#         stride_rsz = stride1
#       resize = True
#     else:
#       assert tuple(stride) == (1, 1)  # should not resize
#       stride1 = stride2 = stride
#       resize = False

#     ## BatchNorm
#     self.bn = nn.BatchNorm2d(
#       in_dim, eps=bn_eps, momentum=bn_mom, 
#       affine=True, track_running_stats=True)

#     ## ReLU
#     self.activ_func = getattr(A, activ)(**kwargs)

#     ## Conv2d + BatchNorm + ReLU
#     self.conv1 = Conv2d(
#       in_dim, out_dim, kernel_size, stride1, 
#       padding, dilation, groups, use_bias, 
#       apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#       activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
#       as_class=False)
#     self.weights.extend(self.conv1.weights)

#     ## Conv2d
#     self.conv2 = Conv2d(
#       out_dim, out_dim, kernel_size, stride2, 
#       padding, dilation, groups, use_bias, 
#       apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
#       activ=None, kwargs={}, w_init_gain=w_init_gain, 
#       as_class=False)
#     self.weights.extend(self.conv2.weights)

#     self.feca = eca_layer(out_dim, type_ECA='f-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.feca.weights)

#     self.teca = eca_layer(out_dim, type_ECA='t-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.teca.weights)

#     ## Shortcut
#     self.input_trans = Identity()
#     if not resize:
#       if out_dim == in_dim:
#         print('Input size is same as the output size...')
#       else:
#         print('Modifying #channels for shortcut (1x1conv)...')
#         ## Increase/Decrease #channels with 1x1 convolution
#         self.input_trans = Conv2d(
#           in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
#           padding=(0, 0), dilation=(1, 1), use_bias=False, 
#           apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#           activ=None, kwargs={}, w_init_gain=w_init_gain, 
#           as_class=False)
#         self.weights.extend(self.input_trans.weights)
#     else:
#       if out_dim == in_dim:
#         print('Resizing input for shortcut (maxpool)...')
#         raise NotImplementedError
#       else:
#         print('Resizing input for shortcut (1x1conv)...')
#         self.input_trans = Conv2d(
#           in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
#           padding=(0, 0), dilation=(1, 1), use_bias=False, 
#           apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#           activ=None, kwargs={}, w_init_gain=w_init_gain, 
#           as_class=False)
#         self.weights.extend(self.input_trans.weights)

#   def forward(self, x, drop=0.0):
#     residual = self.input_trans(x)
#     ## BatchNorm + ReLU
#     x = self.activ_func(self.bn(x))
#     ## Conv + BatchNorm + ReLU
#     x = self.conv1(x, drop=drop)
#     ## Conv
#     x = self.conv2(x) # no dropout
#     ## eca
#     x_f = self.feca(x)
#     x_t = self.teca(x)
#     x = (x_f + x_t)/2 

#     return x + residual

# class Conv2dBrc_CFT_ECA(nn.Module):
#   def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
#                padding='same', dilation=(1, 1), groups=1, use_bias=True, 
#                apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
#                activ=None, kwargs={}, w_init_gain='conv2d', 
#                shortcut_bn=False, as_class=False, 
#                kernal_ECA=3):
#     super(Conv2dBrc_CFT_ECA, self).__init__()
#     self.activ, self.weights = [], []

#     assert activ is not None and apply_bn is True

#     if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
#       stride1, stride2 = stride
#       stride1, stride2 = tuple(stride1), tuple(stride2)
#       if stride1 == (1, 1):
#         assert stride2 != (1, 1)
#         stride_rsz = stride2
#       else:
#         stride_rsz = stride1
#       resize = True
#     else:
#       assert tuple(stride) == (1, 1)  # should not resize
#       stride1 = stride2 = stride
#       resize = False

#     ## BatchNorm
#     self.bn = nn.BatchNorm2d(
#       in_dim, eps=bn_eps, momentum=bn_mom, 
#       affine=True, track_running_stats=True)

#     ## ReLU
#     self.activ_func = getattr(A, activ)(**kwargs)

#     ## Conv2d + BatchNorm + ReLU
#     self.conv1 = Conv2d(
#       in_dim, out_dim, kernel_size, stride1, 
#       padding, dilation, groups, use_bias, 
#       apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#       activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
#       as_class=False)
#     self.weights.extend(self.conv1.weights)

#     ## Conv2d
#     self.conv2 = Conv2d(
#       out_dim, out_dim, kernel_size, stride2, 
#       padding, dilation, groups, use_bias, 
#       apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
#       activ=None, kwargs={}, w_init_gain=w_init_gain, 
#       as_class=False)
#     self.weights.extend(self.conv2.weights)

#     self.ceca = eca_layer(out_dim, type_ECA='c-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.ceca.weights)

#     self.feca = eca_layer(out_dim, type_ECA='f-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.feca.weights)

#     self.teca = eca_layer(out_dim, type_ECA='t-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.teca.weights)

#     ## Shortcut
#     self.input_trans = Identity()
#     if not resize:
#       if out_dim == in_dim:
#         print('Input size is same as the output size...')
#       else:
#         print('Modifying #channels for shortcut (1x1conv)...')
#         ## Increase/Decrease #channels with 1x1 convolution
#         self.input_trans = Conv2d(
#           in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
#           padding=(0, 0), dilation=(1, 1), use_bias=False, 
#           apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#           activ=None, kwargs={}, w_init_gain=w_init_gain, 
#           as_class=False)
#         self.weights.extend(self.input_trans.weights)
#     else:
#       if out_dim == in_dim:
#         print('Resizing input for shortcut (maxpool)...')
#         raise NotImplementedError
#       else:
#         print('Resizing input for shortcut (1x1conv)...')
#         self.input_trans = Conv2d(
#           in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
#           padding=(0, 0), dilation=(1, 1), use_bias=False, 
#           apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#           activ=None, kwargs={}, w_init_gain=w_init_gain, 
#           as_class=False)
#         self.weights.extend(self.input_trans.weights)

#   def forward(self, x, drop=0.0):
#     residual = self.input_trans(x)
#     ## BatchNorm + ReLU
#     x = self.activ_func(self.bn(x))
#     ## Conv + BatchNorm + ReLU
#     x = self.conv1(x, drop=drop)
#     ## Conv
#     x = self.conv2(x) # no dropout
#     ## eca
#     x = self.ceca(x)

#     ## eca
#     x_f = self.feca(x)
#     x_t = self.teca(x)
#     x = (x_f + x_t)/2 

#     return x + residual

# class Conv2dBrc_NFT_ECA(nn.Module):
#   def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
#                padding='same', dilation=(1, 1), groups=1, use_bias=True, 
#                apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
#                activ=None, kwargs={}, w_init_gain='conv2d', 
#                shortcut_bn=False, as_class=False, 
#                kernal_ECA=3):
#     super(Conv2dBrc_NFT_ECA, self).__init__()
#     self.activ, self.weights = [], []

#     assert activ is not None and apply_bn is True

#     if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
#       stride1, stride2 = stride
#       stride1, stride2 = tuple(stride1), tuple(stride2)
#       if stride1 == (1, 1):
#         assert stride2 != (1, 1)
#         stride_rsz = stride2
#       else:
#         stride_rsz = stride1
#       resize = True
#     else:
#       assert tuple(stride) == (1, 1)  # should not resize
#       stride1 = stride2 = stride
#       resize = False

#     ## BatchNorm
#     self.bn = nn.BatchNorm2d(
#       in_dim, eps=bn_eps, momentum=bn_mom, 
#       affine=True, track_running_stats=True)

#     ## ReLU
#     self.activ_func = getattr(A, activ)(**kwargs)

#     ## Conv2d + BatchNorm + ReLU
#     self.conv1 = Conv2d(
#       in_dim, out_dim, kernel_size, stride1, 
#       padding, dilation, groups, use_bias, 
#       apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#       activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
#       as_class=False)
#     self.weights.extend(self.conv1.weights)

#     ## Conv2d
#     self.conv2 = Conv2d(
#       out_dim, out_dim, kernel_size, stride2, 
#       padding, dilation, groups, use_bias, 
#       apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
#       activ=None, kwargs={}, w_init_gain=w_init_gain, 
#       as_class=False)
#     self.weights.extend(self.conv2.weights)

#     self.ceca = eca_layer(out_dim, type_ECA='c-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.ceca.weights)

#     self.feca = eca_layer(out_dim, type_ECA='f-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.feca.weights)

#     self.teca = eca_layer(out_dim, type_ECA='t-ECA', kernal_ECA=kernal_ECA)
#     self.weights.extend(self.teca.weights)

#     ## Shortcut
#     self.input_trans = Identity()
#     if not resize:
#       if out_dim == in_dim:
#         print('Input size is same as the output size...')
#       else:
#         print('Modifying #channels for shortcut (1x1conv)...')
#         ## Increase/Decrease #channels with 1x1 convolution
#         self.input_trans = Conv2d(
#           in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
#           padding=(0, 0), dilation=(1, 1), use_bias=False, 
#           apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#           activ=None, kwargs={}, w_init_gain=w_init_gain, 
#           as_class=False)
#         self.weights.extend(self.input_trans.weights)
#     else:
#       if out_dim == in_dim:
#         print('Resizing input for shortcut (maxpool)...')
#         raise NotImplementedError
#       else:
#         print('Resizing input for shortcut (1x1conv)...')
#         self.input_trans = Conv2d(
#           in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
#           padding=(0, 0), dilation=(1, 1), use_bias=False, 
#           apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
#           activ=None, kwargs={}, w_init_gain=w_init_gain, 
#           as_class=False)
#         self.weights.extend(self.input_trans.weights)

#   def forward(self, x, drop=0.0):
#     residual = self.input_trans(x)
#     ## BatchNorm + ReLU
#     x = self.activ_func(self.bn(x))
#     ## Conv + BatchNorm + ReLU
#     x = self.conv1(x, drop=drop)
#     ## Conv
#     x = self.conv2(x) # no dropout
#     ## eca
#     x_c = self.ceca(x)
#     x_f = self.feca(x)
#     x_t = self.teca(x)
#     x = (x_c + x_f + x_t)/3 

#     return x + residual


class Mish(nn.Module):
  def __init__(self):
    super().__init__()
    print("Mish activation loaded...")

  def forward(self, x):  
    return x *( torch.tanh(F.softplus(x))) 
    

class Conv2dCbr_Res2Net(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               base_width=26, scale=4, channel0=32 ,first_block=False):
    super(Conv2dCbr_Res2Net, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    # self.activ_func = torch.nn.ReLU()
    self.activ_func = Mish()

    self.width = int(out_dim * (base_width / channel0)) * groups
    
    self.conv0 = Conv2d(
      in_dim, self.width * scale, kernel_size, (1, 1), 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv0.weights)

    # If scale == 1, single conv else identity & (scale - 1) convs
    self.nb_branches = max(scale, 2) - 1
    if first_block:
      self.pool = nn.AvgPool2d(kernel_size=3, stride=stride1, padding=1)
    
    self.first_block = first_block
    self.scale = scale

    self.convs = nn.ModuleList()
    for _ in range(self.nb_branches):
      conv1 = Conv2d(
        self.width, self.width, kernel_size, stride1, 
        padding, dilation, groups, use_bias, 
        apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
        activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
        as_class=False)
      self.weights.extend(conv1.weights)
      self.convs.append(conv1)

    self.conv2 = Conv2d(
      self.width * scale, out_dim, [1,1], (1, 1), 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## Conv + BatchNorm + ReLU
    x = self.conv0(x, drop=drop)
    x = self.activ_func(x)

    spx = torch.split(x, self.width, 1)
    for i in range(self.nb_branches):
      if i==0 or self.first_block:
        sp = spx[i]
      else:
        sp = sp + spx[i]      
      sp = self.convs[i](sp)
      sp = self.activ_func(sp)
      if i==0:
        x = sp
      else:
        x = torch.cat((x, sp), 1)
    if self.scale > 1:
      if self.first_block:
        x = torch.cat((x, self.pool(spx[self.nb_branches])), 1)
      else:
        x = torch.cat((x, spx[self.nb_branches]), 1)

    x = self.conv2(x)
    # print(x.size())
    return self.activ_func(x + residual)

class SEModule(nn.Module):
  def __init__(self, channels, reduction=16):
    super(SEModule, self).__init__()
    self.weights = []
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
    self.weights.append(self.fc1.weight)
    # self.relu = nn.ReLU(inplace=True)
    self.relu = Mish()
    self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
    self.weights.append(self.fc2.weight)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):
    x = self.avg_pool(input)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return input * x

class Conv2dCbr_SE_Res2Net(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False, 
               base_width=26, scale=4, channel0=32, first_block=False, reduction=16):
    super(Conv2dCbr_SE_Res2Net, self).__init__()
    self.activ, self.weights = [], []

    # assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm2d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    # self.activ_func = torch.nn.ReLU()
    self.activ_func = Mish()

    self.width = int(out_dim * (base_width / channel0)) * groups
    
    self.conv0 = Conv2d(
      in_dim, self.width * scale, kernel_size, (1, 1), 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv0.weights)

    # If scale == 1, single conv else identity & (scale - 1) convs
    self.nb_branches = max(scale, 2) - 1
    if first_block:
      self.pool = nn.AvgPool2d(kernel_size=3, stride=stride1, padding=1)
    
    self.first_block = first_block
    self.scale = scale

    self.convs = nn.ModuleList()
    for _ in range(self.nb_branches):
      conv1 = Conv2d(
        self.width, self.width, kernel_size, stride1, 
        padding, dilation, groups, use_bias, 
        apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
        activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
        as_class=False)
      self.weights.extend(conv1.weights)      
      self.convs.append(conv1)

    self.conv2 = Conv2d(
      self.width * scale, out_dim, [1,1], (1, 1), 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    ## SE_block
    self.se_block = SEModule(out_dim, reduction=reduction)
    self.weights.extend(self.se_block.weights)
    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## Conv + BatchNorm + ReLU
    x = self.conv0(x, drop=drop)
    x = self.activ_func(x)

    spx = torch.split(x, self.width, 1)
    for i in range(self.nb_branches):
      if i==0 or self.first_block:
        sp = spx[i]
      else:
        sp = sp + spx[i]      
      sp = self.convs[i](sp)
      sp = self.activ_func(sp)
      if i==0:
        x = sp
      else:
        x = torch.cat((x, sp), 1)
    if self.scale > 1:
      if self.first_block:
        x = torch.cat((x, self.pool(spx[self.nb_branches])), 1)
      else:
        x = torch.cat((x, spx[self.nb_branches]), 1)

    x = self.conv2(x)
    x = self.se_block(x)

    # print(x.size())
    return self.activ_func(x + residual)


class GRUlayer(nn.Module):
  def __init__(self, in_dim, hid_dim, proj_dim=0, proj_activ=None, num_layers=1, batch_first=True,
              bidirectional=True, as_class=False):
    super(GRUlayer, self).__init__()
    self.weights = []
    self.proj_dim = proj_dim
    self.proj_activ = proj_activ

    self.lstm = nn.GRU(in_dim, hid_dim, num_layers, batch_first=batch_first,
                        bidirectional=bidirectional)    
    for name, param in self.lstm._parameters.items():
      if name.startswith('weight'):
        self.weights.append(param)

    num_directions = 2 if bidirectional else 1
    self.batch_dim = 0 if batch_first else 1
    self.hid_dim = hid_dim
    self.tot_layers = num_layers * num_directions

    if proj_dim:
      self.projector = nn.Linear(hid_dim*num_directions, proj_dim, bias=False)
      self.weights.append(self.projector.weight)  
      if proj_activ is not None:
        self.activ_func = getattr(A, proj_activ)(**{})

  def forward(self, x, hid_init_op=False, hid_in=(0,0), drop=0.0):
    if x is None:
      batch_size = 128
      x = torch.zeros(batch_size, 1, self.proj_dim).to(torch.device('cuda'))
      
    batch_size = x.size(self.batch_dim)

    if not hid_init_op:
      hid_in = torch.zeros(self.tot_layers, batch_size, self.hid_dim).to(x.device)

    h0 = hid_in

    x, hn = self.lstm(x, h0)
    hid_out = hn
    if self.proj_dim:
      x = self.projector(x)
      if self.proj_activ is not None:
        x = self.activ_func(x) 
    if drop:
      x = F.dropout(x, drop, self.training)

    return x , hid_out

class LSTMlayer(nn.Module):
  def __init__(self, in_dim, hid_dim, proj_dim=0, proj_activ=None, num_layers=1, batch_first=True,
              bidirectional=True, as_class=False):
    super(LSTMlayer, self).__init__()
    self.weights = []
    self.proj_dim = proj_dim
    self.proj_activ = proj_activ

    self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=batch_first,
                        bidirectional=bidirectional) 
    for name, param in self.lstm._parameters.items():
      if name.startswith('weight'):
        self.weights.append(param)

    num_directions = 2 if bidirectional else 1
    self.batch_dim = 0 if batch_first else 1
    self.hid_dim = hid_dim
    self.tot_layers = num_layers * num_directions

    if proj_dim:
      self.projector = nn.Linear(hid_dim*num_directions, proj_dim, bias=False)
      self.weights.append(self.projector.weight)  
      if proj_activ is not None:
        self.activ_func = getattr(A, proj_activ)(**{})

  def forward(self, x, hid_init_op=False, hid_in=(0,0), drop=0.0):
    if x is None:
      batch_size = 128
      x = torch.zeros(batch_size, 1, self.proj_dim).to(torch.device('cuda'))
      
    batch_size = x.size(self.batch_dim)

    if not hid_init_op:
      hid_in = []
      hid_in.append(torch.zeros(self.tot_layers, batch_size, self.hid_dim).to(x.device))
      hid_in.append(torch.zeros(self.tot_layers, batch_size, self.hid_dim).to(x.device))

    h0 = hid_in[0]
    c0 = hid_in[1]

    x, (hn, cn) = self.lstm(x, (h0, c0))
    hid_out = (hn, cn)
    if self.proj_dim:
      x = self.projector(x)
      if self.proj_activ is not None:
        x = self.activ_func(x) 
    if drop:
      x = F.dropout(x, drop, self.training)

    return x , hid_out

from torch.autograd import Variable

class Mask(nn.Module):
  def __init__(self, in_dim, 
               kwargs={}, w_init_gain='linear'):
    super(Mask, self).__init__()
    self.weights = []
    dtype = torch.FloatTensor
    ## Affine transform
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    
    # affine_wight = torch.ones(1,in_dim)
    affine_wight = torch.empty(1,in_dim)
    nn.init.xavier_normal_(affine_wight, gain=w_init_gain)
    self.affine = torch.nn.Parameter(data=affine_wight, requires_grad=True)
    # self.affine = Variable(affine_wight.type(dtype), requires_grad=True)
    self.weights.append(self.affine)

    # bias_wight = torch.zeros(1,in_dim)
    bias_wight = torch.empty(1,in_dim)    
    nn.init.xavier_normal_(bias_wight, gain=w_init_gain)
    self.bias = torch.nn.Parameter(data=bias_wight, requires_grad=True)
    # self.bias = Variable(bias_wight.type(dtype), requires_grad=True)
    self.weights.append(self.bias)

  def forward(self, x):
    # x = torch.mul(self.affine.expand_as(x),x)
    x = x+self.bias.expand_as(x)
    return x

class Con_Mask(nn.Module):
  def __init__(self, in_dim,
               kwargs={}, w_init_gain='linear'):
    super(Con_Mask, self).__init__()
    self.weights = []
    dtype = torch.FloatTensor
    ## Affine transform
    if isinstance(w_init_gain, str):
      w_init_gain = nn.init.calculate_gain(w_init_gain, *kwargs.values())
    
    affine_wight = torch.empty(1,in_dim)
    nn.init.xavier_normal_(affine_wight, gain=w_init_gain)
    self.affine = torch.nn.Parameter(data=affine_wight, requires_grad=True)
    # self.affine = Variable(affine_wight.type(dtype), requires_grad=True)
    self.weights.append(self.affine)

    bias_wight = torch.empty(1,in_dim)    
    nn.init.xavier_normal_(bias_wight, gain=w_init_gain)
    self.bias = torch.nn.Parameter(data=bias_wight, requires_grad=True)
    # self.bias = Variable(bias_wight.type(dtype), requires_grad=True)
    self.weights.append(self.bias)

  def forward(self, x, length):
    x = torch.mul(self.affine.expand_as(x),x)*length/200
    x = x+self.bias.expand_as(x)
    return x


class Conv1dBrc(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride=1, 
               padding='same', dilation=1, groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False):
    super(Conv1dBrc, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride, (list)): # listed list (to resize the input)
      stride1, stride2 = stride
      if stride1 == 1:
        assert stride2 != 1
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert stride == 1  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## BatchNorm
    self.bn = nn.BatchNorm1d(
      in_dim, eps=bn_eps, momentum=bn_mom, 
      affine=True, track_running_stats=True)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv1d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## Conv2d
    self.conv2 = Conv1d(
      out_dim, out_dim, kernel_size, stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv1d(
          in_dim, out_dim, kernel_size= 1, stride= 1, 
          padding=0, dilation=1, use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv1d(
          in_dim, out_dim, kernel_size= 1, stride=stride_rsz, 
          padding=0, dilation=1, use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## BatchNorm + ReLU
    x = self.activ_func(self.bn(x))
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## Conv
    x = self.conv2(x) # no dropout
    return x + residual






class ConvLSTMCell(nn.Module):

  def __init__(self, input_dim, hidden_dim, kernel_size, bias):
    """
    Initialize ConvLSTM cell.
    Parameters
    ----------
    input_dim: int
      Number of channels of input tensor.
    hidden_dim: int
      Number of channels of hidden state.
    kernel_size: (int, int)
      Size of the convolutional kernel.
    bias: bool
      Whether or not to add the bias.
    """

    super(ConvLSTMCell, self).__init__()
    self.weights = []
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.kernel_size = kernel_size
    self.padding = kernel_size[0] // 2, kernel_size[1] // 2
    self.bias = bias

    self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias)
    self.weights.append(self.conv.weight)

  def forward(self, input_tensor, cur_state):
    h_cur, c_cur = cur_state

    combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

    combined_conv = self.conv(combined)
    cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.tanh(cc_g)

    c_next = f * c_cur + i * g
    h_next = o * torch.tanh(c_next)

    return h_next, c_next

  def init_hidden(self, batch_size, image_size):
    height, width = image_size
    return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

  """
  Parameters:
    input_dim: Number of channels in input
    hidden_dim: Number of hidden channels
    kernel_size: Size of kernel in convolutions
    num_layers: Number of LSTM layers stacked on each other
    batch_first: Whether or not dimension 0 is the batch or not
    bias: Bias or no bias in Convolution
    return_all_layers: Return the list of computations for all layers
    Note: Will do same padding.
  Input:
    A tensor of size B, T, C, H, W or T, B, C, H, W
  Output:
    A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
      0 - layer_output_list is the list of lists of length T of each output
      1 - last_state_list is the list of last states
          each element of the list is a tuple (h, c) for hidden state and memory
  Example:
    >> x = torch.rand((32, 10, 64, 128, 128))
    >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
    >> _, last_states = convlstm(x)
    >> h = last_states[0][0]  # 0 for layer index, 0 for h index
  """

  def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
         batch_first=False, bias=True, return_all_layers=False):
    super(ConvLSTM, self).__init__()

    self._check_kernel_size_consistency(kernel_size)

    # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
    kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
    hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
    if not len(kernel_size) == len(hidden_dim) == num_layers:
      raise ValueError('Inconsistent list length.')
    self.weights = []
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.bias = bias
    self.return_all_layers = return_all_layers

    cell_list = []
    for i in range(0, self.num_layers):
      cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
      cell = ConvLSTMCell(input_dim=cur_input_dim,
                      hidden_dim=self.hidden_dim[i],
                      kernel_size=self.kernel_size[i],
                      bias=self.bias)
      cell_list.append(cell)
      self.weights.extend(cell.weights)

    self.cell_list = nn.ModuleList(cell_list)

  def forward(self, input_tensor, hidden_state=None):
    """
    Parameters
    ----------
    input_tensor: todo
      5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
    hidden_state: todo
      None. todo implement stateful
    Returns
    -------
    last_state_list, layer_output
    """
    if not self.batch_first:
      # (t, b, c, h, w) -> (b, t, c, h, w)
      input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

    b, _, _, h, w = input_tensor.size()

    # Implement stateful ConvLSTM
    if hidden_state is not None:
      # raise NotImplementedError()
      hidden_state = hidden_state
    else:
      # Since the init is done in forward. Can send image size here
      hidden_state = self._init_hidden(batch_size=b,
                       image_size=(h, w))

    layer_output_list = []
    last_state_list = []

    seq_len = input_tensor.size(1)
    cur_layer_input = input_tensor

    for layer_idx in range(self.num_layers):

      h, c = hidden_state[layer_idx]
      output_inner = []
      for t in range(seq_len):
        h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                         cur_state=[h, c])
        output_inner.append(h)

      layer_output = torch.stack(output_inner, dim=1)
      cur_layer_input = layer_output

      layer_output_list.append(layer_output)
      last_state_list.append([h, c])

    if not self.return_all_layers:
      layer_output_list = layer_output_list[-1:]
      last_state_list = last_state_list[-1:]

    return layer_output_list, last_state_list

  def _init_hidden(self, batch_size, image_size):
    init_states = []
    for i in range(self.num_layers):
      init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
    return init_states

  @staticmethod
  def _check_kernel_size_consistency(kernel_size):
    if not (isinstance(kernel_size, tuple) or
        (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
      raise ValueError('`kernel_size` must be tuple or list of tuples')

  @staticmethod
  def _extend_for_multilayer(param, num_layers):
    if not isinstance(param, list):
      param = [param] * num_layers
    return param

class Att_FusionModule(nn.Module):
  def __init__(self, input_dim, mode="UP", reduction_ratio=4):
    super(Att_FusionModule, self).__init__()

    self.weights = []
    bottle_dim = int(input_dim/reduction_ratio)
    self.linear_afm1 = Conv2d(input_dim*2, bottle_dim, kernel_size=[1,1], stride=(1,1), apply_bn=True, activ="ReLU", use_bias=False)
    self.linear_afm2 = Conv2d(bottle_dim, input_dim, kernel_size=[1,1], stride=(1,1), apply_bn=True, activ="Tanh", use_bias=False)

    self.weights.extend(self.linear_afm1.weights)
    self.weights.extend(self.linear_afm2.weights)

    self.mode = mode
    self.linear_bmfa1 = Conv2d(input_dim, input_dim, kernel_size=[1,1], stride=(1,1), apply_bn=True, activ=None, use_bias=False)
    
    if mode == "UP":
      self.linear_bmfa2 = Conv2d(input_dim*2, input_dim, kernel_size=[1,1], stride=(1,1), apply_bn=True, activ=None, use_bias=False)
      self.sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    else:
      self.sampling = Conv2d(int(input_dim/2), input_dim, kernel_size=[3,3], stride=(2,2), apply_bn=False, activ=None, use_bias=False)
      self.weights.extend(self.sampling.weights)
      self.linear_bmfa2 = Conv2d(input_dim, input_dim, kernel_size=[1,1], stride=(1,1), apply_bn=True, activ=None, use_bias=False)
    
    self.weights.extend(self.linear_bmfa1.weights)
    self.weights.extend(self.linear_bmfa2.weights)

  def forward(self, x1, x2, channel_axis=-3):
    
    x2 = self.linear_bmfa1(x2)
    if self.mode == "UP":
      x1 = self.linear_bmfa2(x1)
      x1 = self.sampling(x1)
      if x1.size(-1) != x2.size(-1):
        x1 = x1[:,:,:,:-1]
      if x1.size(-2) != x2.size(-2):
        x1 = x1[:,:,:-1,:]  
    else:
      x1 = self.sampling(x1)
      x1 = self.linear_bmfa2(x1)

    att = self.linear_afm2(self.linear_afm1(torch.cat((x1, x2), dim=channel_axis)))
    output = torch.mul(torch.add(att,1),x1) + torch.mul(torch.add(1,torch.mul(att, -1)),x2)

    return output



class GraphAttentionLayer(nn.Module):
  """
  https://github.com/Diego999/pyGAT/blob/master/layers.py
  Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
  """
  def __init__(self, in_features, out_features, dropout, alpha, concat=True):
    super(GraphAttentionLayer, self).__init__()
    self.weights = []
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = alpha
    self.concat = concat

    W_wight = torch.empty(size=(in_features, out_features))
    nn.init.xavier_uniform_(W_wight, gain=1.414)
    self.W = nn.Parameter(data = W_wight, requires_grad=True)

    a_wight = torch.empty(size=(2 * out_features, 1))
    nn.init.xavier_uniform_(a_wight, gain=1.414)
    self.a = nn.Parameter(data = a_wight, requires_grad=True)
    self.weights.append(self.W)
    self.weights.append(self.a)
    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, h, adj):
    """

    :param h: (batch_zize, number_nodes, in_features)
    :param adj: (batch_size, number_nodes, number_nodes)
    :return: (batch_zize, number_nodes, out_features)
    """
    # batchwise matrix multiplication
    # (batch_zize, number_nodes, in_features) * (in_features, out_features)
    # -> (batch_zize, number_nodes, out_features)
    # print(h.size())
    Wh = torch.matmul(h, self.W)
    # print(Wh.size())
    # (batch_zize, number_nodes, number_nodes, 2 * out_features)
    a_input = self.batch_prepare_attentional_mechanism_input(Wh)
    # (batch_zize, number_nodes, number_nodes, 2 * out_features) * (2 * out_features, 1)
    # -> (batch_zize, number_nodes, number_nodes, 1)
    # print(a_input.size())
    e = torch.matmul(a_input, self.a)
    # print(e.size())
    # (batch_zize, number_nodes, number_nodes)
    e = self.leakyrelu(e.squeeze(-1))
    # (batch_zize, number_nodes, number_nodes)
    zero_vec = -9e15 * torch.ones_like(e)

    # (batch_zize, number_nodes, number_nodes)
    attention = torch.where(adj > 0, e, zero_vec)
    # print(attention.size())
    # (batch_zize, number_nodes, number_nodes)
    attention = F.softmax(attention, dim=-1)

    # (batch_zize, number_nodes, number_nodes)
    attention = F.dropout(attention, self.dropout, training=self.training)

    # batched matrix multiplication (batch_zize, number_nodes, out_features)
    h_prime = torch.matmul(attention, Wh)
    # print(h_prime.size())

    if self.concat:
      return F.elu(h_prime)
    else:
      return h_prime

  def batch_prepare_attentional_mechanism_input(self, Wh):
    """
    with batch training
    :param Wh: (batch_zize, number_nodes, out_features)
    :return:
    """
    B, M, E = Wh.shape # (batch_zize, number_nodes, out_features)
    Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)  # (B, M*M, E)
    Wh_repeated_alternating = Wh.repeat(1, M, 1)  # (B, M*M, E)
    all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)  # (B, M*M,2E)
    return all_combinations_matrix.view(B, M, M, 2 * E)

  def _prepare_attentional_mechanism_input(self, Wh_n):
    """
    no batch dimension
    :param Wh_n:
    :return:
    """
    M = Wh_n.size()[0]  # number of nodes（M, E)
    Wh_repeated_in_chunks = Wh_n.repeat_interleave(M, dim=0)  # (M, M, E)
    Wh_repeated_alternating = Wh_n.repeat(M, 1)  # (M, M, E)
    all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)  # (M*M,2E)
    return all_combinations_matrix.view(M, M, 2 * self.out_features)  # （M, M, 2E)

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def preprocess_adj(A):
  '''
  Pre-process adjacency matrix
  :param A: adjacency matrix
  :return:
  '''
  I = np.eye(A.shape[0])
  A_hat = A + I # add self-loops
  D_hat_diag = np.sum(A_hat, axis=1)
  D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
  D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
  D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
  return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

def preprocess_adj_tensor(A):
  '''
  Pre-process adjacency matrix
  :param A: adjacency matrix
  :return:
  '''
  I = torch.eye(A.size()[1], requires_grad=False).to(A.device)
  A_hat = A + I # add self-loops
  D_hat_diag = torch.sum(A_hat, axis=-1)
  D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, -0.5)
  D_hat_diag_inv_sqrt[torch.isinf(D_hat_diag_inv_sqrt)] = 0.
  D_hat_inv_sqrt = torch.diag(D_hat_diag_inv_sqrt)
  
  return torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

class GCNLayer(nn.Module):
  def __init__(self, in_dim, out_dim, acti=True):
    super(GCNLayer, self).__init__()
    self.weights = []
    self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
    self.weights.append(self.linear.weight)
    if acti:
      self.acti = nn.ReLU(inplace=True)
    else:
      self.acti = None
  def forward(self, F):
    output = self.linear(F)
    if not self.acti:
      return output
    return self.acti(output)

class GCN(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_classes, droprate=0.0):
    super(GCN, self).__init__()
    self.weights = []
    self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
    self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
    self.weights.extend(self.gcn_layer1.weights)
    self.weights.extend(self.gcn_layer2.weights)    
    self.dropout = nn.Dropout(droprate)

  def forward(self, Adj, X):
    # A = torch.from_numpy(preprocess_adj(A)).float() ## numpy
    # print(Adj)
    A = preprocess_adj_tensor(Adj.clone())
    # print(A)
    # exit()
    A = A.expand(X.size()[0],-1,-1) ## tensor
    X = self.dropout(X.float())
    F = torch.bmm(A, X)
    F = self.gcn_layer1(F)
    F = self.dropout(F)
    F = torch.bmm(A, F)
    output = self.gcn_layer2(F)
    return output


from einops import rearrange

def exists(val):
  return val is not None

def default(val, d):
  return val if exists(val) else d

def calc_rel_pos(n):
  pos = torch.meshgrid(torch.arange(n), torch.arange(n))
  pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
  rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
  rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
  return rel_pos
    
class LambdaLayer(nn.Module):
  def __init__(self, dim, *, dim_k, n = None, r = None, heads = 4, dim_out = None, dim_u = 1):
    super().__init__()
    self.weights = []
    dim_out = default(dim_out, dim)
    self.u = dim_u # intra-depth dimension
    self.heads = heads

    assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
    dim_v = dim_out // heads

    self.to_q = Conv2d(
      dim, dim_k * heads, kernel_size=(1, 1), use_bias=False,
      apply_bn=True, activ=None, as_class=False)
    self.to_k = Conv2d(
      dim, dim_k * dim_u, kernel_size=(1, 1), use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.to_v = Conv2d(
      dim, dim_v * dim_u, kernel_size=(1, 1), use_bias=False,
      apply_bn=True, activ=None, as_class=False)

    self.weights.extend(self.to_q.weights)
    self.weights.extend(self.to_k.weights)
    self.weights.extend(self.to_v.weights)

    self.local_contexts = exists(r)
    if exists(r):
      assert (r % 2) == 1, 'Receptive kernel size should be odd'
      self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
      self.weights.append(self.pos_conv.weight)
    else:
      assert exists(n), 'You must specify the window size (n=h=w)'
      rel_lengths = 2 * n - 1
      self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
      self.rel_pos = calc_rel_pos(n)
      self.weights.append(self.rel_pos_emb)

  def forward(self, x):
    b, c, hh, ww, u, h = *x.shape, self.u, self.heads

    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)
    # print(q.size(), k.size(), v.size())
    q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
    k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
    v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

    k = k.softmax(dim=-1)

    λc = torch.einsum('b u k m, b u v m -> b k v', k, v)
    Yc = torch.einsum('b h k n, b k v -> b h v n', q, λc)

    if self.local_contexts:
      v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
      λp = self.pos_conv(v)
      Yp = torch.einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
    else:
      n, m = self.rel_pos.unbind(dim = -1)
      rel_pos_emb = self.rel_pos_emb[n, m]
      λp = torch.einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
      Yp = torch.einsum('b h k n, b n k v -> b h v n', q, λp)

    Y = Yc + Yp
    # print('Yc', Yc.mean())
    # print('Yp', Yp.mean())
    out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
    return out

class Conv2dCbr_lambda(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, 
               d_k, heads, d_u, r, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
               activ=None, kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False):
    super(Conv2dCbr_lambda, self).__init__()
    self.activ, self.weights = [], []

    assert activ is not None and apply_bn is True

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    ## lambda_layer
    ## the receptive field for relative positional encoding (23 x 23)
    self.lambda_layer = LambdaLayer(dim = out_dim, dim_out = out_dim,
        r = r, dim_k = d_k, heads = heads, dim_u = d_u)
    self.weights.extend(self.lambda_layer.weights)

    ## Conv2d + BatchNorm
    self.conv2 = Conv2d(
      out_dim, out_dim, [1, 1], stride2, 
      padding, dilation, groups, use_bias, 
      apply_bn=apply_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## Conv + BatchNorm + ReLU
    x = self.conv1(x, drop=drop)
    ## lambda_layer
    x = self.lambda_layer(x)
    ## Conv + BatchNorm
    x = self.conv2(x) # no dropout
    ## ReLU
    return self.activ_func(x + residual)

class LW_LambdaLayer(nn.Module):
  def __init__(self, dim, *, dim_k, n = None, r = None, heads = 4, dim_out = None, dim_u = 1, feat_dim=64):
    super(LW_LambdaLayer, self).__init__()
    self.weights = []
    dim_out = default(dim_out, dim)
    self.u = dim_u # intra-depth dimension
    self.heads = heads

    assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
    dim_v = dim_out // heads

    self.DWConv = Conv2d(
      dim, dim, kernel_size=(2, 2), stride=(2, 2), groups=dim, use_bias=True,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv.weights)

    self.to_q = Conv2d(
      dim, dim_k * heads, kernel_size=(1, 1), use_bias=False,
      apply_bn=True, activ=None, as_class=False)
    self.to_k = Conv2d(
      dim, dim_k * dim_u, kernel_size=(1, 1), use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.to_v = Conv2d(
      dim, dim_v * dim_u, kernel_size=(1, 1), use_bias=False,
      apply_bn=True, activ=None, as_class=False)

    self.weights.extend(self.to_q.weights)
    self.weights.extend(self.to_k.weights)
    self.weights.extend(self.to_v.weights)

    self.up_sample = nn.Upsample(scale_factor=2)
    

    self.local_contexts = exists(r)
    if self.local_contexts:
      assert (r % 2) == 1, 'Receptive kernel size should be odd'
      self.embedding = nn.Parameter(torch.randn([dim_k, dim_u, 1, r, r]), requires_grad=True)
    else:
      self.embedding = nn.Parameter(torch.randn([dim_k, dim_u]), requires_grad=True)
    self.weights.append(self.embedding)    

    self.kk = dim_k
    self.uu = dim_u
    self.padding = r // 2

    b_wight1 = torch.empty(size=(1, dim_k, dim_v))
    nn.init.xavier_normal_(b_wight1, gain=1.0)
    self.B1 = nn.Parameter(data = b_wight1, requires_grad=True)
    self.weights.append(self.B1)

    b_wight2 = torch.empty(size=(1, dim_k, dim_v, feat_dim))
    nn.init.xavier_normal_(b_wight2, gain=1.0)
    self.B2 = nn.Parameter(data = b_wight2, requires_grad=True)
    self.weights.append(self.B2)


  def forward(self, x):
    b, c, hh, ww, u, h = *x.shape, self.u, self.heads

    q = self.to_q(x)

    d = self.DWConv(x)
    hhh, www = d.size(-2), d.size(-1)

    k = self.to_k(d)
    v = self.to_v(d)
    # print(q.size(), k.size(), v.size())
    q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
    k = rearrange(k, 'b (u k) hhh www -> b u k (hhh www)', u = u)
    v = rearrange(v, 'b (u v) hhh www -> b u v (hhh www)', u = u)

    k = k.softmax(dim=-1)

    λc = torch.einsum('b u k m, b u v m -> b k v', k, v)
    λc = λc + self.B1.expand_as(λc)
    Yc = torch.einsum('b h k n, b k v -> b h v n', q, λc)


    if self.local_contexts:

      v = rearrange(v, 'b u v (hhh www) -> b u v hhh www', hhh = hhh, www = www)
      λp = F.conv3d(v, self.embedding, padding=(0, self.padding, self.padding))
      λp = rearrange(λp, 'b k v hhh www -> b (k v) hhh www')
      λp = self.up_sample(λp)

      diffY = hh - λp.size()[-2]
      diffX = ww - λp.size()[-1]

      λp = F.pad(λp, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
      λp = rearrange(λp, 'b (k v) hh ww -> b k v (hh ww)', k = self.kk)

      Bias = torch.repeat_interleave(self.B2, ww, dim=-1)
      λp = λp + Bias.expand_as(λp)
      Yp = torch.einsum('b h k n, b k v n -> b h v n', q, λp)
    else:
      λp = torch.einsum('ku,bvun->bkvn', self.embedding, values)
      Yp = torch.einsum('bhkn,bkvn->bhvn', q, λp)


    Y = Yc + Yp
    out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
    return out

class SubSpectralNorm(nn.Module):
  def __init__(self, C, S, eps=1e-6):
    super(SubSpectralNorm, self).__init__()
    self.S = S
    self.bn = nn.BatchNorm2d(C*S, eps=eps, affine=True, track_running_stats=True)

  def forward(self, x):
    # x: input features with shape {N, C, F, T}
    # S: number of sub-bands
    N, C, F, T = x.size()
    x = x.view(N, C * self.S, F // self.S, T)

    x = self.bn(x)

    return x.view(N, C, F, T)

class BC_Conv(nn.Module):
  def __init__(self, in_dim, out_dim, stride=(1, 1),
    padding='same', dilation=(1, 1), use_bias=True, feat_dim=64):
    super(BC_Conv, self).__init__()
    ## broadcasted residual learning
    self.weights = []
    self.freq_dw_conv = Conv2d(
      in_dim, in_dim, (3, 1), stride, 
      padding, dilation, groups=in_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.freq_dw_conv.weights)

    self.ssn = SubSpectralNorm(in_dim, feat_dim//8)

    self.temp_dw_conv = Conv2d(
      in_dim, in_dim, (1, 3), stride, 
      padding, dilation, groups=in_dim, use_bias=use_bias,
      apply_bn=False, activ="GELU", as_class=False)
    self.weights.extend(self.temp_dw_conv.weights)

    self.bn = nn.BatchNorm2d(in_dim)

    self.conv1 = Conv2d(
      in_dim, out_dim, (1, 1), (1, 1), 
      padding, dilation, groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.conv1.weights)

    self.acti = nn.GELU()

    # self.ln = nn.GroupNorm(in_dim, in_dim)
    self.ln = nn.GroupNorm(1, in_dim)

  def forward(self, x, drop=0.0, freq_axis=-2, time_axis=-1):
    x = self.ln(x)
    identity = x

    # f2
    ##########################
    out = self.ssn(self.freq_dw_conv(x))
    ##########################

    auxilary = out
    out = out.mean(freq_axis, keepdim=True)  # frequency average pooling

    # f1
    ############################
    out = self.bn(self.temp_dw_conv(out))
    out = self.conv1(out, drop)
    ############################

    out = out + identity + auxilary
    out = self.acti(out)

    return out


class BC_Conv_time(nn.Module):
  def __init__(self, in_dim, out_dim, stride=(1, 1),
    padding='same', dilation=(1, 1), use_bias=True, feat_dim=64):
    super(BC_Conv_time, self).__init__()
    ## broadcasted residual learning
    self.weights = []
    self.freq_dw_conv = Conv2d(
      in_dim, in_dim, (3, 1), stride, 
      padding, dilation, groups=in_dim, use_bias=use_bias,
      apply_bn=False, activ="GELU", as_class=False)
    self.weights.extend(self.freq_dw_conv.weights)

    self.ssn = SubSpectralNorm(in_dim, feat_dim//8)

    self.temp_dw_conv = Conv2d(
      in_dim, in_dim, (1, 3), stride, 
      padding, dilation, groups=in_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.temp_dw_conv.weights)

    self.bn = nn.BatchNorm2d(in_dim)

    self.conv1 = Conv2d(
      in_dim, out_dim, (1, 1), (1, 1), 
      padding, dilation, groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.conv1.weights)

    self.acti = nn.GELU()

    # self.ln = nn.GroupNorm(in_dim, in_dim)
    self.ln = nn.GroupNorm(1, in_dim)

  def forward(self, x, drop=0.0, freq_axis=-2, time_axis=-1):
    x = self.ln(x)
    identity = x

    # f2
    ##########################
    out = self.bn(self.temp_dw_conv(out))
    ##########################

    auxilary = out
    out = out.mean(time_axis, keepdim=True)  # frequency average pooling

    # f1
    ############################
    out = self.ssn(self.freq_dw_conv(out))
    out = self.conv1(out, drop)
    ############################

    out = out + identity + auxilary
    out = self.acti(out)

    return out

class BC_Conv_expand(nn.Module):
  def __init__(self, in_dim, out_dim, R_value, stride=(1, 1),
    padding='same', dilation=(1, 1), use_bias=True, feat_dim=64):
    super(BC_Conv_expand, self).__init__()
    ## broadcasted residual learning
    exp_dim = int(in_dim * R_value)
    self.weights = []
    self.conv1 = Conv2d(
      in_dim, exp_dim, (1, 1), stride, 
      padding, dilation, groups=1, use_bias=use_bias,
      apply_bn=False, activ="GELU", as_class=False)
    self.weights.extend(self.conv1.weights)

    self.bn1 = nn.BatchNorm2d(exp_dim, affine=True, track_running_stats=True)

    self.freq_dw_conv = Conv2d(
      exp_dim, exp_dim, (3, 1), stride, 
      padding, dilation, groups=exp_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.freq_dw_conv.weights)

    self.ssn = SubSpectralNorm(exp_dim, feat_dim//8)

    self.temp_dw_conv = Conv2d(
      exp_dim, exp_dim, (1, 3), stride, 
      padding, dilation, groups=exp_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.temp_dw_conv.weights)

    self.bn2 = nn.BatchNorm2d(exp_dim)

    self.conv2 = Conv2d(
      exp_dim, out_dim, (1, 1), (1, 1), 
      padding, dilation, groups=1, use_bias=False,
      apply_bn=True, activ=None, as_class=False)
    self.weights.extend(self.conv2.weights)

    self.acti = nn.GELU()

    # self.ln = nn.GroupNorm(in_dim, in_dim)
    self.ln = nn.GroupNorm(1, in_dim)

  def forward(self, x, drop=0.0, freq_axis=-2, time_axis=-1):
    x = self.ln(x)

    # f2
    ##########################
    out = self.bn1(self.conv1(x, drop))
    identity = out
    out = self.ssn(self.freq_dw_conv(out))
    ##########################

    auxilary = out
    out = out.mean(freq_axis, keepdim=True)  # frequency average pooling

    # f1
    ############################
    out = self.temp_dw_conv(out)
    out = self.bn2(self.acti(out + auxilary + identity))
    out = self.conv2(out, drop)
    ############################

    return out


class BC_Conv_expand_time(nn.Module):
  def __init__(self, in_dim, out_dim, R_value, stride=(1, 1),
    padding='same', dilation=(1, 1), use_bias=True, feat_dim=64):
    super(BC_Conv_expand_time, self).__init__()
    ## broadcasted residual learning
    exp_dim = int(in_dim * R_value)
    self.weights = []
    self.conv1 = Conv2d(
      in_dim, exp_dim, (1, 1), stride, 
      padding, dilation, groups=1, use_bias=use_bias,
      apply_bn=False, activ="GELU", as_class=False)
    self.weights.extend(self.conv1.weights)

    self.bn1 = nn.BatchNorm2d(exp_dim, affine=True, track_running_stats=True)

    self.freq_dw_conv = Conv2d(
      exp_dim, exp_dim, (3, 1), stride, 
      padding, dilation, groups=exp_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.freq_dw_conv.weights)

    self.bn3 = nn.BatchNorm2d(exp_dim)

    self.temp_dw_conv = Conv2d(
      exp_dim, exp_dim, (1, 3), stride, 
      padding, dilation, groups=exp_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.temp_dw_conv.weights)

    self.bn2 = nn.BatchNorm2d(exp_dim)

    self.conv2 = Conv2d(
      exp_dim, out_dim, (1, 1), (1, 1), 
      padding, dilation, groups=1, use_bias=False,
      apply_bn=True, activ=None, as_class=False)
    self.weights.extend(self.conv2.weights)

    self.acti = nn.GELU()

    # self.ln = nn.GroupNorm(in_dim, in_dim)
    self.ln = nn.GroupNorm(1, in_dim)

  def forward(self, x, drop=0.0, freq_axis=-2, time_axis=-1):
    x = self.ln(x)

    # f2
    ##########################
    out = self.bn1(self.conv1(x, drop))
    identity = out
    out = self.bn3(self.temp_dw_conv(out))
    ##########################

    auxilary = out
    out = out.mean(time_axis, keepdim=True)  # frequency average pooling

    # f1
    ############################
    out = self.freq_dw_conv(out)
    out = self.bn2(self.acti(out + auxilary + identity))
    out = self.conv2(out, drop)
    ############################

    return out

class BC_Conv_expand_attn2(nn.Module):
  def __init__(self, in_dim, out_dim, R_value, stride=(1, 1),
    padding='same', dilation=(1, 1), use_bias=True, feat_dim=64):
    super(BC_Conv_expand_attn2, self).__init__()
    ## broadcasted residual learning
    exp_dim = int(in_dim * R_value)
    self.weights = []
    self.conv1 = Conv2d(
      in_dim, exp_dim, (1, 1), stride, 
      padding, dilation, groups=1, use_bias=use_bias,
      apply_bn=False, activ="GELU", as_class=False)
    self.weights.extend(self.conv1.weights)

    self.bn1 = nn.BatchNorm2d(exp_dim, affine=True, track_running_stats=True)

    self.freq_dw_conv = Conv2d(
      exp_dim, exp_dim, (3, 1), stride, 
      padding, dilation, groups=exp_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.freq_dw_conv.weights)

    self.ssn = SubSpectralNorm(exp_dim, feat_dim//8)

    self.temp_dw_conv = Conv2d(
      exp_dim, exp_dim, (1, 3), stride, 
      padding, dilation, groups=exp_dim, use_bias=use_bias,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.temp_dw_conv.weights)

    self.bn2 = nn.BatchNorm2d(exp_dim)

    self.conv2 = Conv2d(
      exp_dim, out_dim, (1, 1), (1, 1), 
      padding, dilation, groups=1, use_bias=False,
      apply_bn=True, activ=None, as_class=False)
    self.weights.extend(self.conv2.weights)

    self.acti = nn.GELU()

    # self.ln = nn.GroupNorm(in_dim, in_dim)
    self.ln = nn.GroupNorm(1, in_dim)

    self.attn = Dense(feat_dim, feat_dim, use_bias=False, apply_bn=False, activ=None)
    self.weights.extend(self.attn.weights)


  def forward(self, x, drop=0.0, freq_axis=-2, time_axis=-1):
    x = self.ln(x)

    # f2
    ##########################
    out = self.bn1(self.conv1(x, drop))
    identity = out
    out = self.ssn(self.freq_dw_conv(out))
    ##########################

    auxilary = out
    attn = self.attn(torch.mean(out, dim=(-3,time_axis)))
    attn = F.softmax(attn, dim=-1).unsqueeze(-1).unsqueeze(-3)

    out = attn*out
    out = out.sum(freq_axis, keepdim=True)  # frequency average pooling

    # f1
    ############################
    out = self.temp_dw_conv(out)
    out = self.bn2(self.acti(out + auxilary + identity))
    out = self.conv2(out, drop)
    ############################

    return out

# class LPU(nn.Module):
#   """
#   Local Perception Unit to extract local infomation.
#   LPU(X) = DWConv(X) + X
#   """
#   def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), 
#                padding='same', dilation=(1, 1), use_bias=True, 
#                w_init_gain='conv2d', as_class=False):
#     super(LPU, self).__init__()
#     self.weights = []
#     self.DWConv1 = Conv2d(
#       in_dim, out_dim, (3, 1), (1, 1), 
#       padding, dilation, groups=in_dim, use_bias=use_bias, 
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.DWConv1.weights)

#     self.DWConv2 = Conv2d(
#       in_dim, out_dim, (1, 3), (1, 1), 
#       padding, dilation, groups=in_dim, use_bias=use_bias, 
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.DWConv2.weights)
    
#   def forward(self, x, drop=0.0):
#     k = self.DWConv1(x, drop)
#     return self.DWConv2(k, drop) + x + k

class LPU(nn.Module):
  """
  Local Perception Unit to extract local infomation.
  LPU(X) = DWConv(X) + X
  """
  def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), 
               padding='same', dilation=(1, 1), use_bias=True, 
               w_init_gain='conv2d', as_class=False):
    super(LPU, self).__init__()
    self.weights = []
    self.DWConv = Conv2d(
      in_dim, out_dim, kernel_size, stride, 
      padding, dilation, groups=in_dim, use_bias=use_bias, 
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv.weights)
    
  def forward(self, x, drop=0.0):
    return self.DWConv(x, drop) + x

# class LPU_Mod(nn.Module):
#   """
#   Local Perception Unit to extract local infomation.
#   LPU_Mod(X) = DWConv(X) + X
#   """
#   def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, 
#                padding='same', dilation=1, use_bias=True, 
#                w_init_gain='conv1d', as_class=False, feat_dim=80):
#     super(LPU_Mod, self).__init__()
#     self.weights = []
#     self.DWConv = Conv1d(
#       in_dim*feat_dim, out_dim*feat_dim, kernel_size, stride, 
#       padding, dilation, groups=in_dim*feat_dim, use_bias=use_bias, 
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.DWConv.weights)
    
#   def forward(self, x, drop=0.0):
#     b, c, f, t = x.shape
#     return self.DWConv(x.view(b, c*f, t), drop).view(b, c, f, t) + x

class LPU_Mod(nn.Module):
  """
  Local Perception Unit to extract local infomation.
  LPU_Mod(X) = DWConv(X) + X
  """
  def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), 
               padding='same', dilation=(1, 1), use_bias=True, 
               w_init_gain='conv2d', as_class=False, feat_dim=80):
    super(LPU_Mod, self).__init__()
    self.weights = []
    self.DWConv1 = Conv2d(
      in_dim, out_dim, (3, 1), (1, 1), 
      padding, dilation, groups=in_dim, use_bias=use_bias, 
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv1.weights)

    self.DWConv2 = Conv2d(
      out_dim, out_dim, (1, 3), (1, 1), 
      padding, dilation, groups=out_dim, use_bias=use_bias, 
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv2.weights)
    
  def forward(self, x, drop=0.0):
    k = self.DWConv1(x, drop)
    return self.DWConv2(k, drop) + x + k

class LMHSA(nn.Module):
  """
  Lightweight Multi-head-self-attention module.
  Inputs:
      Q: [N, C, H, W]
      K: [N, C, H / stride, W / stride]
      V: [N, C, H / stride, W / stride]
  Outputs:
      X: [N, C, H, W]
  """
  def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=64):
    super(LMHSA, self).__init__()
    self.weights = []
    self.DWConv = Conv2d(
      channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv.weights)
    # self.DWConv = Conv2d(
    #   channels, channels, kernel_size=stride, stride=stride, groups=1, use_bias=True,
    #   apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv.weights)
    # self.DWConv_k = Conv2d(
    #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
    #   apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv_k.weights)

    # self.DWConv_v = Conv2d(
    #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
    #   apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv_v.weights)

    self.fc_q = Dense(channels, heads * d_k, use_bias=False, apply_bn=False, activ=None)
    self.fc_k = Dense(channels, heads * d_k, use_bias=False, apply_bn=False, activ=None)
    self.fc_v = Dense(channels, heads * d_v, use_bias=False, apply_bn=False, activ=None)
    self.fc_o = Dense(heads * d_k, channels, use_bias=False, apply_bn=False, activ=None)

    self.weights.extend(self.fc_q.weights)
    self.weights.extend(self.fc_k.weights)
    self.weights.extend(self.fc_v.weights)
    self.weights.extend(self.fc_o.weights)

    self.channels = channels
    self.d_k = d_k
    self.d_v = d_v
    self.heads = heads
    self.scaled_factor = self.d_k ** -0.5

    b_wight = torch.empty(size=(1, self.heads, feat_dim , feat_dim//stride[1]))
    nn.init.xavier_normal_(b_wight, gain=1.0)
    self.B = nn.Parameter(data = b_wight, requires_grad=True)
    self.weights.append(self.B)

    # self.ln = nn.GroupNorm(channels, channels)
    self.ln = nn.GroupNorm(1, channels)

  def forward(self, x, drop=0.0):
    b, c, f, t = x.shape
    x = self.ln(x)
    # Get q, k, v
    # Reshape
    q = x.permute(0, 2, 3, 1).contiguous() # [b, f, t, c]
    # exit()
    q = self.fc_q(q)
    q = q.view(b, f, t, self.heads, self.d_k).permute(0, 3, 1, 2, 4).contiguous()  # [b, heads, f, t, d_k]

    k = self.DWConv(x)
    v = k
    # k = self.DWConv_k(x, drop)
    k_b, k_c, k_f, k_t = k.shape
    k = k.permute(0, 2, 3, 1).contiguous()
    k = self.fc_k(k)
    k = k.view(k_b, k_f, k_t, self.heads, self.d_k).permute(0, 3, 1, 2, 4).contiguous()  # [b, heads, k_f, k_t, d_k]

    # v = self.DWConv_v(x, drop)
    v_b, v_c, v_f, v_t = v.shape
    v = v.permute(0, 2, 3, 1).contiguous()
    v = self.fc_v(v)
    v = v.view(v_b, v_f, v_t, self.heads, self.d_v).permute(0, 3, 1, 2, 4).contiguous() # [b, heads, v_f, v_t, d_v]

    # Attention
    attn = torch.einsum('... x y j, ... a b j -> ... x y a b', q, k) * self.scaled_factor # [b, heads, f, t k_f, k_t]
    attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t = attn.shape
    attn = attn.view(attn_b, attn_head, attn_f * attn_t, attn_k_f * attn_k_t)

    Bias = torch.repeat_interleave(torch.repeat_interleave(self.B, attn_t, dim=-2), attn_k_t, dim=-1)
    attn = attn + Bias.expand_as(attn)
    attn = torch.softmax(attn, dim = -1) # [b, heads, f *t, k_f * k_t]
    attn = attn.view(attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t) # [b, heads, f, t k_f, k_t]

    result = torch.einsum('... a b x y, ... x y c -> ... a b c', attn, v).permute(0, 2, 3, 1, 4).contiguous() # [b, f, t, heads, d_k]

    result = result.view(b, f, t, self.heads * self.d_v)
    result = self.fc_o(result, drop).permute(0, 3, 1, 2).contiguous()

    return result

class LMHSA2(nn.Module):
  """
  Lightweight Multi-head-self-attention module.
  Inputs:
      Q: [N, C, H, W]
      K: [N, C, H / stride, W / stride]
      V: [N, C, H / stride, W / stride]
  Outputs:
      X: [N, C, H, W]
  """
  def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=64):
    super(LMHSA2, self).__init__()
    self.weights = []
    self.DWConv = Conv2d(
      channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv.weights)
    # self.DWConv = Conv2d(
    #   channels, channels, kernel_size=stride, stride=stride, groups=1, use_bias=True,
    #   apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv.weights)
    # self.DWConv_k = Conv2d(
    #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
    #   apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv_k.weights)

    # self.DWConv_v = Conv2d(
    #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
    #   apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv_v.weights)

    self.fc_q = Conv2d(
      channels, heads * d_k, kernel_size=(3, 1), stride=(1, 1), groups=channels, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_k = Conv2d(
      channels, heads * d_k, kernel_size=(3, 1), stride=(1, 1), groups=channels, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_v = Conv2d(
      channels, heads * d_v, kernel_size=(3, 1), stride=(1, 1), groups=channels, use_bias=False,
      apply_bn=False, activ=None, as_class=False)

    self.fc_o = Conv2d(
      heads * d_k, channels, kernel_size=(1, 1), stride=(1, 1), groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)

    self.weights.extend(self.fc_q.weights)
    self.weights.extend(self.fc_k.weights)
    self.weights.extend(self.fc_v.weights)
    self.weights.extend(self.fc_o.weights)

    self.channels = channels
    self.d_k = d_k
    self.d_v = d_v
    self.heads = heads
    self.scaled_factor = self.d_k ** -0.5

    b_wight = torch.empty(size=(1, self.heads, feat_dim , feat_dim//stride[1]))
    nn.init.xavier_normal_(b_wight, gain=1.0)
    self.B = nn.Parameter(data = b_wight, requires_grad=True)
    self.weights.append(self.B)

  # self.ln = nn.GroupNorm(channels, channels)
    self.ln = nn.GroupNorm(1, channels)

  def forward(self, x, drop=0.0):
    b, c, f, t = x.shape
    d_k, d_v, h = self.d_k, self.d_v, self.heads
    x = self.ln(x)

    # Get q, k, v
    # exit()
    q = self.fc_q(x) # [b, heads*d_k, f, t]
    q = q.view(b, h, d_k, f, t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, f, t, d_k]

    k = self.DWConv(x)
    v = k
    # k = self.DWConv_k(x, drop)
    _, _, k_f, k_t = k.shape
    k = self.fc_k(k)
    k = k.view(b, h, d_k, k_f, k_t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, k_f, k_t, d_k]

    # v = self.DWConv_v(x, drop)
    _, _, v_f, v_t = v.shape
    v = self.fc_v(v)
    v = v.view(b, h, d_v, v_f, v_t).permute(0, 1, 3, 4, 2).contiguous() # [b, heads, v_f, v_t, d_v]

    # Attention
    attn = torch.einsum('... x y j, ... a b j -> ... x y a b', q, k) * self.scaled_factor # [b, heads, f, t k_f, k_t]
    attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t = attn.shape
    attn = attn.view(attn_b, attn_head, attn_f * attn_t, attn_k_f * attn_k_t)

    Bias = torch.repeat_interleave(torch.repeat_interleave(self.B, attn_t, dim=-2), attn_k_t, dim=-1)
    attn = attn + Bias.expand_as(attn)
    attn = torch.softmax(attn, dim = -1) # [b, heads, f *t, k_f * k_t]
    attn = attn.view(attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t) # [b, heads, f, t k_f, k_t]

    result = torch.einsum('... a b x y, ... x y c -> ... a b c', attn, v).permute(0, 2, 3, 1, 4).contiguous() # [b, f, t, heads, d_k]

    result = result.view(b, f, t, self.heads * self.d_v)
    # result = self.fc_o(result, drop).permute(0, 3, 1, 2).contiguous()
    result = self.fc_o(result.permute(0, 3, 1, 2).contiguous(), drop)
    return result

# class LMHSA3(nn.Module):
#   """
#   Lightweight Multi-head-self-attention module.
#   Inputs:
#       Q: [N, C, H, W]
#       K: [N, C, H / stride, W / stride]
#       V: [N, C, H / stride, W / stride]
#   Outputs:
#       X: [N, C, H, W]
#   """
#   def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=64):
#     super(LMHSA3, self).__init__()
#     self.weights = []
#     self.DWConv = Conv2d(
#       channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.DWConv.weights)
#     # self.DWConv = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=1, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv.weights)
#     # self.DWConv_k = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv_k.weights)

#     # self.DWConv_v = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv_v.weights)

#     self.fc_q = Conv2d(
#       channels, heads * d_k, kernel_size=(1, 3), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_k = Conv2d(
#       channels, heads * d_k, kernel_size=(1, 3), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_v = Conv2d(
#       channels, heads * d_v, kernel_size=(1, 3), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)

#     self.fc_o = Conv2d(
#       heads * d_k, channels, kernel_size=(1, 1), stride=(1, 1), groups=1, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)

#     self.weights.extend(self.fc_q.weights)
#     self.weights.extend(self.fc_k.weights)
#     self.weights.extend(self.fc_v.weights)
#     self.weights.extend(self.fc_o.weights)

#     self.channels = channels
#     self.d_k = d_k
#     self.d_v = d_v
#     self.heads = heads
#     self.scaled_factor = self.d_k ** -0.5

#     b_wight = torch.empty(size=(1, self.heads, feat_dim , feat_dim//stride[1]))
#     nn.init.xavier_normal_(b_wight, gain=1.0)
#     self.B = nn.Parameter(data = b_wight, requires_grad=True)
#     self.weights.append(self.B)

#   # self.ln = nn.GroupNorm(channels, channels)
#     self.ln = nn.GroupNorm(1, channels)

#   def forward(self, x, drop=0.0):
#     b, c, f, t = x.shape
#     d_k, d_v, h = self.d_k, self.d_v, self.heads
#     x = self.ln(x)

#     # Get q, k, v
#     # exit()
#     q = self.fc_q(x) # [b, heads*d_k, f, t]
#     q = q.view(b, h, d_k, f, t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, f, t, d_k]

#     k = self.DWConv(x)
#     v = k
#     # k = self.DWConv_k(x, drop)
#     _, _, k_f, k_t = k.shape
#     k = self.fc_k(k)
#     k = k.view(b, h, d_k, k_f, k_t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, k_f, k_t, d_k]

#     # v = self.DWConv_v(x, drop)
#     _, _, v_f, v_t = v.shape
#     v = self.fc_v(v)
#     v = v.view(b, h, d_v, v_f, v_t).permute(0, 1, 3, 4, 2).contiguous() # [b, heads, v_f, v_t, d_v]

#     # Attention
#     attn = torch.einsum('... x y j, ... a b j -> ... x y a b', q, k) * self.scaled_factor # [b, heads, f, t k_f, k_t]
#     attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t = attn.shape
#     attn = attn.view(attn_b, attn_head, attn_f * attn_t, attn_k_f * attn_k_t)

#     Bias = torch.repeat_interleave(torch.repeat_interleave(self.B, attn_t, dim=-2), attn_k_t, dim=-1)
#     attn = attn + Bias.expand_as(attn)
#     attn = torch.softmax(attn, dim = -1) # [b, heads, f *t, k_f * k_t]
#     attn = attn.view(attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t) # [b, heads, f, t k_f, k_t]

#     result = torch.einsum('... a b x y, ... x y c -> ... a b c', attn, v).permute(0, 2, 3, 1, 4).contiguous() # [b, f, t, heads, d_k]

#     result = result.view(b, f, t, self.heads * self.d_v)
#     # result = self.fc_o(result, drop).permute(0, 3, 1, 2).contiguous()
#     result = self.fc_o(result.permute(0, 3, 1, 2).contiguous(), drop)
#     return result

# class LMHSA8(nn.Module):
#   """
#   Lightweight Multi-head-self-attention module.
#   Inputs:
#       Q: [N, C, H, W]
#       K: [N, C, H / stride, W / stride]
#       V: [N, C, H / stride, W / stride]
#   Outputs:
#       X: [N, C, H, W]
#   """
#   def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=64):
#     super(LMHSA8, self).__init__()
#     self.weights = []
#     self.DWConv = Conv2d(
#       channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.DWConv.weights)
#     # self.DWConv = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=1, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv.weights)
#     # self.DWConv_k = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv_k.weights)

#     # self.DWConv_v = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv_v.weights)
#     if stride[0] == 1:
#       l = stride[0]
#     else:
#       l = stride[0]+1
#     self.fc_q = Conv2d(
#       channels, heads * d_k, kernel_size=(l, l), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_k = Conv2d(
#       channels, heads * d_k, kernel_size=(1, 1), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_v = Conv2d(
#       channels, heads * d_v, kernel_size=(1, 1), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)

#     self.fc_o = Conv2d(
#       heads * d_k, channels, kernel_size=(1, 1), stride=(1, 1), groups=1, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)

#     self.weights.extend(self.fc_q.weights)
#     self.weights.extend(self.fc_k.weights)
#     self.weights.extend(self.fc_v.weights)
#     self.weights.extend(self.fc_o.weights)

#     self.channels = channels
#     self.d_k = d_k
#     self.d_v = d_v
#     self.heads = heads
#     self.scaled_factor = self.d_k ** -0.5

#     b_wight = torch.empty(size=(1, self.heads, feat_dim , feat_dim//stride[1]))
#     nn.init.xavier_normal_(b_wight, gain=1.0)
#     self.B = nn.Parameter(data = b_wight, requires_grad=True)
#     self.weights.append(self.B)

#   # self.ln = nn.GroupNorm(channels, channels)
#     self.ln = nn.GroupNorm(1, channels)

#   def forward(self, x, drop=0.0):
#     b, c, f, t = x.shape
#     d_k, d_v, h = self.d_k, self.d_v, self.heads
#     x = self.ln(x)

#     # Get q, k, v
#     # exit()
#     q = self.fc_q(x) # [b, heads*d_k, f, t]
#     q = q.view(b, h, d_k, f, t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, f, t, d_k]

#     k = self.DWConv(x)
#     v = k
#     # k = self.DWConv_k(x, drop)
#     _, _, k_f, k_t = k.shape
#     k = self.fc_k(k)
#     k = k.view(b, h, d_k, k_f, k_t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, k_f, k_t, d_k]

#     # v = self.DWConv_v(x, drop)
#     _, _, v_f, v_t = v.shape
#     v = self.fc_v(v)
#     v = v.view(b, h, d_v, v_f, v_t).permute(0, 1, 3, 4, 2).contiguous() # [b, heads, v_f, v_t, d_v]

#     # Attention
#     attn = torch.einsum('... x y j, ... a b j -> ... x y a b', q, k) * self.scaled_factor # [b, heads, f, t k_f, k_t]
#     attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t = attn.shape
#     attn = attn.view(attn_b, attn_head, attn_f * attn_t, attn_k_f * attn_k_t)

#     Bias = torch.repeat_interleave(torch.repeat_interleave(self.B, attn_t, dim=-2), attn_k_t, dim=-1)
#     attn = attn + Bias.expand_as(attn)
#     attn = torch.softmax(attn, dim = -1) # [b, heads, f *t, k_f * k_t]
#     attn = attn.view(attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t) # [b, heads, f, t k_f, k_t]

#     result = torch.einsum('... a b x y, ... x y c -> ... a b c', attn, v).permute(0, 2, 3, 1, 4).contiguous() # [b, f, t, heads, d_k]

#     result = result.view(b, f, t, self.heads * self.d_v)
#     # result = self.fc_o(result, drop).permute(0, 3, 1, 2).contiguous()
#     result = self.fc_o(result.permute(0, 3, 1, 2).contiguous(), drop)
#     return result

class LMHSA12(nn.Module):
  def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=80):
    super(LMHSA12, self).__init__()
    self.weights = []
    self.DWConv = Conv2d(
      channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv.weights)

    # if stride[0] == 1:
    #   l = stride[0]+2
    # else:
    #   l = stride[0]+1
    # channels
    l = 3
    self.fc_q1 = Conv2d(
      channels, heads * d_k, kernel_size=(l, 1), stride=(1, 1), groups=channels//1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_q2 = Conv2d(
      heads * d_k, heads * d_k, kernel_size=(1, l), stride=(1, 1), groups=channels//1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_k = Conv2d(
      channels, heads * d_k, kernel_size=(1, 1), stride=(1, 1), groups=channels//1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_v = Conv2d(
      channels, heads * d_v, kernel_size=(1, 1), stride=(1, 1), groups=channels//1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)

    self.fc_o = Conv2d(
      heads * d_k, channels, kernel_size=(1, 1), stride=(1, 1), groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)

    self.weights.extend(self.fc_q1.weights)
    self.weights.extend(self.fc_q2.weights)
    self.weights.extend(self.fc_k.weights)
    self.weights.extend(self.fc_v.weights)
    self.weights.extend(self.fc_o.weights)

    self.channels = channels
    self.d_k = d_k
    self.d_v = d_v
    self.heads = heads
    self.scaled_factor = self.d_k ** -0.5

    b_wight = torch.empty(size=(1, self.heads, feat_dim , feat_dim//stride[1]))
    nn.init.xavier_normal_(b_wight, gain=1.0)
    self.B = nn.Parameter(data = b_wight, requires_grad=True)
    self.weights.append(self.B)

  # self.ln = nn.GroupNorm(channels, channels)
    self.ln = nn.GroupNorm(1, channels)

  def forward(self, x, drop=0.0):
    b, c, f, t = x.shape
    d_k, d_v, h = self.d_k, self.d_v, self.heads
    x = self.ln(x)

    # Get q, k, v
    # exit()
    q = self.fc_q1(x)
    q = self.fc_q2(q.mean(-2, keepdim=True))+q # [b, heads*d_k, f, t]
    q = q.view(b, h, d_k, f, t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, f, t, d_k]

    k = self.DWConv(x)
    v = k
    # k = self.DWConv_k(x, drop)
    _, _, k_f, k_t = k.shape
    k = self.fc_k(k)
    k = k.view(b, h, d_k, k_f, k_t).permute(0, 1, 3, 4, 2).contiguous()  # [b, heads, k_f, k_t, d_k]

    # v = self.DWConv_v(x, drop)
    _, _, v_f, v_t = v.shape
    v = self.fc_v(v)
    v = v.view(b, h, d_v, v_f, v_t).permute(0, 1, 3, 4, 2).contiguous() # [b, heads, v_f, v_t, d_v]

    # Attention
    attn = torch.einsum('... x y j, ... a b j -> ... x y a b', q, k) * self.scaled_factor # [b, heads, f, t k_f, k_t]
    attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t = attn.shape
    attn = attn.view(attn_b, attn_head, attn_f * attn_t, attn_k_f * attn_k_t)
    Bias = torch.repeat_interleave(torch.repeat_interleave(self.B, attn_t, dim=-2), attn_k_t, dim=-1)
    attn = attn + Bias.expand_as(attn)
    attn = torch.softmax(attn, dim = -1) # [b, heads, f *t, k_f * k_t]
    # attn = torch.softmax(attn*0.5, dim = -1) # [b, heads, t, k_t]

    attn_out = attn
    attn = attn.view(attn_b, attn_head, attn_f, attn_t, attn_k_f, attn_k_t) # [b, heads, f, t k_f, k_t]

    result = torch.einsum('... a b x y, ... x y c -> ... a b c', attn, v).permute(0, 2, 3, 1, 4).contiguous() # [b, f, t, heads, d_k]

    result = result.view(b, f, t, self.heads * self.d_v)
    # result = self.fc_o(result, drop).permute(0, 3, 1, 2).contiguous()
    result = self.fc_o(result.permute(0, 3, 1, 2).contiguous(), drop)

    return result, attn_out

# class LMHSA_mod(nn.Module):
#   def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=80):
#     super(LMHSA_mod, self).__init__()
#     self.weights = []
#     self.DWConv = Conv2d(
#       channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.DWConv.weights)

#     if stride[0] == 1:
#       l = stride[0]+2
#     else:
#       l = stride[0]+1

#     self.fc_q1 = Conv2d(
#       channels, heads * d_k // feat_dim, kernel_size=(l, 1), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_q2 = Conv2d(
#       heads * d_k // feat_dim, heads * d_k // feat_dim, kernel_size=(1, l), stride=(1, 1), groups=channels, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_k = Conv1d(
#       channels*feat_dim//stride[1], heads * d_k, kernel_size=1, stride=1, groups=channels*feat_dim//stride[1], use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_v = Conv1d(
#       channels*feat_dim//stride[1], heads * d_v, kernel_size=1, stride=1, groups=channels*feat_dim//stride[1], use_bias=False,
#       apply_bn=False, activ=None, as_class=False)

#     self.fc_o = Conv1d(
#       heads * d_k, channels*feat_dim, kernel_size=1, stride=1, groups=heads * d_k, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)

#     self.weights.extend(self.fc_q1.weights)
#     self.weights.extend(self.fc_q2.weights)
#     self.weights.extend(self.fc_k.weights)
#     self.weights.extend(self.fc_v.weights)
#     self.weights.extend(self.fc_o.weights)

#     self.channels = channels
#     self.d_k = d_k
#     self.d_v = d_v
#     self.heads = heads
#     self.tem = 0.1
#     self.scaled_factor = self.d_k ** -0.5

#     # b_wight = torch.empty(size=(1, self.heads, feat_dim, feat_dim//stride[1]))
#     # nn.init.xavier_normal_(b_wight, gain=1.0)
#     # self.B = nn.Parameter(data = b_wight, requires_grad=True)
#     # self.weights.append(self.B)

#   # self.ln = nn.GroupNorm(channels, channels)
#     self.ln = nn.GroupNorm(1, channels)

#   def forward(self, x, drop=0.0):
#     b, c, f, t = x.shape
#     d_k, d_v, h = self.d_k, self.d_v, self.heads
#     x = self.ln(x)

#     # Get q, k, v
#     q = self.fc_q1(x)
#     q = self.fc_q2(q.mean(-2, keepdim=True)) + q # [b, heads*d_k, f, t]    
#     q = q.view(b, h*d_k, t) # [b, heads*d_k, f, t]
#     q = q.view(b, h, d_k, t).permute(0, 1, 3, 2).contiguous()  # [b, heads, t, d_k]

#     k = self.DWConv(x)
#     v = k
#     # k = self.DWConv_k(x, drop)
#     _, _, k_f, k_t = k.shape

#     k = self.fc_k(k.view(b, c*k_f, k_t))
#     k = k.view(b, h, d_k, k_t).permute(0, 1, 3, 2).contiguous()  # [b, heads, k_t, d_k]

#     # v = self.DWConv_v(x, drop)
#     _, _, v_f, v_t = v.shape
#     v = self.fc_v(v.view(b, c*v_f, v_t))
#     v = v.view(b, h, d_v, v_t).permute(0, 1, 3, 2).contiguous() # [b, heads, v_t, d_v]

#     # Attention
#     attn = torch.einsum('... x j, ... a j -> ... x a', q, k) * self.scaled_factor # [b, heads, t, k_t]
#     attn_b, attn_head, attn_t, attn_k_t = attn.shape

#     # attn = torch.softmax(attn / self.tem, dim = -1) # [b, heads, t, k_t]
#     attn = F.sigmoid(attn) # [b, heads, t, k_t]
#     attn = attn.view(attn_b, attn_head, attn_t, attn_k_t) # [b, heads, t, k_t]

#     result = torch.einsum('... a x, ... x c -> ... a c', attn, v).permute(0, 2, 1, 3).contiguous() # [b, t, heads, d_k]
#     result = result.view(b, t, self.heads * self.d_v)
#     result = self.fc_o(result.permute(0, 2, 1).contiguous(), drop)
#     result = result.view(b, c, f, t)
#     return result

# class LMHSA_mod(nn.Module):
#   def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=80):
#     super(LMHSA_mod, self).__init__()
#     self.weights = []
#     self.DWConv = Conv2d(
#       channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#       apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv.weights)
#     # self.DWConv_k = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv_k.weights)
#     # self.DWConv_v = Conv2d(
#     #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.weights.extend(self.DWConv_v.weights)

#     if stride[0] == 1:
#       l = stride[0]+2
#     else:
#       l = stride[0]+1

#     # self.fc_q1 = Conv2d(
#     #   channels, heads * d_k // feat_dim, kernel_size=(l, 1), stride=(1, 1), groups=channels, use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.fc_q2 = Conv2d(
#     #   heads * d_k // feat_dim, heads * d_k // feat_dim, kernel_size=(1, l), stride=(1, 1), groups=channels, use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.fc_k = Conv1d(
#     #   channels*feat_dim//stride[1], heads * d_k, kernel_size=1, stride=1, groups=channels*feat_dim//stride[1], use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.fc_v = Conv1d(
#     #   channels*feat_dim//stride[1], heads * d_v, kernel_size=1, stride=1, groups=channels*feat_dim//stride[1], use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)

#     self.fc_q1 = Conv2d(
#       channels, heads * d_k // feat_dim, kernel_size=(l, 1), stride=(1, 1), groups=1, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_q2 = Conv2d(
#       heads * d_k // feat_dim, heads * d_k // feat_dim, kernel_size=(1, l), stride=(1, 1), groups=1, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_k = Conv1d(
#       channels*feat_dim//stride[1], heads * d_k, kernel_size=1, stride=1, groups=1, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)
#     self.fc_v = Conv1d(
#       channels*feat_dim//stride[1], heads * d_v, kernel_size=1, stride=1, groups=1, use_bias=False,
#       apply_bn=False, activ=None, as_class=False)

#     # self.fc_q1 = Conv2d(
#     #   channels, heads * d_k // feat_dim, kernel_size=(l, 1), stride=(1, 1), groups=heads, use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.fc_q2 = Conv2d(
#     #   heads * d_k // feat_dim, heads * d_k // feat_dim, kernel_size=(1, l), stride=(1, 1), groups=heads, use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.fc_k = Conv1d(
#     #   channels*feat_dim//stride[1], heads * d_k, kernel_size=1, stride=1, groups=heads, use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)
#     # self.fc_v = Conv1d(
#     #   channels*feat_dim//stride[1], heads * d_v, kernel_size=1, stride=1, groups=heads, use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)

#     # self.fc_o = Conv1d(
#     #   heads * d_k, channels*feat_dim, kernel_size=1, stride=1, groups=heads * d_k, use_bias=False,
#     #   apply_bn=False, activ=None, as_class=False)

#     # self.weights.extend(self.fc_q1.weights)
#     # self.weights.extend(self.fc_q2.weights)
#     # self.weights.extend(self.fc_k.weights)
#     # self.weights.extend(self.fc_v.weights)
#     # self.weights.extend(self.fc_o.weights)

#     self.channels = channels
#     self.d_k = d_k
#     self.d_v = d_v
#     self.heads = heads
#     self.tem = 1.0
#     self.scaled_factor = self.d_k ** -0.5

#     # b_wight = torch.empty(size=(1, self.heads, feat_dim, feat_dim//stride[1]))
#     # nn.init.xavier_normal_(b_wight, gain=1.0)
#     # self.B = nn.Parameter(data = b_wight, requires_grad=True)
#     # self.weights.append(self.B)

#     self.ln = nn.GroupNorm(1, channels)

#   def forward(self, x, drop=0.0):
#     b, c, f, t = x.shape
#     d_k, d_v, h = self.d_k, self.d_v, self.heads
#     x = self.ln(x)

#     # Get q, k, v
#     q = self.fc_q1(x)
#     q = self.fc_q2(q.mean(-2, keepdim=True)) + q # [b, heads*d_k, f, t]    
#     q = q.view(b, h*d_k, t) # [b, heads*d_k, f, t]
#     q = q.view(b, h, d_k, t).permute(0, 1, 3, 2)  # [b, heads, t, d_k]

#     # k = self.DWConv_k(x, drop)
#     k = self.DWConv(x)
#     v = k.clone()
#     _, _, k_f, k_t = k.shape
#     k = self.fc_k(k.view(b, c*k_f, k_t))
#     k = k.view(b, h, d_k, k_t).permute(0, 1, 3, 2)  # [b, heads, k_t, d_k]

#     # v = self.DWConv_v(y, drop)
#     _, _, v_f, v_t = v.shape
#     v = self.fc_v(v.view(b, c*v_f, v_t))
#     v = v.view(b, h, d_v, v_t).permute(0, 1, 3, 2) # [b, heads, v_t, d_v]

#     # Attention
#     attn = torch.einsum('... x j, ... a j -> ... x a', q, k) * self.scaled_factor # [b, heads, t, k_t]
#     attn_b, attn_head, attn_t, attn_k_t = attn.shape

#     attn = torch.softmax(attn / self.tem, dim = -1) # [b, heads, t, k_t]
#     attn = attn.view(attn_b, attn_head, attn_t, attn_k_t) # [b, heads, t, k_t]

#     result = torch.einsum('... a x, ... x c -> ... a c', attn, v).permute(0, 2, 1, 3) # [b, t, heads, d_k]
#     result = torch.reshape(result, (b, t, self.heads * self.d_v))
#     # result = self.fc_o(result.permute(0, 2, 1), drop)
#     # result = result.view(b, c, f, t)
#     result = result.permute(0, 2, 1).view(b, c, f, t)

#     return result

class LMHSA_mod(nn.Module):
  def __init__(self, channels, d_k, d_v, heads, stride=(2, 2), feat_dim=80):
    super(LMHSA_mod, self).__init__()
    self.weights = []
    self.DWConv = Conv2d(
      channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
      apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv.weights)

    l = 3 

    self.dif_dim1 = False
    self.dif_dim2 = False
    self.dif_dim3 = False
    
    if heads * d_k / feat_dim <= 1:
      min_dim1 = 1
    else:
      min_dim1 = self.found_num(heads * d_k / feat_dim)
    if float(min_dim1) != (heads * d_k / feat_dim):
      self.dif_dim1 = True
      self.fix_dim1 = Conv1d(
        feat_dim*min_dim1, heads * d_k, kernel_size=1, stride=1, groups=1, use_bias=False,
        apply_bn=False, activ=None, as_class=False)   

    if heads * d_k / (feat_dim//stride[1]) <= 1:
      min_dim2 = 1
    else:
      min_dim2 = self.found_num(heads * d_k / (feat_dim//stride[1]))
    if float(min_dim2) != (heads * d_k / (feat_dim//stride[1])):
      self.dif_dim2 = True
      self.fix_dim2 = Conv1d(
        (feat_dim//stride[1])*min_dim2, heads * d_k, kernel_size=1, stride=1, groups=1, use_bias=False,
        apply_bn=False, activ=None, as_class=False)   

    # if heads * d_v / (feat_dim//stride[1]) <= 1:
    #   min_dim3 = 1
    # else:
    #   min_dim3 = self.found_num(heads * d_v / (feat_dim//stride[1]))
    # if float(min_dim3) != (heads * d_v / (feat_dim//stride[1])):
    #   self.dif_dim3 = True
    #   self.fix_dim3 = Conv1d(
    #     (feat_dim//stride[1])*min_dim3, heads * d_v, kernel_size=1, stride=1, groups=1, use_bias=False,
    #     apply_bn=False, activ=None, as_class=False)
    min_dim3 = heads * d_v * stride[1]

    # print(self.dif_dim1, self.dif_dim2)
    # print(math.ceil(min_dim1//4), math.ceil(min_dim2//4), math.ceil(min_dim3//4))

    self.fc_q1 = Conv2d(
      channels, min_dim1, kernel_size=(3, 1), stride=(1, 1), groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_q2 = Conv1d(
      min_dim1, min_dim1, kernel_size=3, stride=1, groups=math.ceil(min_dim1/min_dim1), use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_k1 = Conv2d(
      channels, min_dim2, kernel_size=(1, 1), stride=(1, 1), groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_k2 = Conv1d(
      min_dim2, min_dim2, kernel_size=1, stride=1, groups=math.ceil(min_dim2/min_dim2), use_bias=False,
      apply_bn=False, activ=None, as_class=False)    
    self.fc_v1 = Conv2d(
      channels, min_dim3, kernel_size=(1, 1), stride=(1, 1), groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)
    self.fc_v2 = Conv1d(
      min_dim3, min_dim3, kernel_size=1, stride=1, groups=math.ceil(min_dim3/min_dim3), use_bias=False,
      apply_bn=False, activ=None, as_class=False)    

    self.fc_o = Conv2d(
      heads * d_v, channels, kernel_size=(1, 1), stride=(1, 1), groups=1, use_bias=False,
      apply_bn=False, activ=None, as_class=False)

    # if heads * d_v < min_dim1 * feat_dim:
    #   print("UserWarning : C_v dimension is too small")
    #   self.fix_dim = Conv1d(
    #     heads * d_v, min_dim1 * feat_dim, kernel_size=1, stride=1, groups=1, use_bias=False,
    #     apply_bn=False, activ=None, as_class=False)

    self.channels = channels
    self.d_k = d_k
    self.d_v = d_v
    self.heads = heads
    self.tem = 1.0
    self.scaled_factor = self.d_k ** -0.5
    self.min_dim1 = min_dim1

    self.ln = nn.GroupNorm(1, channels)

  def found_num(self, x):
    # for i in range(1,12):
    #   if 2^i < x <= 2^(i+1):
    #     return 2^(i+1)
    if x <= 2:
      return 2
    elif x <= 4:
      return 4
    elif x <= 8:
      return 8
    elif x <= 16:
      return 16
    elif x <= 32:
      return 32
    elif x <= 64:
      return 64
    elif x <= 128:
      return 128
    elif x <= 256:
      return 256
    elif x <= 512:
      return 512
    elif x <= 1024:
      return 1024
    elif x <= 2048:
      return 2048

  def forward(self, x, drop=0.0):
    b, c, f, t = x.shape
    d_k, d_v, h = self.d_k, self.d_v, self.heads
    x = self.ln(x)

    # Get q, k, v
    q = self.fc_q1(x)
    # q = q.view(b, h*d_k, t) # [b, heads*d_k, f, t]
    if self.dif_dim1:
      res_q = self.fix_dim1(q.view(b, q.shape[1]*q.shape[2], q.shape[3]))
    else:
      res_q = q.view(b, h*d_k, t) # [b, heads*d_k, f, t]
    qq = self.fc_q2(q.mean(-2, keepdim=False)) 
    q = torch.repeat_interleave(qq, res_q.shape[-2]//qq.shape[-2], dim=-2) + res_q # [b, heads*d_k, f, t]
    q = q.view(b, h, d_k, t).permute(0, 1, 3, 2)  # [b, heads, t, d_k]

    k = self.DWConv(x)
    v = k.clone()

    _, _, k_f, k_t = k.shape
    k = self.fc_k1(k)
    # k = k.view(b, h*d_k, k_t) # [b, heads*d_k, k_f, k_t]
    if self.dif_dim2:
      res_k = self.fix_dim2(k.view(b, k.shape[1]*k.shape[2], k.shape[3])) # [b, heads*d_k, k_f, k_t]
    else:
      res_k = k.view(b, h*d_k, k_t) # [b, heads*d_k, k_f, k_t]
    kk = self.fc_k2(k.mean(-2, keepdim=False))
    k = torch.repeat_interleave(kk, res_k.shape[-2]//kk.shape[-2], dim=-2) + res_k
    k = k.view(b, h, d_k, k_t).permute(0, 1, 3, 2)  # [b, heads, k_t, d_k]

    _, _, v_f, v_t = v.shape
    v = self.fc_v1(v)
    # v = v.view(b, h*d_v, v_t) # [b, heads*d_v, v_f, v_t]
    # if self.dif_dim3:
    #   res_v = self.fix_dim3(v.view(b, v.shape[1]*v.shape[2], v.shape[3])) # [b, heads*d_v, v_f, v_t]   
    # else:
    res_v = v.view(b, h*d_v*f, v_t) # [b, heads*d_v, v_f, v_t]   
    vv = self.fc_v2(v.mean(-2, keepdim=False))
    v = torch.repeat_interleave(vv, res_v.shape[-2]//vv.shape[-2], dim=-2) + res_v
    v = v.view(b, h, f*d_v, v_t).permute(0, 1, 3, 2) # [b, heads, v_t, f*d_v]

    # Attention
    attn = torch.einsum('... x j, ... a j -> ... x a', q, k) * self.scaled_factor # [b, heads, t, k_t]
    attn_b, attn_head, attn_t, attn_k_t = attn.shape

    attn = torch.softmax(attn, dim = -1) # [b, heads, t, k_t]
    # attn = torch.softmax(attn*0.7, dim = -1) # [b, heads, t, k_t]
    attn_out = attn
    attn = attn.view(attn_b, attn_head, attn_t, attn_k_t) # [b, heads, t, k_t]

    result = torch.einsum('... a x, ... x c -> ... a c', attn, v).permute(0, 2, 1, 3) # [b, t, heads, f*d_v]
    result = torch.reshape(result, (b, t, self.heads * f * self.d_v))
    result = result.permute(0, 2, 1)
    # if result.shape[-2] < self.min_dim1* f:
    #   result = self.fix_dim(result)
    result = result.view(b, self.heads * self.d_v, f, t)
    result = self.fc_o(result, drop)

    return result, attn_out


class IRFFN(nn.Module):
  """
  Inverted Residual Feed-forward Network
  """
  def __init__(self, in_dim, out_dim, R_value, stride=(1, 1), 
               padding='same', dilation=(1, 1), use_bias=True):
    super(IRFFN, self).__init__()
    exp_dim = int(in_dim * R_value)
    self.weights = []
    self.conv1 = Conv2d(
      in_dim, exp_dim, (1, 1), stride, 
      padding, dilation, groups=1, use_bias=use_bias,
      apply_bn=False, activ="GELU", as_class=False)
    self.weights.extend(self.conv1.weights)

    self.bn1 = nn.BatchNorm2d(exp_dim, affine=True, track_running_stats=True)

    self.DWConv = Conv2d(
      exp_dim, exp_dim, (3, 3), stride, 
      padding, dilation, groups=exp_dim, use_bias=use_bias, 
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.DWConv.weights)

    self.bn2 = nn.BatchNorm2d(exp_dim, affine=True, track_running_stats=True)
    self.acti = nn.GELU()

    self.conv2 = Conv2d(
      exp_dim, out_dim, (1, 1), stride, 
      padding, dilation, groups=1, use_bias=use_bias, 
      apply_bn=True, activ=None, as_class=False)
    self.weights.extend(self.conv2.weights)

    # self.ln = nn.GroupNorm(in_dim, in_dim)
    self.ln = nn.GroupNorm(1, in_dim)

  def forward(self, x, drop=0.0):
    x = self.ln(x)
    x = self.bn1(self.conv1(x))
    x = self.bn2(self.acti(self.DWConv(x, drop) + x))
    x = self.conv2(x)
    return x

class CMTBlock(nn.Module):
  def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
    super(CMTBlock, self).__init__()
    self.weights = []
    # Local Perception Unit
    self.lpu = LPU(in_channels, in_channels, kernel_size, use_bias=use_bias)
    self.weights.extend(self.lpu.weights)
    # Lightweight MHSA
    self.lmhsa = LMHSA(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
    # self.lmhsa = LMHSA2(in_channels, d_k, d_v, num_heads, stride=stride)     
    self.weights.extend(self.lmhsa.weights)

    # Inverted Residual FFN
    self.irffn = IRFFN(in_channels, in_channels, R, use_bias=use_bias)
    self.weights.extend(self.irffn.weights)

  def forward(self, x, drop=0.0):
    x = self.lpu(x, drop)

    # x = self.lmhsa(x, drop) + x
    x = self.lmhsa(x) + x
    x = self.irffn(x, drop) + x
    return x

class CMTBlock2(nn.Module):
  def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
    super(CMTBlock2, self).__init__()
    self.weights = []
    # Local Perception Unit
    self.lpu = LPU(in_channels, in_channels, kernel_size, use_bias=use_bias)
    self.weights.extend(self.lpu.weights)
    # Lightweight MHSA
    # self.lmhsa = LMHSA(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
    self.lmhsa = LMHSA2_SDW(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)     
    self.weights.extend(self.lmhsa.weights)

    # Inverted Residual FFN
    self.irffn = IRFFN(in_channels, in_channels, R, use_bias=use_bias)
    self.weights.extend(self.irffn.weights)

  def forward(self, x, drop=0.0):
    x = self.lpu(x, drop)

    # x = self.lmhsa(x, drop) + x
    x = self.lmhsa(x) + x
    x = self.irffn(x, drop) + x
    return x

class CMTBlock3(nn.Module):
  def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
    super(CMTBlock3, self).__init__()
    self.weights = []
    # Local Perception Unit
    self.lpu = LPU(in_channels, in_channels, kernel_size, use_bias=use_bias)
    self.weights.extend(self.lpu.weights)
    # Lightweight MHSA
    # self.lmhsa = LMHSA(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
    self.lmhsa = LMHSA3(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)     
    self.weights.extend(self.lmhsa.weights)

    # Inverted Residual FFN
    self.irffn = IRFFN(in_channels, in_channels, R, use_bias=use_bias)
    self.weights.extend(self.irffn.weights)

  def forward(self, x, drop=0.0):
    x = self.lpu(x, drop)

    # x = self.lmhsa(x, drop) + x
    x = self.lmhsa(x) + x
    x = self.irffn(x, drop) + x
    return x

class CMTBlock_lambda(nn.Module):
  def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
    super(CMTBlock_lambda, self).__init__()
    self.weights = []
    # Local Perception Unit
    self.lpu = LPU(in_channels, in_channels, kernel_size, use_bias=use_bias)
    self.weights.extend(self.lpu.weights)
    # Lightweight MHSA
    self.lmhsa = LW_LambdaLayer(dim = in_channels, dim_out = in_channels,
        r = 3, dim_k = d_k, heads = num_heads, dim_u = num_heads, feat_dim=feat_dim)   
    # self.lmhsa = LambdaLayer(dim = in_channels, dim_out = in_channels,
    #     r = 7, dim_k = d_k, heads = num_heads, dim_u = num_heads)
    self.weights.extend(self.lmhsa.weights)

    # Inverted Residual FFN
    self.irffn = IRFFN(in_channels, in_channels, R, use_bias=use_bias)
    self.weights.extend(self.irffn.weights)

  def forward(self, x, drop=0.0):
    x = self.lpu(x, drop)

    # x = self.lmhsa(x, drop) + x
    x = self.lmhsa(x) + x
    x = self.irffn(x, drop) + x
    return x

class Patch_Aggregate(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size=(2, 2), stride=(2, 2), 
               padding='same', dilation=(1, 1), use_bias=True):
    super(Patch_Aggregate, self).__init__()
    self.weights = []
    self.conv = Conv2d(
      in_dim, out_dim, kernel_size, stride, 
      padding, dilation, use_bias=use_bias, 
      apply_bn=False, activ=None, as_class=False)
    self.weights.extend(self.conv.weights)

    # self.ln = nn.GroupNorm(out_dim, out_dim)
    self.ln = nn.GroupNorm(1, out_dim)

  def forward(self, x, drop=0.0):
    x = self.conv(x)
    x = self.ln(x)
    return x

# class Patch_Aggregate_mod(nn.Module):
#   def __init__(self, in_dim, out_dim, kernel_size=(2, 2), stride=(2, 2), 
#                padding='same', dilation=(1, 1), use_bias=True):
#     super(Patch_Aggregate_mod, self).__init__()
#     self.weights = []
#     self.conv_x = Conv2d(
#       in_dim, out_dim, kernel_size, stride, 
#       padding, dilation, use_bias=use_bias, 
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.conv_x.weights)

#     self.ln_x = nn.GroupNorm(1, out_dim)

#     self.conv_y = Conv2d(
#       in_dim, out_dim, kernel_size, stride, 
#       padding, dilation, use_bias=use_bias, 
#       apply_bn=False, activ=None, as_class=False)
#     self.weights.extend(self.conv_y.weights)

#     self.ln_y = nn.GroupNorm(1, out_dim)

#   def forward(self, x, y, drop=0.0):
#     x = self.conv_x(x)
#     x = self.ln_x(x)
#     y = self.conv_y(y)
#     y = self.ln_y(y)
#     return x, y

class BC_CMTBlock(nn.Module):
  def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
    super(BC_CMTBlock, self).__init__()
    self.weights = []
    # Local Perception Unit
    self.lpu = LPU(in_channels, in_channels, kernel_size, use_bias=use_bias)
    self.weights.extend(self.lpu.weights)
    # Lightweight MHSA
    self.lmhsa = LMHSA(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
    # self.lmhsa = LMHSA2(in_channels, d_k, d_v, num_heads, stride=stride)     
    self.weights.extend(self.lmhsa.weights)

    # Inverted Residual FFN
    self.irffn = BC_Conv(in_channels, in_channels, use_bias=use_bias, feat_dim=feat_dim)
    self.weights.extend(self.irffn.weights)

  def forward(self, x, drop=0.0):
    x = self.lpu(x, drop)
    x = self.lmhsa(x) + x
    x = self.irffn(x, drop) + x
    return x

class BC_CMTBlock_expand12(nn.Module):
  def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
    super(BC_CMTBlock_expand12, self).__init__()
    self.weights = []
    # Local Perception Unit
    self.lpu = LPU(in_channels, in_channels, kernel_size, use_bias=use_bias)
    self.weights.extend(self.lpu.weights)

    # Lightweight MHSA
    self.lmhsa = LMHSA12(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
    self.weights.extend(self.lmhsa.weights)

    # Inverted Residual FFN
    self.irffn = BC_Conv_expand(in_channels, in_channels, R, use_bias=use_bias, feat_dim=feat_dim)
    self.weights.extend(self.irffn.weights)

  def forward(self, x, drop=0.0):
    x = self.lpu(x, drop)
    x = self.lmhsa(x) + x
    x = self.irffn(x, drop) + x
    return x


class BC_CMTBlock_mod(nn.Module):
  def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
    super(BC_CMTBlock_mod, self).__init__()
    self.weights = []
    # Local Perception Unit
    self.lpu = LPU_Mod(in_channels, in_channels, kernel_size[0], use_bias=use_bias, feat_dim=feat_dim)
    self.weights.extend(self.lpu.weights)

    # Lightweight MHSA
    # self.lmhsa = LMHSA12(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
    self.lmhsa = LMHSA_mod(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
    self.weights.extend(self.lmhsa.weights)

    # Inverted Residual FFN
    self.irffn = BC_Conv_expand(in_channels, in_channels, R, use_bias=use_bias, feat_dim=feat_dim)
    self.weights.extend(self.irffn.weights)

  def forward(self, x, drop=0.0):
    x = self.lpu(x, drop)
    # out, attn = self.lmhsa(x, t_attn)
    out, attn = self.lmhsa(x)
    x = out + x
    x = self.irffn(x, drop) + x
    return x, attn

# class BC_CMTBlock_mod(nn.Module):
#   def __init__(self, in_channels, kernel_size, d_k, d_v, num_heads, stride=(2, 2), R = 3.6, use_bias=True, feat_dim=64):
#     super(BC_CMTBlock_mod, self).__init__()
#     self.weights = []
#     # Local Perception Unit
#     self.lpu = LPU_Mod(in_channels, in_channels, kernel_size[0], use_bias=use_bias, feat_dim=feat_dim)
#     self.weights.extend(self.lpu.weights)

#     # Lightweight MHSA
#     self.lmhsa = LMHSA_mod(in_channels, d_k, d_v, num_heads, stride=stride, feat_dim=feat_dim)
#     self.ln = nn.GroupNorm(1, in_channels)
#     self.weights.extend(self.lmhsa.weights)

#     # Inverted Residual FFN
#     self.irffn = BC_Conv_expand(in_channels, in_channels, R, use_bias=use_bias, feat_dim=feat_dim)
#     self.weights.extend(self.irffn.weights)

#   def forward(self, x, y, drop=0.0):
#     x = self.lpu(x, drop)
#     y = self.lmhsa(x, y) + self.ln(y)
#     x = x + y
#     x = self.irffn(x, drop) + x
#     return x, y

class MHSA(nn.Module):
  """
  Lightweight Multi-head-self-attention module.
  Inputs:
      Q: [N, C, W]
      K: [N, C, W / stride]
      V: [N, C, W / stride]
  Outputs:
      X: [N, C, W]
  """
  def __init__(self, channels, d_k, d_v, heads, stride=2, q_time_step=64, k_time_step=None):
    super(MHSA, self).__init__()
    self.weights = []
    # self.DWConv = Conv1d(
    #   channels, channels, kernel_size=stride, stride=stride, groups=channels, use_bias=True,
    #   apply_bn=False, activ=None, as_class=False)
    # self.weights.extend(self.DWConv.weights)

    self.fc_q = Dense(channels, heads * d_k, use_bias=False, apply_bn=False, activ=None)
    self.fc_k = Dense(channels, heads * d_k, use_bias=False, apply_bn=False, activ=None)
    self.fc_v = Dense(channels, heads * d_v, use_bias=False, apply_bn=False, activ=None)
    self.fc_o = Dense(heads * d_k, channels, use_bias=False, apply_bn=False, activ=None)

    self.weights.extend(self.fc_q.weights)
    self.weights.extend(self.fc_k.weights)
    self.weights.extend(self.fc_v.weights)
    self.weights.extend(self.fc_o.weights)

    self.channels = channels
    self.d_k = d_k
    self.d_v = d_v
    self.heads = heads
    self.scaled_factor = self.d_k ** -0.5

    if k_time_step == None:
      k_time_step = q_time_step
    # b_wight = torch.empty(size=(1, self.heads, q_time_step , k_time_step))
    # nn.init.xavier_normal_(b_wight, gain=1.0)
    # self.B = nn.Parameter(data = b_wight, requires_grad=True)
    # self.weights.append(self.B)

  def forward(self, q, k, v, attn_mask=None, drop=0.0):
    # Get q, k, v
    b, c, t = q.shape
    # Reshape
    q = q.permute(0, 2, 1).contiguous() # [b, t, c]
    # exit()
    q = self.fc_q(q)
    q = q.view(b, t, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, t, d_k]

    # k = self.DWConv_k(x, drop)
    # k = self.DWConv(k)
    k_b, k_c, k_t = k.shape
    k = k.permute(0, 2, 1).contiguous()
    k = self.fc_k(k)
    k = k.view(k_b, k_t, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, k_t, d_k]

    # v = self.DWConv_v(x, drop)
    # v = self.DWConv(v)
    v_b, v_c, v_t = v.shape
    v = v.permute(0, 2, 1).contiguous()
    v = self.fc_v(v)
    v = v.view(v_b, v_t, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() # [b, heads, v_t, d_v]

    # Attention
    attn = torch.einsum('... x j, ... a j -> ... x a', q, k) * self.scaled_factor # [b, heads, t, k_t]
    attn_b, attn_head, attn_t, attn_k_t = attn.shape

    if attn_mask is not None:
      attn_mask = torch.triu(torch.ones(attn_b, attn_k_t, attn_t, dtype=torch.int)).to(attn.device).unsqueeze(dim=1).expand_as(attn)
      attn_mask = torch.gt(attn_mask.transpose(-2, -1), 0.5)
      attn.masked_fill_(~attn_mask, -1e9)
      # exit()
    # Bias = torch.repeat_interleave(torch.repeat_interleave(self.B, attn_t, dim=-2), attn_k_t, dim=-1)
    # attn = attn + Bias.expand_as(attn)
    attn = torch.softmax(attn, dim = -1) # [b, heads, t, k_t]
    result = torch.einsum('... a x, ... x c -> ... a c', attn, v).permute(0, 2, 1, 3).contiguous() # [b, t, heads, d_v]

    result = result.view(b, t, self.heads * self.d_v)
    result = self.fc_o(result, drop).permute(0, 2, 1).contiguous()

    return result

class Conv2dNext(nn.Module):
  def __init__(self, in_dim, out_dim, hid_dim, kernel_size, stride=(1, 1), 
               padding='same', dilation=(1, 1), groups=1, use_bias=True, 
               activ="GELU", kwargs={}, w_init_gain='conv2d', 
               shortcut_bn=False, as_class=False):
    super(Conv2dNext, self).__init__()
    self.activ, self.weights = [], []

    if isinstance(stride[0], (tuple, list)): # listed list (to resize the input)
      stride1, stride2 = stride
      stride1, stride2 = tuple(stride1), tuple(stride2)
      if stride1 == (1, 1):
        assert stride2 != (1, 1)
        stride_rsz = stride2
      else:
        stride_rsz = stride1
      resize = True
    else:
      assert tuple(stride) == (1, 1)  # should not resize
      stride1 = stride2 = stride
      resize = False

    ## Conv2d + BatchNorm + ReLU
    self.conv1 = Conv2d(
      in_dim, out_dim, kernel_size, stride1, 
      padding, dilation, groups, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs=kwargs, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv1.weights)

    self.ln = nn.GroupNorm(1, out_dim)

    ## Conv2d + BatchNorm
    self.conv2 = Conv2d(
      out_dim, hid_dim, (1, 1), stride2, 
      padding, dilation, 1, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=activ, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv2.weights)

    ## Conv2d + BatchNorm
    self.conv3 = Conv2d(
      hid_dim, out_dim, (1, 1), stride2, 
      padding, dilation, 1, use_bias, 
      apply_bn=False, bn_mom=bn_mom, bn_eps=bn_eps, 
      activ=None, kwargs={}, w_init_gain=w_init_gain, 
      as_class=False)
    self.weights.extend(self.conv3.weights)

    ## Shortcut
    self.input_trans = Identity()
    if not resize:
      if out_dim == in_dim:
        print('Input size is same as the output size...')
      else:
        print('Modifying #channels for shortcut (1x1conv)...')
        ## Increase/Decrease #channels with 1x1 convolution
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)
    else:
      if out_dim == in_dim:
        print('Resizing input for shortcut (maxpool)...')
        raise NotImplementedError
      else:
        print('Resizing input for shortcut (1x1conv)...')
        self.input_trans = Conv2d(
          in_dim, out_dim, kernel_size=(1, 1), stride=stride_rsz, 
          padding=(0, 0), dilation=(1, 1), use_bias=False, 
          apply_bn=shortcut_bn, bn_mom=bn_mom, bn_eps=bn_eps, 
          activ=None, kwargs={}, w_init_gain=w_init_gain, 
          as_class=False)
        self.weights.extend(self.input_trans.weights)

    ## ReLU
    self.activ_func = getattr(A, activ)(**kwargs)

  def forward(self, x, drop=0.0):
    residual = self.input_trans(x)
    ## Conv + LayerNorm
    x = self.ln(self.conv1(x, drop=drop))
    ## Conv + GELU
    x = self.conv2(x, drop=drop)
    ## Conv + GELU
    x = self.conv3(x, drop=drop)
    ## ReLU
    return x + residual

if __name__=="__main__":
  from torch_utils import to_gpu
  torch.manual_seed(123)

  activ_func, activ_kwargs = "LeakyReLU", dict(negative_slope=0.2)
  activ, kwargs = activ_func, activ_kwargs

  ## Dense
  x = torch.randn(32, 64)
  layer = Dense(
    64, 512, use_bias=True, 
    apply_bn=True, activ=activ, kwargs=kwargs, 
    w_init_gain='linear')
  y = layer(x); print(y.size(), len(layer.weights))

  ## Tdense
  x = torch.randn(32, 64, 300)
  layer = Tdense(
    64, 512, [-2,-1,0,1,2], use_bias=False, 
    apply_bn=True, activ=activ, kwargs=kwargs, 
    w_init_gain='conv1d')
  layer = layer.cuda()
  for n, xx in enumerate(layer.children()): print(n, xx)
  for n, xx in enumerate(layer.conv.children()): print(n, xx)
  print(list(layer.conv._parameters.keys()))
  print(layer.conv._buffers)
  x = to_gpu(x)
  y = layer(x); print(y.size(), len(layer.weights))
  print(y.sum().cpu().detach().numpy())
  # y = layer(x); print(y.size(), len(layer.weights))
  exit()

  ## GatedTdense
  x = torch.randn(32, 64, 300)
  layer = GatedTdense(
    64, 256, [-2,-1,0,1,2], use_bias=False, 
    apply_bn=False, activ=activ, kwargs=kwargs, 
    w_init_gain='conv1d')
  layer = layer.cuda()
  for n, xx in enumerate(layer.children()): print(n, xx)
  for n, xx in enumerate(layer.conv.children()): print(n, xx)
  print(list(layer.conv._parameters.keys()))
  print(layer.conv._buffers)
  x, mem_init = to_gpu(x), to_gpu(torch.zeros_like(x))
  y, mem = layer(x, mem_init); print(y.size(), len(layer.weights))
  print(y.sum().cpu().detach().numpy())
  print(mem.sum().cpu().detach().numpy())
  # y, mem = layer(x, torch.zeros_like(x))
  # print(y.size(), mem.size(), len(layer.weights))

  ## Conv1d
  x = torch.randn(32, 64, 300)
  layer = Conv1d(
    64, 512, kernel_size=5, stride=1, 
    padding='same', dilation=1, use_bias=False, 
    apply_bn=True, activ=activ, kwargs=kwargs, 
    w_init_gain='conv1d')
  y = layer(x); print(y.size(), len(layer.weights))

  ## Conv2d
  x = torch.randn(32, 1, 64, 300)
  layer = Conv2d(
    1, 24, kernel_size=(3, 3), stride=(1, 1), 
    padding='same', dilation=(1, 1), use_bias=False, 
    apply_bn=True, activ=activ, kwargs=kwargs, 
    w_init_gain='conv2d')
  y = layer(x); print(y.size(), len(layer.weights))

  ## Conv2dCbr/Conv2dBrc
  ConvBlock = Conv2dCbr # Conv2dCbr, Conv2dBrc
  x = torch.randn(32, 96, 16, 300)
  layer = ConvBlock(
    96, 192, kernel_size=(3, 3), stride=[(2, 2), (1, 1)], 
    padding='same', dilation=(1, 1), use_bias=False, 
    apply_bn=True, activ=activ, kwargs=kwargs, 
    w_init_gain='conv2d', shortcut_bn=True)
  y = layer(x); print(y.size(), len(layer.weights))
