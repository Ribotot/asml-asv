import torch
from torch import nn

from torch.nn import functional as F
try:
    import models
except:
    import os, sys
    sys.path.append(os.getcwd())
    
from models.asml.layers import Identity, Dense, Tdense, Conv1d, Conv2d, Unet2D_up, Unet2D_down, GraphAttentionLayer, GCN

def l2_normalize(tensor, dim, eps=1e-12):
  l2_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
  l2_norm = torch.max(l2_norm, eps*torch.ones_like(l2_norm))
  return tensor / l2_norm

class StatsPooling(nn.Module):
  def __init__(self, var_eps=1e-12, normalize_stats=False):
    super(StatsPooling, self).__init__()
    self.var_eps = torch.tensor(var_eps)
    self.normalize_stats = normalize_stats

  def forward(self, x, seqmask=None, time_axis=-1):
    """ x is a 3D tensor in shape (B, C, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    if seqmask is None:
      # pool_mean = torch.mean(x, dim=time_axis)
      # pool_var = torch.mean(x**2, dim=time_axis) \
      #          - torch.pow(pool_mean.clone().detach(), 2)
      pool_var, pool_mean = torch.var_mean(
        x, dim=time_axis, keepdim=False, unbiased=False)
    else:
      xm = x * seqmask.unsqueeze(dim=1)
      seq_len = torch.sum(seqmask, dim=time_axis, keepdim=True)
      pool_mean = torch.sum(xm, dim=time_axis) / seq_len
      pool_var = torch.sum(xm**2, dim=time_axis) / seq_len \
               - torch.pow(pool_mean, 2)
               # - torch.pow(pool_mean.clone().detach(), 2)
    ## Variance flooring
    # pool_var = torch.max(pool_var, self.var_eps.expand_as(pool_var))
    pool_var = torch.max(pool_var, self.var_eps*torch.ones_like(pool_var))
    pool_std = torch.sqrt(pool_var)
    ## l2-normalize the mean/std vectors
    if self.normalize_stats:
      pool_mean = l2_normalize(pool_mean, dim=-1, eps=1e-12)
      pool_std = l2_normalize(pool_std, dim=-1, eps=1e-12)
    pool_stats = torch.cat((pool_mean, pool_std), dim=-1)
    return pool_stats


class MHAStatsPooling(nn.Module):
  def __init__(self, num_heads=1, attn_opts={}, 
               var_eps=1e-12, normalize_stats=False):
    super(MHAStatsPooling, self).__init__()
    self.var_eps = torch.tensor(var_eps)
    self.normalize_stats = normalize_stats
    self.weights = []

    self.num_heads = num_heads
    self.key_dim = attn_opts['key_dim']
    self.split_keys = attn_opts['split_keys']
    if self.split_keys:
      self.split_size = self.key_dim // num_heads
    # self.scale_by_dim = attn_opts['scale_by_dim']

    if not self.split_keys:
      if attn_opts['key_units']:
        self.key_networks = []
        key_dim = self.key_dim
        units = attn_opts['key_units']
        splice = attn_opts['key_splice']
        use_bias = attn_opts['key_use_bias']
        apply_bn = attn_opts['key_apply_bn']
        activ_str = attn_opts['key_activ']
        kernel_init_scale = attn_opts['key_kernel_init_scale']
        _key_network = Tdense(
          key_dim, units, splice, use_bias=use_bias, 
          apply_bn=apply_bn, **attn_opts['key_bn_opts'], 
          activ=activ_str, kwargs={}, 
          w_init_gain=kernel_init_scale, as_class=True)
        self.weights.extend(_key_network.weights)
        self.key_networks.append(_key_network)
        key_dim = units
        self.key_networks = nn.Sequential(*self.key_networks)
      else:
        self.key_networks = Identity()
        units = self.key_dim

      self.query_network = Tdense(
        units, num_heads, [0], 
        use_bias=False, apply_bn=False, activ=None, 
        w_init_gain=attn_opts['query_kernel_init_scale'], 
        as_class=True)
      self.weights.extend(self.query_network.weights)
    else:
      if attn_opts['key_units']:
        self.key_networks = []
        key_dim = self.key_dim
        for units, splice, use_bias, apply_bn, activ_str, kernel_init_scale in \
          zip(attn_opts['key_units'], attn_opts['key_splice'], 
              attn_opts['key_use_bias'], attn_opts['key_apply_bn'], 
              attn_opts['key_activ'], attn_opts['key_kernel_init_scale']):
          assert key_dim % num_heads == 0 and units % num_heads == 0
          split_key_networks = nn.ModuleList()
          for _ in range(num_heads):
            split_dims = key_dim // num_heads
            split_units = units // num_heads
            _split_key_network = Tdense(
              split_dims, split_units, splice, use_bias=use_bias, 
              apply_bn=apply_bn, **attn_opts['key_bn_opts'], 
              activ=activ_str, kwargs={}, 
              w_init_gain=kernel_init_scale, as_class=True)
            self.weights.extend(_split_key_network.weights)
            split_key_networks.append(_split_key_network)
          self.key_networks.append(split_key_networks) # listed list
          key_dim = units
        self.key_networks = nn.ModuleList(
          [nn.Sequential(*_net) for _net in zip(*self.key_networks)])
      else:
        self.key_networks = nn.ModuleList(
          [Identity() for _ in range(num_heads)])
        units = self.key_dim

      self.query_network = nn.ModuleList()
      for _ in range(num_heads):
        split_dims = units // num_heads
        _query_network = Tdense(
          split_dims, 1, [0], use_bias=False, 
          apply_bn=False, activ=None, 
          w_init_gain=attn_opts['query_kernel_init_scale'], 
          as_class=True)
        self.weights.extend(_query_network.weights)
        self.query_network.append(_query_network)

  def forward(self, x, key, seqmask=None, time_axis=-1):
    """ x is a 3D tensor in shape (B, C1, T) 
        key is a 3D tensor in shape (B, C2, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    if seqmask is None:
      if not self.split_keys:
        key_out = self.key_networks(key)
        query_times_key = self.query_network(key_out)
        alignment = torch.softmax(query_times_key, dim=time_axis)

        ## loss
        alignment_corr = torch.bmm(alignment.transpose(1,2), alignment)
        eyes = torch.eye(alignment.shape[-1]).unsqueeze(0).repeat(alignment.shape[0],1,1)
        from torch_utils import to_gpu
        eyes = to_gpu(eyes, alignment_corr.device)
        loss = torch.norm(alignment_corr - eyes)/alignment.shape[-1]

        x_split_size = x.size(1)
        pool_mean = torch.einsum('bct,bht->bhc', x, alignment).contiguous()
        pool_var = torch.einsum('bct,bht->bhc', x**2, alignment).contiguous() \
                 - torch.pow(pool_mean, 2)
                 # - torch.pow(pool_mean.clone().detach(), 2)
      else:
        key_out = [_net(_key) for _key, _net in zip(
          torch.split(key, self.split_size, dim=1), self.key_networks)]
        query_times_key = [_net(_key) for _key, _net in zip(
          key_out, self.query_network)]
        alignment = [torch.softmax(_score, dim=time_axis) for _score in query_times_key]

        x_split_size = x.size(1) // self.num_heads
        pool_mean = [torch.sum(torch.mul(_x, _align), dim=time_axis) \
          for _x, _align in zip(torch.split(x, x_split_size, dim=1), alignment)]
        pool_var = [torch.sum(torch.mul(_x**2, _align), dim=time_axis) - torch.pow(_mean, 2) \
          for _x, _align, _mean in zip(torch.split(x, x_split_size, dim=1), alignment, pool_mean)]
        pool_mean = torch.stack(pool_mean, dim=1)
        pool_var = torch.stack(pool_var, dim=1)
    else:
      raise NotImplementedError

    ## Variance flooring
    # pool_var = torch.max(pool_var, self.var_eps.expand_as(pool_var))
    pool_var = torch.max(pool_var, self.var_eps*torch.ones_like(pool_var))
    pool_std = torch.sqrt(pool_var)
    ## l2-normalize the mean/std vectors
    if self.normalize_stats:
      pool_mean = l2_normalize(pool_mean, dim=-1, eps=1e-12)
      pool_std = l2_normalize(pool_std, dim=-1, eps=1e-12)
    ## Reshape
    pool_mean = pool_mean.view(-1, self.num_heads*x_split_size)
    pool_std = pool_std.view(-1, self.num_heads*x_split_size)
    pool_stats = torch.cat((pool_mean, pool_std), dim=-1)
    return pool_stats, loss

class MHAMeanPooling(MHAStatsPooling):
  def __init__(self, num_heads=1, attn_opts={}, 
               var_eps=1e-12, normalize_stats=False):
    super(MHAMeanPooling, self).__init__(
      num_heads, attn_opts, 0, normalize_stats)

  def forward(self, x, key, seqmask=None, time_axis=-1):
    """ x is a 3D tensor in shape (B, C1, T) 
        key is a 3D tensor in shape (B, C2, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    if seqmask is None:
      if not self.split_keys:
        key_out = self.key_networks(key)
        query_times_key = self.query_network(key_out)
        alignment = torch.softmax(query_times_key, dim=time_axis)

        x_split_size = x.size(1)
        pool_mean = torch.einsum('bct,bht->bhc', x, alignment).contiguous()
      else:
        key_out = [_net(_key) for _key, _net in zip(
          torch.split(key, self.split_size, dim=1), self.key_networks)]
        query_times_key = [_net(_key) for _key, _net in zip(
          key_out, self.query_network)]
        alignment = [torch.softmax(_score, dim=time_axis) for _score in query_times_key]

        x_split_size = x.size(1) // self.num_heads
        pool_mean = [torch.sum(torch.mul(_x, _align), dim=time_axis) \
          for _x, _align in zip(torch.split(x, x_split_size, dim=1), alignment)]
        pool_mean = torch.stack(pool_mean, dim=1)
    else:
      raise NotImplementedError

    ## l2-normalize the mean vector
    if self.normalize_stats:
      pool_mean = l2_normalize(pool_mean, dim=-1, eps=1e-12)
    ## Reshape
    pool_mean = pool_mean.view(-1, self.num_heads*x_split_size)
    return pool_mean


class LDEPooling(nn.Module):
  def __init__(self, in_dim, n_codes=64, normalize_codes=True, norm_eps=1e-12):
    super(LDEPooling, self).__init__()
    self.normalize_codes, self.norm_eps = normalize_codes, norm_eps
    
    init_val = 1.0 / ((n_codes*in_dim)**0.5)
    self.codes = nn.Parameter(torch.Tensor(in_dim, n_codes))
    self.scales = nn.Parameter(torch.Tensor(n_codes))
    nn.init.uniform_(self.codes, -init_val, init_val)
    nn.init.uniform_(self.scales, 0, 1)

  def forward(self, x, seqmask=None):
    """ x is a 3D tensor in shape (B, C, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    if seqmask is None:
      residual = x.unsqueeze(dim=2) - self.codes[None,...,None] # (B, C, K, T)
      residual_l2sqr = torch.pow(residual, 2).sum(dim=1) # (B, K, T)
      weights_tc = torch.softmax(-self.scales[None,:,None] * residual_l2sqr, dim=1) # (B, K, T)
      embeddings = torch.einsum('bckt,bkt->bkc', residual, weights_tc) \
                 / x.size(-1) # (B, K, C)
      ## l2-normalize the codes
      if self.normalize_codes:
        embeddings = l2_normalize(embeddings, dim=-1, eps=self.norm_eps)
      return embeddings
    else:
      raise NotImplementedError


class MHLDEPooling(nn.Module):
  def __init__(self, in_dim, n_codes=64, n_heads=1, normalize_codes=True, norm_eps=1e-12):
    super(MHLDEPooling, self).__init__()
    self.normalize_codes, self.norm_eps = normalize_codes, norm_eps

    assert in_dim % n_heads == 0
    self.dim_per_head = in_dim // n_heads
    self.n_heads = n_heads
    self.n_codes = n_codes
    
    init_val = 1.0 / ((n_codes*self.dim_per_head)**0.5)
    self.codes = nn.Parameter(torch.Tensor(in_dim, n_codes))
    self.scales = nn.Parameter(torch.Tensor(n_heads, n_codes))
    nn.init.uniform_(self.codes, -init_val, init_val)
    nn.init.uniform_(self.scales, 0, 1)

  def forward(self, x, seqmask=None):
    """ x is a 3D tensor in shape (B, C, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    if seqmask is None:
      residual = x.unsqueeze(dim=2) - self.codes[None,...,None] # (B,C,K,T)
      residual = residual.view(x.size(0), self.n_heads, self.dim_per_head, self.n_codes, x.size(-1)) # (B,H,C/H,K,T)
      residual_l2sqr = torch.pow(residual, 2).sum(dim=2) # (B,H,K,T)
      weights_tc = torch.softmax(-self.scales[None,...,None] * residual_l2sqr, dim=1) # (B,H,K,T)
      embeddings = torch.einsum('bhckt,bhkt->bhkc', residual, weights_tc) \
                 / x.size(-1) # (B,H,K,C/H)
      ## l2-normalize the codes
      if self.normalize_codes:
        embeddings = l2_normalize(embeddings, dim=-1, eps=self.norm_eps)
      return embeddings
    else:
      raise NotImplementedError

class ScaledDotProductAttention(nn.Module):
  # Scaled Dot-Product Attention
  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)

  def forward(self, q, k, v, mask=None):
    attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)

    attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)
    return output, attn

class MultiHeadAttention(nn.Module):
 # Multi-Head Attention module '''
  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v
    # self.time_dim = time_dim
    self.weights = []

    self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
    self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
    
    self.weights.append(self.w_qs.weight)
    self.weights.append(self.w_ks.weight)
    self.weights.append(self.w_vs.weight)
    self.weights.append(self.fc.weight)

    self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    self.dropout = nn.Dropout(dropout)
    # self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
    # self.batch_norm = nn.BatchNorm1d(d_model, eps=1e-12, momentum=0.99, 
    #     affine=True, track_running_stats=True)
      
  def forward(self, q, k, v, mask=None):

    # d_k, d_v, n_head, time_dim = self.d_k, self.d_v, self.n_head, self.time_dim
    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
    sz_b, len_q, len_k, len_v = q.size(0), q.size(2), k.size(2), v.size(2)
    
    residual = q
    

    q = q.transpose(-1, -2)
    # q = self.layer_norm(q)
    k = k.transpose(-1, -2)
    # k = self.layer_norm(k)
    v = v.transpose(-1, -2)
    # v = self.layer_norm(v)

    # Pass through the pre-attention projection: b x lq x (n*dv)
    # Separate different heads: b x lq x n x dv
    q = self.w_qs(q).view(sz_b, n_head, d_k, len_q)
    k = self.w_ks(k).view(sz_b, n_head, d_k, len_k)
    v = self.w_vs(v).view(sz_b, n_head, d_v, len_v)
    # Transpose for attention dot product: b x n x lq x dv
    q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
    if mask is not None:
        mask = mask.unsqueeze(1)   # For head axis broadcasting.

    q, attn = self.attention(q, k, v, mask=mask)
    # Transpose to move the head dimension back: b x lq x n x dv
    # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
    q = q.transpose(2, 3).contiguous().view(sz_b, len_q, -1)
    q = self.dropout(self.fc(q))
    q = q.transpose(-1, -2)
    q = torch.cat((q, residual), dim=1)
    # q += residual
    # q = self.batch_norm(q)
    return q, attn

class SelfAttStatsPooling(nn.Module):
  def __init__(self, num_heads=1, attn_opts={}, 
               var_eps=1e-12, normalize_stats=False):
    super(SelfAttStatsPooling, self).__init__()
    self.var_eps = torch.tensor(var_eps)
    self.normalize_stats = normalize_stats
    self.weights = []

    self.num_heads = num_heads
    self.model_dim = attn_opts['model_dim']
    self.key_dim = attn_opts['key_dim']
    self.value_dim = attn_opts['value_dim']    
    # self.time_dim = attn_opts['time_dim']
    self.selfattention = MultiHeadAttention(n_head=self.num_heads, 
      d_model=self.model_dim, d_k=self.key_dim, d_v=self.value_dim, dropout=0.1)
    self.stats_pooling = StatsPooling(var_eps=1e-12, normalize_stats=True)
    self.weights.extend(self.selfattention.weights)

  def forward(self, x, seqmask=None, time_axis=-1):
    """ x is a 3D tensor in shape (B, C1, T) 
        key is a 3D tensor in shape (B, C2, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    query, alignment = self.selfattention(x,x,x)  
    pool_stats = self.stats_pooling(query)
    return pool_stats

class DoubleStatsPooling(nn.Module):
  def __init__(self, normalize_stats=False):
    super(DoubleStatsPooling, self).__init__()
    self.stats_pooling = StatsPooling(var_eps=1e-12, normalize_stats=normalize_stats)
  def forward(self, x, freq_axis=-2, time_axis=-1):
    """ x is a 3D tensor in shape (B, C, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    pool_var, pool_mean = torch.var_mean(
      x, dim=freq_axis, keepdim=False, unbiased=False)
    pool_var = torch.max(pool_var, 1e-12*torch.ones_like(pool_var))
    pool_std = torch.sqrt(pool_var)

    pool_mean_out = self.stats_pooling(pool_mean, time_axis=time_axis)
    pool_std_out = self.stats_pooling(pool_std, time_axis=time_axis)
      
    pool_stats = torch.cat((pool_mean_out, pool_std_out), dim=-1)
    return pool_stats

class AttentiveStatsPooling(nn.Module):
  def __init__(self, input_dim=256, bottle_dim=128, use_global_info=False,
               apply_bn=True, activ="ReLU", normalize_stats=False):
    super(AttentiveStatsPooling, self).__init__()
    """   use  apply_bn=True, activ="ReLU", normalize_stats=False
          or   apply_bn=False, activ="Tanh", normalize_stats=True
    """
    self.use_global_info = use_global_info
    self.normalize_stats = normalize_stats
    self.weights = []

    assert apply_bn != normalize_stats

    if use_global_info:
      self.linear1 = Conv1d(3*input_dim, bottle_dim, kernel_size=1, stride=1, apply_bn=apply_bn, activ=activ, use_bias=True)
    else:
      self.linear1 = Conv1d(input_dim, bottle_dim, kernel_size=1, stride=1, apply_bn=apply_bn, activ=activ, use_bias=True)
    self.linear2 = Conv1d(bottle_dim, input_dim, kernel_size=1, stride=1, use_bias=True)

    self.weights.extend(self.linear1.weights)
    self.weights.extend(self.linear2.weights)
    
  def forward(self, x, channel_axis=-2, time_axis=-1):

    if self.use_global_info:
      n_frm = x.size(time_axis)
      x_mean = x.mean(dim=time_axis, keepdim=True)
      x_var = x.square().mean(dim=time_axis, keepdim=True) - x_mean.square()
      x_std = x_var.clamp(min=1e-12).sqrt()
      x_aug = torch.cat((x, x_mean.expand(-1,-1,n_frm), x_std.expand(-1,-1,n_frm)), dim=channel_axis) 
    else:
      x_aug = x

    aw_norm = F.softmax(self.linear2(self.linear1(x_aug)), dim=time_axis)

    att_h = torch.mul(aw_norm, x)
    mean = torch.sum(att_h,dim=time_axis)
    
    variance = torch.sum(torch.mul(att_h, x),dim=time_axis) - torch.mul(mean,mean)
    variance = torch.max(variance, 1e-12*torch.ones_like(variance))
    
    pool_mean = mean
    pool_std = torch.sqrt(variance)

    if self.normalize_stats:
      pool_mean = l2_normalize(pool_mean, dim=time_axis, eps=1e-12)
      pool_std = l2_normalize(pool_std, dim=time_axis, eps=1e-12)
    pool_stats = torch.cat((pool_mean,pool_std),dim=time_axis)

    return pool_stats

class AttentiveDoubleStatsPooling(nn.Module):
  def __init__(self, input_dim=256, bottle_dim=128, kernel=[1,1],  use_global_info=False,
               apply_bn=True, activ="ReLU", normalize_stats=False, use_att_stats_pool=False, use_both_axis=False):
    super(AttentiveDoubleStatsPooling, self).__init__()
    """   use  apply_bn=True, activ="ReLU"
          or   apply_bn=False, activ="Tanh"
    """
    self.use_att_stats_pool = use_att_stats_pool
    self.use_global_info = use_global_info
    self.normalize_stats = normalize_stats
    self.use_both_axis = use_both_axis
    self.weights = []

    if use_att_stats_pool:
      if normalize_stats:
        self.att_stats_pooling_mean = AttentiveStatsPooling(input_dim=input_dim, bottle_dim=bottle_dim, use_global_info=use_global_info, 
                                                          apply_bn=False, activ="Tanh", normalize_stats=normalize_stats)
        self.att_stats_pooling_std = AttentiveStatsPooling(input_dim=input_dim, bottle_dim=bottle_dim, use_global_info=use_global_info, 
                                                          apply_bn=False, activ="Tanh", normalize_stats=normalize_stats)
      else:
        self.att_stats_pooling_mean = AttentiveStatsPooling(input_dim=input_dim, bottle_dim=bottle_dim, use_global_info=use_global_info, 
                                                          apply_bn=True, activ=activ, normalize_stats=normalize_stats)
        self.att_stats_pooling_std = AttentiveStatsPooling(input_dim=input_dim, bottle_dim=bottle_dim, use_global_info=use_global_info, 
                                                          apply_bn=True, activ=activ, normalize_stats=normalize_stats)    
      self.weights.extend(self.att_stats_pooling_mean.weights)
      self.weights.extend(self.att_stats_pooling_std.weights)                                                                                                                      
    else:
      self.stats_pooling = StatsPooling(var_eps=1e-12, normalize_stats=normalize_stats)


    if use_global_info:
      self.linear1 = Conv2d(3*input_dim, bottle_dim, kernel, stride=(1,1), apply_bn=apply_bn, activ=activ, use_bias=True)
    else:
      self.linear1 = Conv2d(input_dim, bottle_dim, kernel, stride=(1,1), apply_bn=apply_bn, activ=activ, use_bias=True)
    self.weights.extend(self.linear1.weights)  

    self.linear2 = Conv2d(
      bottle_dim, input_dim, kernel, stride=(1,1), padding='same', 
      use_bias=True, apply_bn=False, bn_mom=0.99, bn_eps=1e-6, 
      activ=None, kwargs=dict(), w_init_gain='conv2d', 
      as_class=True)    
    self.weights.extend(self.linear2.weights)

    self.softmax_2d = nn.Softmax2d()

  def forward(self, x, chan_axis=-3, freq_axis=-2, time_axis=-1):
    """ x is a 3D tensor in shape (B, C, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    if self.use_global_info:
      n_frm = x.size(time_axis)
      n_frq = x.size(freq_axis)
      x_mean = x.mean(dim=(freq_axis, time_axis), keepdim=True)
      x_var = x.square().mean(dim=(freq_axis, time_axis), keepdim=True) - x_mean.square()
      x_std = x_var.clamp(min=1e-12).sqrt()
      x_aug = torch.cat((x, x_mean.expand(-1,-1,n_frq,n_frm), x_std.expand(-1,-1,n_frq,n_frm)), dim=chan_axis) 
    else:
      x_aug = x 

    lin_out = self.linear2(self.linear1(x_aug))
    
    if self.use_both_axis:
      aw_norm = self.softmax_2d(lin_out)
    else:
      aw_norm = F.softmax(lin_out, dim=freq_axis)
    att_h = torch.mul(aw_norm, x)

    pool_mean = torch.sum(att_h,dim=freq_axis)
    pool_var = torch.sum(torch.mul(att_h, x),dim=freq_axis) - torch.mul(pool_mean,pool_mean)
    pool_var = torch.max(pool_var, 1e-12*torch.ones_like(pool_var))
    pool_std = torch.sqrt(pool_var)

    if self.use_att_stats_pool:
      pool_mean_out = self.att_stats_pooling_mean(pool_mean)
      pool_std_out = self.att_stats_pooling_std(pool_std)
    else:
      pool_mean_out = self.stats_pooling(pool_mean)
      pool_std_out = self.stats_pooling(pool_std)

    pool_stats = torch.cat((pool_mean_out, pool_std_out), dim=time_axis)

    return pool_stats

class DoubleStatsPooling_new_dim(nn.Module):
  def __init__(self, normalize_stats=False):
    super(DoubleStatsPooling_new_dim, self).__init__()
    self.stats_pooling = DoubleStatsPooling(normalize_stats=normalize_stats)

    self.weights = []

  def forward(self, x, chan_axis=-3, freq_axis=-2, time_axis=-1):
    """ x is a 3D tensor in shape (B, C, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    x1, x2, x3, x4, x5 = torch.split(x, 4, dim=freq_axis)
    x4 = torch.cat((x4, x5), dim=freq_axis)
    x_mean = torch.mean(x, dim=freq_axis, keepdim=True)

    pool_1 = self.stats_pooling(torch.cat((x1,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)
    pool_2 = self.stats_pooling(torch.cat((x2,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)
    pool_3 = self.stats_pooling(torch.cat((x3,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)
    pool_4 = self.stats_pooling(torch.cat((x4,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)

    pool_stats = torch.mean(torch.cat((pool_1, pool_2, pool_3, pool_4), dim=freq_axis), dim=freq_axis)

    return pool_stats

class DoubleStatsPooling_new_dim2(nn.Module):
  def __init__(self, normalize_stats=False):
    super(DoubleStatsPooling_new_dim2, self).__init__()
    self.weights = []
    self.stats_pooling1 = AttentiveDoubleStatsPooling(input_dim=128, bottle_dim=128, kernel=[1,1],  use_global_info=False,
                                                      apply_bn=True, activ="ReLU", normalize_stats=False, use_att_stats_pool=True, use_both_axis=False)
    self.weights.extend(self.stats_pooling1.weights)
    self.stats_pooling2 = AttentiveDoubleStatsPooling(input_dim=128, bottle_dim=128, kernel=[1,1],  use_global_info=False,
                                                      apply_bn=True, activ="ReLU", normalize_stats=False, use_att_stats_pool=True, use_both_axis=False)
    self.weights.extend(self.stats_pooling2.weights)
    self.stats_pooling3 = AttentiveDoubleStatsPooling(input_dim=128, bottle_dim=128, kernel=[1,1],  use_global_info=False,
                                                      apply_bn=True, activ="ReLU", normalize_stats=False, use_att_stats_pool=True, use_both_axis=False)
    self.weights.extend(self.stats_pooling3.weights)
    self.stats_pooling4 = AttentiveDoubleStatsPooling(input_dim=128, bottle_dim=128, kernel=[1,1],  use_global_info=False,
                                                      apply_bn=True, activ="ReLU", normalize_stats=False, use_att_stats_pool=True, use_both_axis=False)
    self.weights.extend(self.stats_pooling4.weights)

  def forward(self, x, chan_axis=-3, freq_axis=-2, time_axis=-1):
    """ x is a 3D tensor in shape (B, C, T) 
        seqmask is a 2D tensor in shape (B, T)
    """
    x1, x2, x3, x4, x5 = torch.split(x, 4, dim=freq_axis)
    x4 = torch.cat((x4, x5), dim=freq_axis)
    x_mean = torch.mean(x, dim=freq_axis, keepdim=True)

    pool_1 = self.stats_pooling1(torch.cat((x1,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)
    pool_2 = self.stats_pooling2(torch.cat((x2,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)
    pool_3 = self.stats_pooling3(torch.cat((x3,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)
    pool_4 = self.stats_pooling4(torch.cat((x4,x_mean), dim=freq_axis)).unsqueeze(dim=freq_axis)

    pool_stats = torch.mean(torch.cat((pool_1, pool_2, pool_3, pool_4), dim=freq_axis), dim=freq_axis)

    return pool_stats

if __name__=="__main__":
  ## StatsPooling
  var_eps, normalize_stats = 1e-12, True
  x = torch.randn(32, 1500, 300)
  pooling = StatsPooling(var_eps, normalize_stats)
  y = pooling(x)
  print(y.size())

  ## MHAPooling
  from hparams import Hyperparams as hp
  num_heads, attn_opts = hp.num_attention_heads, hp.attention_opts
  x = torch.randn(32, 1500, 300)
  key = torch.randn(32, 512, 300)
  pooling = MHAPooling(num_heads, attn_opts, var_eps, normalize_stats)
  print(pooling.key_networks)
  print(pooling.query_network)
  print(len(pooling.weights))
  # from torch_utils import to_gpu
  # pooling.cuda()
  # x, key = to_gpu(x), to_gpu(key)
  y = pooling(x, key=key)
  print(y.size())
  exit()

  ## LDEPooling
  # in_dim, n_codes, n_heads = 192, 64, 2
  # normalize_codes, norm_eps = True, 1e-12
  # x = torch.randn(32, in_dim, 300)
  # # pooling = LDEPooling(in_dim, n_codes, normalize_codes, norm_eps)
  # pooling = MHLDEPooling(in_dim, n_codes, n_heads, normalize_codes, norm_eps)
  # y = pooling(x)
  # y = y.view(y.size(0), in_dim*n_codes)
  # print(y.size())
