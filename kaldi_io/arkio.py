#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys, os, re, gzip, struct
import random
from six.moves import range

from kaldi_io.compressed_matrix import GlobalHeader, PerColHeader

class ArchiveReader(object):
  """Read kaldi features"""
  def __init__(self, list_or_iter=None):
    self.fd = dict()
    if list_or_iter is not None:
      arkfiles = set()
      for arkp in list_or_iter:
        arkfile = arkp.split(':')[0]
        arkfiles.add(arkfile)

      for arkfile in arkfiles:
        self.fd[arkfile] = open(arkfile, 'rb')
        # print('opened %s...' % arkfile)

  def close(self):
    for name in self.fd:
      self.fd[name].close()

  def read_mat2d(self, arkfile_offset, start=0, length=0):
    arkfile, offset = arkfile_offset.rsplit(':', 1)
    # if arkfile not in self.fd:
    #   fd = open(arkfile, 'rb')
    #   assert fd is not None
    #   self.fd[arkfile] = fd
    #   print('opened {}...'.format(arkfile))

    ## Move the file descriptor to the "offset" position
    self.fd[arkfile].seek(int(offset))
    try:
      mat = read_kaldi_binary(self.fd[arkfile], start, length)
    except:
      raise IOError("Cannot read features from %s" % arkfile_offset)
    return mat

  def read_mat3d(self, arkfile_offset, start=0, length=0):
    pass

#################################################
# Data-type independent helper functions,

def open_or_fd(file, mode='rb'):
  """ fd = open_or_fd(file)
   Open file, gzipped file, pipe, or forward the file-descriptor.
   Eventually seeks in the 'file' argument contains ':offset' suffix.
  """
  offset = None
  try:
    if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
      # print("\t strip 'ark:' prefix from r{x,w}filename (optional),")
      prefix, file = file.split(':', 1)

    if re.search(':[0-9]+$', file):
      # print("\t separate offset from filename (optional),")
      file, offset = file.rsplit(':', 1)

    if file[-1] == '|':
      # print("\t input pipe?")
      fd = popen(file[:-1], 'rb') # custom,
    elif file[0] == '|':
      # print("\t output pipe?")
      fd = popen(file[1:], 'wb') # custom,
    elif file.split('.')[-1] == 'gz':
      # print("\t is it gzipped?")
      fd = gzip.open(file, mode)
    else:
      # print("\t a normal file...")
      fd = open(file, mode)
  except TypeError:
    # print("\t opened file descriptor...")
    fd = file
  # print('fd = {}'.format(fd))
  # Eventually seek to offset,
  if offset != None: fd.seek(int(offset))
  return fd

# based on '/usr/local/lib/python3.4/os.py'
def popen(cmd, mode="rb"):
  if not isinstance(cmd, str):
    raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

  import subprocess, io, threading

  # cleanup function for subprocesses,
  def cleanup(proc, cmd):
    ret = proc.wait()
    if ret > 0:
      raise ValueError('cmd %s returned %d !' % (cmd,ret))
    return

  # text-mode,
  if mode == "r":
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdout)
  elif mode == "w":
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdin)
  # binary,
  elif mode == "rb":
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdout
  elif mode == "wb":
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdin
  # sanity,
  else:
    raise ValueError("invalid mode %s" % mode)


def read_token(fd):
  token = ''
  while 1:
    char = fd.read(1).decode("latin1")
    if char == ' ' or char == '':
      break
    token += char
  token = token.strip()
  if token == '':
    return None # end of file,
  assert(re.match('^\S+$',token) != None) # check format (no whitespace!)
  return token


#################################################
## Integer vectors (alignments, ...),

def read_int32vec_arkscp(ark_or_scp):
  ext = ark_or_scp.rsplit('.', 1)[-1]
  assert ext in ['ark', 'scp'], 'Invalid file extention "%s"' % ext

  if ext == 'ark':
    fd = open_or_fd(ark_or_scp)  # normal file
    try:
      key = read_token(fd)
      while key:
        arr = read_int32vec_binary(fd)
        yield key, arr
        key = read_token(fd)
    finally:
      if fd is not ark_or_scp: fd.close()

  elif ext == 'scp':
    fd = open_or_fd(ark_or_scp)
    try:
      for line in fd:  # fd is an opened file descriptor
        key, arkp = line.decode().split(' ')
        arr = read_int32vec_binary(arkp)
        yield key, arr
    finally:
      if fd is not ark_or_scp : fd.close()


def read_int32vec_binary(fd, start=0, length=0, endian='<'):
  if isinstance(fd, str):
    arkfile, offset = fd.rstrip().split(':')
    fd = open(arkfile, 'rb')
    fd.seek(int(offset))
  assert fd.read(2).decode() == '\0B'
  assert not ((length == 0) and (start is None))

  ## No header in this type
  assert fd.read(1) == b'\4' # 'int8'
  rows = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'

  ## Set appropriate length (i.e., the number of rows)
  if length == 0:
    length = rows - start
  elif start is None:
    start = random.randint(0, rows-length)
  res = rows - length
  assert res >= start
  rows = length
  col_left = res - start

  fd.seek(start * 5, 1)  # skip to the "start" position
  ## Option 1: single liner
  vec = np.frombuffer(fd.read(length * 5), dtype=[('size','int8'), ('value','int32')], count=length)
  fd.seek(col_left * 5, 1)
  assert vec[0]['size'] == 4  # 'int32'
  return vec[:]['value']  # values are in 2nd column
  ## Option 2: for loop
  vec = np.empty(length, dtype=np.int32)
  for i in range(length):
    assert fd.read(1) == b'\4' # 'int8'
    vec[i] = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'
  fd.seek(col_left * 5, 1)
  return vec


#################################################
## Float matrices (features, transformations, ...),

def read_kaldi_arkscp(ark_or_scp, endian='<'):
  ext = ark_or_scp.rsplit('.', 1)[-1]
  assert ext in ['ark', 'scp'], 'Invalid file extention "%s"' % ext

  if ext == 'ark':
    fd = open_or_fd(ark_or_scp)  # normal file
    try:
      key = read_token(fd)
      while key:
        arr = read_kaldi_binary(fd, endian=endian)
        yield key, arr
        key = read_token(fd)
    finally:
      if fd is not ark_or_scp: fd.close()

  elif ext == 'scp':
    fd = open_or_fd(ark_or_scp)
    try:
      for line in fd:  # fd is an opened file descriptor
        key, arkp = line.decode().split(' ', 1)
        arr = read_kaldi_binary(arkp, endian=endian)
        yield key, arr
    finally:
      if fd is not ark_or_scp : fd.close()


def read_kaldi_binary(fd, start=0, length=0, endian='<'):
  """ 
  Note that "length=0" basically implies full utterance.

  Options with (start, length)
    (0, 0): read without crop
    (0, l): read "length" rows from scratch
    (t, 0): crop from "start" to the end
    (t, l): crop from "start" to "start+length"
    (None, 0): not allowed (*never necessary)
    (None, l): set random start pos and read "length" rows
  """
  if isinstance(fd, str):
    arkfile, offset = fd.rstrip().split(':')
    fd = open(arkfile, 'rb')
    fd.seek(int(offset))
  assert fd.read(2).decode() == '\0B'
  assert not ((length == 0) and (start is None))

  ## Data type :: 'Compressed Matrix', 'Float/Double Matrix/Vector'
  header = read_token(fd)

  ## 'CM', 'CM2', 'CM3' are possible values,
  if header == 'CM':
    dtype_cm = np.dtype(endian+'u1')  # dtype_cm = 'uint8'

    ## Read global header,
    gbhead = GlobalHeader.read(fd, header, endian)
    pcheads = PerColHeader.read(fd, gbhead)
    rows, cols = gbhead.rows, gbhead.cols

    ## Set appropriate length (i.e., the number of rows)
    if length == 0:
      length = rows - start  # crop from "start" to the end of utterance
    elif start is None:
      start = random.randint(0, rows-length)  # set random start position to crop
    res = rows - length
    assert res >= start
    rows = length
    col_left = res - start

    ## Read data
    mat = np.zeros((cols, rows), dtype='float32')
    ## Special treatment for the first pchead (to prevent negative pointer)
    fd.seek(start, 1)
    mat[0] = PerColHeader.char_to_float(
                np.frombuffer(fd.read(rows), dtype=dtype_cm, count=rows), 
                *gbhead.uint_to_float(pcheads[0]).tolist())
    ## The rest pcheads can be dealt with a loop
    for i, pchead in enumerate(pcheads[1:], start=1):
      fd.seek(res, 1)
      mat[i] = PerColHeader.char_to_float(
                  np.frombuffer(fd.read(rows), dtype=dtype_cm, count=rows), 
                  *gbhead.uint_to_float(pchead).tolist())
    fd.seek(col_left, 1)  # as if we read the whole data
    return mat.T  # transpose (from <column-major> to <row-major>)

  ## 'FM', 'FV', 'DM', 'DV' are possible values,
  else:
    if header == 'FM' or header == 'FV':
      dtype, sample_size = endian+'f', 4
    elif header == 'DM' or header == 'DV':
      dtype, sample_size = endian+'d', 8
    else:
      raise ValueError('Unknown header "%s" is contained' % header)

    ## Dimensions of matrix
    ## Option 1 ##
    assert fd.read(1) == b'\4' # 'int8'
    rows = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'
    if 'M' in header:
      assert fd.read(1) == b'\4' # 'int8'
      cols = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'
    else:
      cols = 1
    ## Option 2 ##
    # if 'M' in header:
    #   s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
    # else:
    #   s1, rows = np.frombuffer(fd.read(5), dtype='int8,int32')
    #   cols = 1

    ## Set appropriate length (i.e., the number of rows)
    if length == 0:
      length = rows - start
    elif start is None:
      start = random.randint(0, rows-length)
    res = rows - length
    assert res >= start
    rows = length
    col_left = res - start

    ## Read data
    stride_row = cols * sample_size
    fd.seek(start * stride_row, 1)  # skip to the "start" position
    buf = fd.read(rows * stride_row)
    vec = np.frombuffer(buf, dtype=np.dtype(dtype))
    fd.seek(col_left * stride_row, 1)  # as if we read the whole data

    if 'M' in header:
      return np.reshape(vec, (rows, cols))
    return vec


## Writing,
def write_kaldi_binary(file_or_fd, array, key, compress=0, endian='<'):
  """ 
  with open(arkfile, 'wb') as f:
    kaldi_io.write_kaldi_binary(f, array, key='uttID', compress=2)
  """
  assert isinstance(array, np.ndarray)
  fd = open_or_fd(file_or_fd, mode='wb')
  if sys.version_info[0] == 3: assert(fd.mode == 'wb')

  ## Write utterance identifier (uttID)
  fd.write((key+' ').encode("latin1")) # uttID
  ## Write "binary" token ('\0B')
  fd.write(b'\0B')

  ndim, dtype = array.ndim, array.dtype
  if compress:
    assert ndim == 2 and dtype == np.float32, "%sd array with %s" % (ndim, dtype)
    ## Compute & Write global header
    gbhead = GlobalHeader.compute(array, compress, endian)
    gbhead.write(fd)
    if gbhead.type == 'CM':
      ## Compute & Write per-column headers
      pcheads = PerColHeader.compute(array, gbhead)
      pcheads.write(fd, gbhead)

      ## Write data
      array = pcheads.float_to_char(array.T)
      fd.write(array.tobytes())

  elif dtype == np.float32 or dtype == np.float64:
    assert 0 < ndim < 3
    if ndim == 1:
      ## Write token
      header = b'FV ' if dtype == np.float32 else b'DV '
      fd.write(header)

      ## Dim info
      fd.write(b'\04') # 'int8'
      fd.write(struct.pack(endian+'i', array.shape[0])) # 'int32'

    elif ndim == 2:
      ## Write token
      header = b'FM ' if dtype == np.float32 else b'DM '
      fd.write(header)

      rows, cols = array.shape
      ## Row info
      fd.write(b'\04') # 'int8'
      fd.write(struct.pack(endian+'i', rows)) # 'int32'
      ## Column info
      fd.write(b'\04') # 'int8'
      fd.write(struct.pack(endian+'i', cols)) # 'int32'

    if endian not in dtype.str:
      array = array.astype(dtype.newbyteorder())
    ## Write data
    fd.write(array.tobytes())

  elif dtype == np.int32:
    ## Dim info
    fd.write(b'\4') # 'int8'
    fd.write(struct.pack(endian+'i', array.shape[0])) # 'int32'
    for elem in array:
      fd.write(b'\4')
      fd.write(struct.pack(endian+'i', elem))

  else:
    raise ValueError('Unsupported dtype %s' % dtype)


#################################################
# Segments related,
#

# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
def read_segments_as_bool_vec(segments_file):
  """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
   using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
   - t-beg, t-end is in seconds,
   - assumed 100 frames/second,
  """
  segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
  # Sanity checks,
  assert(len(segs) > 0) # empty segmentation is an error,
  assert(len(np.unique([rec[1] for rec in segs ])) == 1) # segments with only 1 wav-file,
  # Convert time to frame-indexes,
  start = np.rint([100 * rec[2] for rec in segs]).astype(int)
  end = np.rint([100 * rec[3] for rec in segs]).astype(int)
  # Taken from 'read_lab_to_bool_vec', htk.py,
  frms = np.repeat(np.r_[np.tile([False,True], len(end)), False],
           np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
  assert np.sum(end-start) == np.sum(frms)
  return frms

