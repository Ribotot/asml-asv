import os
import struct
from collections import defaultdict as ddict
from collections import OrderedDict as odict
from tqdm import tqdm

from kaldi_io.arkio import read_token, read_kaldi_binary, write_kaldi_binary
from kaldi_io.compressed_matrix import GlobalHeader

def zpad_string(string, length=3):
  string = str(string)
  if len(string) == length:
    return string
  else:
    res = length - len(string)
    return '0'*res + string


def make_utt2spk(scplist, utt2spk, delimiter='/'):
  if isinstance(scplist, str):
    scplist = [scplist]

  with open(utt2spk, 'wt') as f:
    for scpfile in scplist:
      for scpline in open(scpfile):
        uttID = scpline.split(' ')[0]
        spkID = uttID.split(delimiter)[0]
        f.write('{} {}\n'.format(uttID, spkID))

def get_spk2utt(scplist, unique=False, postfix_list=None):
  if isinstance(scplist, str):
    scplist = [scplist]
  for scpfile in scplist:
    assert os.path.isfile(scpfile), '%s does not exist!' % scpfile

  if unique:
    spk2uttq = ddict(set)
    for scpfile in scplist:
      for scpline in open(scpfile):
        if scpline != '':
          uttID = scpline.split(' ', 1)[0]
          spkID = uttID.split('/', 1)[0]
          uttID_uniq = uttID
          for postfix in postfix_list:
            uttID_uniq = uttID_uniq.replace(postfix, '')
          spk2uttq[spkID].add(uttID_uniq)
    ## set >> list
    for spkID in spk2uttq:
      spk2uttq[spkID] = list(spk2uttq[spkID])
    return spk2uttq
  else:
    spk2utt = ddict(list)
    for scpfile in scplist:
      for scpline in open(scpfile):
        if scpline != '':
          uttID = scpline.split(' ', 1)[0]
          spkID = uttID.split('/', 1)[0]
          spk2utt[spkID].append(uttID)
    return spk2utt

def get_uttq2utts(scplist, postfix_list=None):
  if isinstance(scplist, str):
    scplist = [scplist]
  for scpfile in scplist:
    assert os.path.isfile(scpfile), '%s does not exist!' % scpfile

  uttq2utts = ddict(list)
  for scpfile in scplist:
    for scpline in open(scpfile):
      if scpline != '':
        uttID = scpline.split(' ', 1)[0]
        uttID_uniq = uttID
        for postfix in postfix_list:
          uttID_uniq = uttID_uniq.replace(postfix, '')
        uttq2utts[uttID_uniq].append(uttID)
  return uttq2utts


def get_uttinfo(fpath):
  spkID, uttID = fpath.split('/')[-2:]
  uttID = spkID + '/' + uttID.rsplit('.', 1)[0]
  return spkID, uttID

def append_uttids(scp_in, scp_out=None, postfix=''):
  ## Single scp file
  if isinstance(scp_in, str):
    assert postfix != ''
    assert os.path.isfile(scp_in), '%s does not exist!' % scp_in
    if scp_in == scp_out:
      os.rename(scp_in, scp_in+'.tmp')
      scp_in = scp_in+'.tmp'

    print('appending "%s" to %s...' % (postfix, scp_in))
    with open(scp_out, 'wt') as f:
      for line in open(scp_in):
        uttID, rest = line.split(' ', 1)
        newline = " ".join([uttID+postfix, rest])
        f.write(newline)
    os.remove(scp_in)
  ## List of scp files
  elif isinstance(scp_in, list):
    assert scp_out is None
    for _scp_in in scp_in:
      append_uttids(scp_in=_scp_in, scp_out=_scp_in, postfix=postfix)

def prepend_uttids(scp_in, scp_out=None, prefix=''):
  ## Single scp file
  if isinstance(scp_in, str):
    assert prefix != ''
    assert os.path.isfile(scp_in), '%s does not exist!' % scp_in
    if scp_in == scp_out:
      os.rename(scp_in, scp_in+'.tmp')
      scp_in = scp_in+'.tmp'

    print('prepending "%s" to %s...' % (prefix, scp_in))
    with open(scp_out, 'wt') as f:
      for line in open(scp_in):
        uttID, rest = line.split(' ', 1)
        newline = " ".join([prefix+uttID, rest])
        f.write(newline)
    os.remove(scp_in)
  ## List of scp files
  elif isinstance(scp_in, list):
    assert scp_out is None
    for _scp_in in scp_in:
      prepend_uttids(scp_in=_scp_in, scp_out=_scp_in, prefix=prefix)

def append_lines(scp_in, scp_out=None, postfix='', pos=0, delim=" "):
  """ Append <postfix> to the end of <pos>-split line """

  ## Single scp file
  if isinstance(scp_in, str):
    assert postfix != ''
    assert os.path.isfile(scp_in), '%s does not exist!' % scp_in
    if scp_in == scp_out:
      os.rename(scp_in, scp_in+'.tmp')
      scp_in += '.tmp'

    print('appending "%s" to %s...' % (postfix, scp_in))
    with open(scp_out, 'wt') as f:
      for line in open(scp_in):
        line = line.strip()
        if pos >= 0:
          line_split = line.split(delim)
          line = delim.join(line_split[:pos] + [line_split[pos]+postfix] + line_split[pos+1:])
        elif pos == -1:
          line = delim.join([ll+postfix for ll in line.split(delim)])
        f.write(line+'\n')

  ## List of scp files
  elif isinstance(scp_in, list):
    assert scp_out is None
    for _scp_in in scp_in:
      append_lines(scp_in=_scp_in, scp_out=_scp_in, 
                   postfix=postfix, pos=pos, delim=delim)

def prepend_lines(scp_in, scp_out=None, prefix='', pos=0, delim=" "):
  """ Prepend <prefix> to the beginning of <pos>-split line """

  ## Single scp file
  if isinstance(scp_in, str):
    assert prefix != ''
    assert os.path.isfile(scp_in), '%s does not exist!' % scp_in
    if scp_in == scp_out:
      os.rename(scp_in, scp_in+'.tmp')
      scp_in += '.tmp'

    print('prepending "%s" to %s...' % (prefix, scp_in))
    with open(scp_out, 'wt') as f:
      for line in open(scp_in):
        line = line.strip()
        if pos >= 0:
          line_split = line.split(delim)
          line = delim.join(line_split[:pos] + [prefix+line_split[pos]] + line_split[pos+1:])
        elif pos == -1:
          line = delim.join([prefix+ll for ll in line.split(delim)])
        f.write(line+'\n')

  ## List of scp files
  elif isinstance(scp_in, list):
    assert scp_out is None
    for _scp_in in scp_in:
      prepend_lines(scp_in=_scp_in, scp_out=_scp_in, 
                    prefix=prefix, pos=pos, delim=delim)


def gather_scps(scplist, scp_out):
  assert isinstance(scplist, list)
  for scpfile in scplist:
    assert os.path.isfile(scpfile), '%s does not exist!' % scpfile
  os.makedirs(os.path.dirname(scp_out), exist_ok=True)

  print('writing to %s...' % scp_out)
  with open(scp_out, 'wt') as f:
    for scpfile in scplist:
      for line in open(scpfile):
        f.write(line)

def gather_arks(arklist, ark_out, compress=2, endian='<'):
  ## Must get a list of arkfile paths
  assert isinstance(arklist, list) and len(arklist) > 1
  os.makedirs(os.path.dirname(ark_out), exist_ok=True)

  with open(ark_out, 'wb') as f:
    # offset = 0
    for arkfile in arklist:
      ext = arkfile.rsplit('.', 1)[-1]
      assert ext == 'ark', 'file extension must be "ark"'
      assert os.path.isfile(arkfile), "%s does not exist" % arkfile
      print('opened {}...'.format(arkfile))

      fd = open(arkfile, 'rb')
      key = read_token(fd)  # uttID
      while key:
        array = read_kaldi_binary(fd, endian=endian)
        write_kaldi_binary(f, array, key, compress, endian)
        key = read_token(fd)

def make_scp(arkfile, endian='<', scp_out=''):
  if isinstance(arkfile, list):
    for _arkfile in arkfile:
      assert os.path.isfile(_arkfile), '%s does not exist!' % _arkfile
    for _arkfile in arkfile:
      make_scp(_arkfile, endian)

    if scp_out != '':
      os.makedirs(os.path.dirname(scp_out), exist_ok=True)
      scplist = [_arkfile.replace('.ark', '.scp') for _arkfile in arkfile]
      gather_scps(scplist, scp_out)

  elif isinstance(arkfile, str):
    ext = arkfile.rsplit('.', 1)[-1]
    assert os.path.isfile(arkfile), '%s does not exist!' % arkfile
    assert ext == 'ark', 'file extension must be "ark"'

    scpfile = arkfile.rsplit('.', 1)[0] + '.scp'
    print('writing %s...' % scpfile)
    with open(scpfile, 'wt') as f:
      offset = 0

      fd = open(arkfile, 'rb')
      key = read_token(fd)  # uttID
      while key:
        ## size of "uttID"
        offset += len(key) + 1  # uttID + ' '

        ## Write scpline to scp file
        scpline = '{} {}:{}'.format(key, arkfile, offset)
        f.write(scpline+'\n')

        ## size of "binary" token
        assert fd.read(2).decode() == '\0B'
        offset += 2

        ## Count "header"
        header = read_token(fd)
        offset += len(header) + 1
        if header in ['CM', 'CM2', 'CM3']:
          gbhead = GlobalHeader.read(fd, header, endian)
          offset += 16  # size of GlobalHeader
          if header == 'CM':
            fd.seek(8 * gbhead.cols, 1)
            offset += 8 * gbhead.cols  # size of PerColHeader
            size_of_data = gbhead.rows * gbhead.cols # sample_size = 1 for 'uint8'

        elif header in ['FM', 'FV', 'DM', 'DV']:
          ## matrix/vector shape info
          assert fd.read(1) == b'\4' # 'int8'
          rows = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'
          offset += 5
          if 'M' in header:
            assert fd.read(1) == b'\4' # 'int8'
            cols = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'
            offset += 5
          else:
            cols = 1

          sample_size = 4 if header[0] == 'F' else 8
          size_of_data = rows * cols * sample_size

        fd.seek(size_of_data, 1)
        offset += size_of_data
        key = read_token(fd)


def make_utt2len(arkfile, endian='<', utt2len=''):
  if isinstance(arkfile, list):
    for _arkfile in arkfile:
      assert os.path.isfile(_arkfile), '%s does not exist!' % _arkfile
    for _arkfile in arkfile:
      make_utt2len(_arkfile, endian)

    if utt2len != '':
      scplist = [_arkfile.replace('.ark', '.len') for _arkfile in arkfile]
      gather_scps(scplist, utt2len)

  elif isinstance(arkfile, str):
    ext = arkfile.rsplit('.', 1)[-1]
    assert os.path.isfile(arkfile), '%s does not exist!' % arkfile
    assert ext == 'ark', 'file extension must be "ark"'

    scpfile = arkfile.rsplit('.', 1)[0] + '.len'
    print('writing %s...' % scpfile)
    with open(scpfile, 'wt') as f:
      fd = open(arkfile, 'rb')
      key = read_token(fd)  # uttID
      while key:
        ## size of "binary" token
        assert fd.read(2).decode() == '\0B'

        ## Count "header"
        header = read_token(fd)
        if header in ['CM', 'CM2', 'CM3']:
          gbhead = GlobalHeader.read(fd, header, endian)
          rows = gbhead.rows
          if header == 'CM':
            fd.seek(8 * gbhead.cols, 1)
            size_of_data = gbhead.rows * gbhead.cols # sample_size = 1 for 'uint8'

        elif header in ['FM', 'FV', 'DM', 'DV']:
          ## matrix/vector shape info
          assert fd.read(1) == b'\4' # 'int8'
          rows = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'
          if 'M' in header:
            assert fd.read(1) == b'\4' # 'int8'
            cols = struct.unpack(endian+'i', fd.read(4))[0] # 'int32'
          else:
            cols = 1

          sample_size = 4 if header[0] == 'F' else 8
          size_of_data = rows * cols * sample_size

        line = '{} {}'.format(key, rows)
        f.write(line+'\n')

        fd.seek(size_of_data, 1)
        key = read_token(fd)

def make_spk2idx(feats_scp, spk2idx='spk2idx', endian='<'):
  spkIDset = set()
  if isinstance(feats_scp, str):
    assert os.path.isfile(feats_scp), '%s does not exist' % feats_scp
    feats_scp = [feats_scp]
  elif isinstance(feats_scp, list):
    for scpfile in feats_scp:
      assert os.path.isfile(scpfile), '%s does not exist' % scpfile

  for scpfile in feats_scp:
    for line in open(scpfile):
      spkIDset.add(line.split(' ', 1)[0].split('/')[-2])

  with open(spk2idx, 'wt') as f:
    for spk_idx, spkID in enumerate(sorted(list(spkIDset))):
      f.write('{} {}\n'.format(spkID, spk_idx))

def make_reco2dur(wpathlist, fs, reco2dur):
  os.makedirs(os.path.dirname(reco2dur), exist_ok=True)
  with open(reco2dur, 'wt') as f:
    for wpath in tqdm(wpathlist):
      wpath = wpath.rstrip()
      _, uttID = get_uttinfo(wpath)
      nsamp = nsamples(wpath)
      f.write('{} {}\n'.format(uttID, nsamp/float(fs)))

def nsamples(fname):
  fid = open(fname, 'rb')
  try:
    fsize, endian = _read_riff_chunk(fid)  # file size in bytes
    while fid.tell() < 44:
      ## read the next chunk
      chunk_id = fid.read(4)
      if chunk_id == b'fmt ':
        size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid, endian)
      elif chunk_id == b'data':
        size = struct.unpack(endian+'i', fid.read(4))[0]
        # assert fid.tell() == 44
        break
      elif chunk_id == b'fact':
        _skip_unknown_chunk(fid, endian)
      elif chunk_id == b'LIST':
        ## Someday this could be handled properly but for now skip it
        _skip_unknown_chunk(fid, endian)
      else:
        warnings.warn("Chunk (non-data) not understood, skipping it.", WavFileWarning)
        _skip_unknown_chunk(fid, endian)
  finally:
    fid.close()
  return int(size/ba)

def compute_gmv_stats(arklist=None, feats_scp=None, gmv_path='gmv.npy', endian='<'):
  raise NotImplementedError

  dnum = 0
  gmean_in = np.empty([1, hp.feat_dim], dtype='float64')
  gmean2_in = np.empty([1, hp.feat_dim], dtype='float64')
  gmean_out = np.empty([1, hp.feat_dim], dtype='float64')
  gmean2_out = np.empty([1, hp.feat_dim], dtype='float64')

  for _ in range(3):
    for fdicts in TBG(datacomposer.minibatch(mb_size=hp.upb)):
      fmat_in = np.vstack([fdict['mfbe_ane_ns'] for fdict in fdicts])
      fmat_in = fmat_in.reshape(-1, hp.feat_dim)
      gmean_in += np.sum(fmat_in, axis=0)
      gmean2_in += np.sum(fmat_in**2, axis=0)
      dnum += fmat_in.shape[0]

      fmat_out = np.vstack([fdict['mfbe_ane_cs'] for fdict in fdicts])
      fmat_out = fmat_out.reshape(-1, hp.feat_dim)
      gmean_out += np.sum(fmat_out, axis=0)
      gmean2_out += np.sum(fmat_out**2, axis=0)
  ## Input statistics
  gmean_in = 0.5 * (gmean_in/float(dnum) + gmean_out/float(dnum))
  gmean2_in = 0.5 * (gmean2_in/float(dnum) + gmean2_out/float(dnum))
  gstd_in = np.sqrt(gmean2_in - gmean_in**2)
  ## Output statistics
  gmean_out /= float(dnum)
  gmean2_out /= float(dnum)
  gstd_out = np.sqrt(gmean2_out - gmean_out**2)
  ## Make a dictionary
  gmv_dict = dict(gmean_in=gmean_in, gstd_in=gstd_in, gmean2_in=gmean2_in, 
                  gmean_out=gmean_out, gstd_out=gstd_out, gmean2_out=gmean2_out, 
                  dnum_in=2*dnum, dnum_out=dnum)
  ## Convert to float32
  for key in gmv_dict.keys():
    gmv_dict[key] = gmv_dict[key].astype('float32')
  ## Save
  save_dict_npy(gmv_path, gmv_dict)
