'''
This code is not finished 
DO NOT use it with your data!
'''

import os
import sys
import argparse
import shlex
import time
import subprocess
import random
import h5py


ext = ('.h5',) # legal extensions of hdf5 file


parser = argparse.ArgumentParser(description='Process ptrepack to shrink overwritten .h5 file size.')
parser.add_argument('integers', metavar='path', type=str, nargs='?', help='path of a file or a folder (repack all .h5 files in the folder).')
args = parser.parse_args()
parser.print_help() # print the help information when run the code w/ -h or --help. Can be commonted if don't want to show

# command to process the repack
def repack_command(inpath):
    indir = os.path.dirname(inpath)
    infilename, infileext = os.path.splitext(inpath)
    tempfilename = 'tempfile' + str(random.randrange(1e5, 9e5)) + infileext
    temppath = os.path.join(indir, tempfilename) # use the same folder to save the tempfile
    print('repacking ' + inpath)
    print('to ' + temppath + ' ...')
    print('Original file size: {}'.format(os.path.getsize(inpath)))

    # ptrepack seams does not work with 
    # command = ["ptrepack", "-o", "--chunkshape=auto", "--propindexes", inpath, temppath]
    # p = subprocess.Popen(command)

    # just use simple creating a new file and copy the data
    with h5py.File(inpath, 'r') as fh_in:
        with h5py.File(temppath, 'w') as fh_out:
            fh_out.create_group('data')
            fh_out.create_group('raw')
            fh_out.create_group('prop')
            
            # copy attribute of file
            for attr in fh_in.attrs:
                print('attr: ', attr)
                fh_out.attrs[attr] = fh_in.attrs[attr]
            # copy datasets and groups
            for key in fh_in.keys():
                print('key: ', key)
                fh_in.copy(key, fh_out[key], shallow=True)

    # delete tempfile
    print('removing original file ...')
    os.remove(inpath)
    print('saving tempfile back to ' + inpath + ' ...')
    os.rename(temppath, inpath)
    print('New file size: {}'.format(os.path.getsize(inpath)))

    print('done for ' + inpath)
    print('\n')

print('\n===============\n')
# check the args
if len(sys.argv[1:]) > 1: # path is given
    paths = sys.argv[1:]
else: # no path given and input
    paths = shlex.split(input('Tpye path(s): '))

# get unique paths
    paths = list(set(paths))

# check file/folder exist
paths = list(filter(lambda p: (os.path.exists(p) and p.endswith(ext)) or os.path.isdir(p), paths))

print('Input unique path(s):\n{}'.format('\n'.join(paths)))

print('\n\n')
if not paths: # no available path
    print('No available path. Code stoped!')
else:
    # run all paths
    for path in paths:
        if os.path.isdir(path): # is a folder
            pass
            # get all legal files by extensions
            path_files = list(filter(lambda p: os.path.exists(p) and p.endswith(ext), paths))
            # run all files
            for pathfile in path_files:
                repack_command(pathfile)
        else: # is a file
            repack_command(path)

    print('All repacking finished')