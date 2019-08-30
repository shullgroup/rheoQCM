import os
import sys
import argparse
import shlex
import time
import subprocess
import random


ext = ('.h5',) # legal extensions of hdf5 file

# command to process the repack
def repack_command(inpath):
    infilename, infileext = os.path.splitext(inpath)
    tempfilename = 'tempfile' + str(random.randrange(1e5, 9e5)) + infileext
    print('repacking ' + inpath)
    print('to ' + tempfilename + ' ...')
    command = ["ptrepack", "-o", "--chunkshape=auto", "--propindexes", inpath, tempfilename]
    p = subprocess.Popen(command)
    # delete tempfile
    print('removing original file ...')
    os.remove(inpath)
    print('saving tempfile back to ' + inpath + ' ...')
    os.rename(tempfilename, inpath)

    print('done for ' + inpath)
    print('\n')

parser = argparse.ArgumentParser(description='Process ptrepack to shrink overwritten .h5 file size.')
parser.add_argument('integers', metavar='path', type=str, nargs='?', help='path of a file or a folder (repack all .h5 files in the folder).')
parser.print_help()

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