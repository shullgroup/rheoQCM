'''
This code uses h5repack from "HDF5 TOOlS". 
Install it before running this code.'
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
org_prefix = 'orgfile_' # the prefix to be added to rename the original file

if sys.platform == 'win32':
    h5tool_file = 'h5repack.exe'
elif sys.platform == 'linux':
    h5tool_file = 'h5repack'
else:
    print('system not support yet')
    exit(0)

h5tool_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), h5tool_file)
print(h5tool_path)

def gen_cmd(inpath, outpath):

    return [h5tool_path, '-i', inpath, '-o', outpath]

# command to process the repack
def repack_command(inpath):
    # convert to absolute path
    inpath = os.path.abspath(inpath)
    print('File: ' + inpath)
    size0 = os.path.getsize(inpath)
    # get dir, name, extension
    indir = os.path.dirname(inpath)
    infilename = os.path.basename(inpath)
    _, infileext = os.path.splitext(inpath)

    orgfilename = org_prefix + infilename
    orgfilepath = os.path.join(indir, orgfilename)
    # rename input file
    print('rename original file "{}" to "{}"'.format(infilename, orgfilename))
    os.rename(inpath, orgfilepath)

    print('repacking...')
    # use h5repack 
    command = gen_cmd(orgfilepath, inpath)
    p = subprocess.Popen(command)
    (output, err) = p.communicate()
    p_status = p.wait()
    print('command output: ', output)
    print('repacking is done')

    size1 = os.path.getsize(inpath)

    # delete original file
    if os.path.exists(inpath): # new file exists
        print('removing original file ...')
        os.remove(orgfilepath)
        print('done')
        print('Original file size: {}; New file size: {}'.format(size0, size1))
    else:
        print('new file not found!')

    print('\n')


def loop_paths(paths, include_subfolder):
    '''
    paths should be absolute paths
    '''
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        if os.path.isdir(path): # is a folder
            print('repacking folder: {}'.format(path))
            # get all legal files by extensions
            sub_paths = os.listdir(path)
            sub_paths = [os.path.join(os.path.abspath(path), sub_path) for sub_path in sub_paths]
            path_files = list(filter(lambda p: os.path.isfile(p) and p.endswith(ext), sub_paths)) # legal file in path
            path_dirs = list(filter(lambda p: os.path.isdir(p), sub_paths)) # subpaths in path

            # run all files in path
            for pathfile in path_files:
                repack_command(pathfile)

            # run all subpath in path
            if include_subfolder is True:
                for pathdir in path_dirs:
                    loop_paths([pathdir], include_subfolder)
        else: # is a file
            if path.endswith(ext):
                repack_command(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process ptrepack to shrink .h5 file size. NOTE: This code uses h5repack from "HDF5 TOOlS". Install it before running this code.')
    
    parser.add_argument('path', metavar='path', type=str, nargs='*', help='path of a file or a folder (repack all .h5 files in the folder). Multiple paths are available.')
    parser.add_argument('-sf', '--subfolder', action='store_true', default='False', help='Include files in subfolders')
    args = parser.parse_args()
    # parser.print_help() # print the help information when run the code w/ -h or --help. Can be commonted if don't want to show


    print('\n===============\n')
    # check the args
    if args.path: # path is given
        paths = args.path
    else: # no path given and input
        paths = shlex.split(input('Tpye path(s): '))

    # get unique paths
    paths = list(set(paths))
    # check file/folder exist
    paths = list(filter(lambda p: (os.path.exists(p) and p.endswith(ext)) or os.path.isdir(p), paths))

    subfolder = args.subfolder # get subfolder setup

    print('Input unique path(s):\n{}'.format('\n'.join(paths)))

    print('\n')
    if not paths: # no available path
        print('No available path. Code stoped!')
    else:
        # run all paths
        loop_paths(paths, subfolder)
        print('All repacking finished')