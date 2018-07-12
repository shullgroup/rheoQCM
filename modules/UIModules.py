'''
modules for GUI
'''


def open_file(path):
    import os, subprocess
    platform = system_check()
    if platform == "win32": # windows
        os.startfile(path)
    else: # mac and linux
        opener ="open" if platform == "darwin" else "xdg-open" 
        subprocess.call([opener, path]) # this opens a new window on Linux every time

def system_check():
    import sys
    return sys.platform


