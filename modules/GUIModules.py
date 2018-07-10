'''
modules for GUI
'''


def open_file(path):
    import os, sys, subprocess

    if sys.platform == "win32": # windows
        os.startfile(path)
    else: # mac and linux
        opener ="open" if sys.platform == "darwin" else "xdg-open" 
        subprocess.call([opener, path]) # this opens a new window on Linux every time