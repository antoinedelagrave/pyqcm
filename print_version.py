import os
import subprocess
git_hash = str(subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']))
version = str(subprocess.check_output(['git', 'describe', '--tags']))
git_hash = git_hash[2:-3]
version = version[2:-3]
version = version.rsplit("-")[0]

fout = open("pyqcm/qcm_git_hash.py", "w")
fout.write("git_hash = '{:s}'\n".format(git_hash))
fout.write("version = '{:s}'\n".format(version))
fout.close() 
