import os

SMILE_path = "/home/u5584091/Downloads/opensmile-2.3.0/inst/bin/SMILExtract"
os.system("echo '' > opensmile_log") #hacky way to ensure file existence and clear log
OPENSMILE_log = open("opensmile_log", "w")
