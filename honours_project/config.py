import os

SMILE_path = "/home/your/opensmile/path/inst/bin/SMILExtract"
os.system("echo '' > opensmile_log") #hacky way to ensure file existence and clear log
OPENSMILE_log = open("opensmile_log", "w")

raise("Configure Opensmile Path in config.py")