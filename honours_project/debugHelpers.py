"""
Some helpers for playing audio.
"""
import random
import os

def playAudio(path):
    os.system("mplayer " + path)

def playRandomAudio(path_list):
    ran = random.randint(0, len(path_list))
    playAudio(path_list[ran])
    