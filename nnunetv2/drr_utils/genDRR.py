import platform
from linux_genDRR import linuxgenDRR
from windows_genDRR import windowsgenDRR


def genDRR(sdr, height, delx, rotation, translation, ctDir, saveIMG):
    if platform.system().lower() == "linux":
        linuxgenDRR(sdr, height, delx, rotation, translation, ctDir, saveIMG)
    else:
        windowsgenDRR(sdr, height, delx, rotation, translation, ctDir, saveIMG)
