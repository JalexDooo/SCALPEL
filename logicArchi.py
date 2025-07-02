import warnings
warnings.filterwarnings('ignore')

import os
import csv
# import shap
import glob
import h5py
import time
import random
import numpy as np
import pandas as pd
from termcolor import cprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import auc
from sklearn.manifold import TSNE

from transformers import BertModel, BertTokenizer

from config import cfg
import dataset
import models
import loss_functions
import metrics
from utils import read_csv, read_csv_rpkm, FixRNABert, split_dataset, split_dataset1, GradualWarmupScheduler, Sensitivity, Specificity, Accuracy, AUC_SC, roc_and_prc_from_lfc

'''
{'7-Sep': 0, 'A1CF': 0, 'AAMP': 0, 'AAR2': 0, 'AARD': 0, 'AARS': 0, 'AARS2': 0, 'AARSD1': 0, 'AASDHPPT': 0, 'ABCB7': 0, 'ABCC1': 0, 'ABCE1': 0, 'ABCF1': 0, 'ABCF3': 0, 'ABHD11': 0, 'ABHD16A': 0, 'ABHD17A': 0, 'ABT1': 0, 'AC004381.6': 0, 'ACAD8': 0, 'ACADVL': 0, 'ACIN1': 0, 'ACLY': 0, 'ACO1': 0, 'ACO2': 0, 'ACOT8': 0, 'ACSL3': 0, 'ACTL6A': 0, 'ACTR10': 0, 'ACTR1A': 0, 'ACTR1B': 0, 'ACTR2': 0, 'ACTR3': 0, 'ACTR6': 0, 'ACTR8': 0, 'ADA': 0, 'ADAD1': 0, 'ADAD2': 0, 'ADAM15': 0, 'ADAMTS17': 0, 'ADAMTS19': 0, 'ADAR': 0, 'ADARB1': 0, 'ADARB2': 0, 'ADAT1': 0, 'ADAT2': 0, 'ADAT3': 0, 'ADIPOR2': 0, 'ADNP': 0, 'ADNP2': 0, 'ADPRM': 0, 'ADRM1': 0, 'ADSL': 0, 'ADSS': 0, 'AEBP2': 0, 'AEN': 0, 'AFF1': 0, 'AFF2': 0, 'AFF3': 0, 'AFF4': 0, 'AFG3L2': 0, 'AGAP1': 0, 'AGFG1': 0, 'AGO1': 0, 'AGO2': 0, 'AGO3': 0, 'AGO4': 0, 'AGTRAP': 0, 'AHCY': 0, 'AHCYL1': 0, 'AHDC1': 0, 'AHR': 0, 'AIFM1': 0, 'AIMP1': 0, 'AIMP2': 0, 'AIRE': 0, 'AK2': 0, 'AK4': 0, 'AKAP1': 0, 'AKAP11': 0, 'AKAP8': 0, 'AKAP8L': 0, 'AKIRIN1': 0, 'AKIRIN2': 0, 'AKNA': 0, 'AKR1B15': 0, 'AKR7A2': 0, 'AKT1S1': 0, 'AKTIP': 0, 'ALDH1B1': 0, 'ALDH4A1': 0, 'ALDH5A1': 0, 'ALDH9A1': 0, 'ALDOA': 0, 'ALG1': 0, 'ALG11': 0, 'ALG13': 0, 'ALG14': 0, 'ALG1L': 0, 'ALG2': 0, 'ALG5': 0, 'ALG8': 0, 'ALKBH1': 0, 'ALKBH5': 0, 'ALKBH8': 0, 'ALS2': 0, 'ALX1': 0, 'ALYREF': 0, 'AMD1': 0, 'AMZ2': 0, 'ANAPC1': 0, 'ANAPC10': 0, 'ANAPC11': 0, 'ANAPC13': 0, 'ANAPC15': 0, 'ANAPC2': 0, 'ANAPC4': 0, 'ANAPC5': 0, 'ANG': 0, 'ANGEL1': 0, 'ANGEL2': 0, 'ANKHD1': 0, 'ANKLE2': 0, 'ANKRD10': 0, 'ANKRD11': 0, 'ANKRD17': 0, 'ANKRD46': 0, 'ANKRD49': 0, 'ANKRD63': 0, 'ANKRD65': 0, 'ANKZF1': 0, 'ANLN': 0, 'ANXA11': 0, 'ANXA2': 0, 'ANXA4': 0, 'ANXA6': 0, 'AP1S1': 0, 'AP2M1': 0, 'AP2S1': 0, 'AP3S2': 0, 'APC': 0, 'APEX1': 0, 'API5': 0, 'APLP2': 0, 'APOBEC1': 0, 'APOBEC2': 0, 'APOBEC3F': 0, 'APOBEC3G': 0, 'APOBEC3H': 0, 'APOBEC4': 0, 'APTX': 0, 'AQP7': 0, 'AQR': 0, 'AR': 0, 'ARCN1': 0, 'ARF1': 0, 'ARF4': 0, 'ARF6': 0, 'ARGFX': 0, 'ARGLU1': 0, 'ARHGAP11A': 0, 'ARHGAP18': 0, 'ARHGAP27': 0, 'ARHGEF28': 0, 'ARHGEF5': 0, 'ARHGEF7': 0, 'ARID1A': 0, 'ARID2': 0, 'ARID3A': 0, 'ARID5B': 0, 'ARIH1': 0, 'ARL2': 0, 'ARL4D': 0, 'ARL5B': 0, 'ARL6IP1': 0, 'ARL6IP4': 0, 'ARMC5': 0, 'ARMC7': 0, 'ARNT': 0, 'ARNT2': 0, 'ARNTL': 0, 'ARNTL2': 0, 'ARPC2': 0, 'ARPC3': 0, 'ARPC5L': 0, 'ARPP19': 0, 'ARX': 0, 'ASCC1': 0, 'ASCC3': 0, 'ASCL1': 0, 'ASCL2': 0, 'ASCL3': 0, 'ASCL4': 0, 'ASH1L': 0, 'ASNA1': 0, 'ASNS': 0, 'ASPM': 0, 'ATF1': 0, 'ATF2': 0, 'ATF3': 0, 'ATF4': 0, 'ATF5': 0, 'ATF6': 0, 'ATF7': 0, 'ATIC': 0, 'ATL2': 0, 'ATMIN': 0, 'ATOH1': 0, 'ATOH7': 0, 'ATOH8': 0, 'ATP13A1': 0, 'ATP1A1': 0, 'ATP1B3': 0, 'ATP2A2': 0, 'ATP5A1': 0, 'ATP5B': 0, 'ATP5C1': 0, 'ATP5D': 0, 'ATP5F1': 0, 'ATP5I': 0, 'ATP6AP1': 0, 'ATP6AP2': 0, 'ATP6V0B': 0, 'ATP6V0C': 0, 'ATP6V0D1': 0, 'ATP6V1A': 0, 'ATP6V1B2': 0, 'ATP6V1C1': 0, 'ATP6V1D': 0, 'ATP6V1E1': 0, 'ATP6V1F': 0, 'ATP6V1G1': 0, 'ATP6V1H': 0, 'ATR': 0, 'ATRIP': 0, 'ATRX': 0, 'ATXN1': 0, 'ATXN10': 0, 'ATXN1L': 0, 'ATXN2': 0, 'ATXN2L': 0, 'ATXN7L3': 0, 'AUH': 0, 'AURKA': 0, 'AURKAIP1': 0, 'AURKB': 0, 'AZGP1': 0, 'B3GNTL1': 0, 'B4GALNT1': 0, 'BACH1': 0, 'BACH2': 0, 'BAG3': 0, 'BAHD1': 0, 'BAK1': 0, 'BANF1': 0, 'BANP': 0, 'BAP1': 0, 'BARD1': 0, 'BARHL1': 0, 'BARHL2': 0, 'BARX1': 0, 'BARX2': 0, 'BATF': 0, 'BATF2': 0, 'BATF3': 0, 'BAZ2A': 0, 'BAZ2B': 0, 'BBS2': 0, 'BBS7': 0, 'BBX': 0, 'BCAR1': 0, 'BCAS2': 0, 'BCCIP': 0, 'BCDIN3D': 0, 'BCKDHA': 0, 'BCL11A': 0, 'BCL11B': 0, 'BCL6': 0, 'BCL6B': 0, 'BCL7B': 0, 'BCL9L': 0, 'BCLAF1': 0, 'BCS1L': 0, 'BDP1': 0, 'BET1': 0, 'BICC1': 0, 'BIRC5': 0, 'BIRC6': 0, 'BLM': 0, 'BLMH': 0, 'BLOC1S6': 0, 'BLVRB': 0, 'BMP2': 0, 'BMS1': 0, 'BNC1': 0, 'BNC2': 0, 'BNIP1': 0, 'BOLA1': 0, 'BOLA3': 0, 'BOLL': 0, 'BORA': 0, 'BPNT1': 0, 'BPTF': 0, 'BRAF': 0, 'BRAP': 0, 'BRAT1': 0, 'BRCA1': 0, 'BRCA2': 0, 'BRD2': 0, 'BRD4': 0, 'BRD7': 0, 'BRD8': 0, 'BRD9': 0, 'BRF1': 0, 'BRF2': 0, 'BRI3': 0, 'BRIP1': 0, 'BRIX1': 0, 'BRPF1': 0, 'BSX': 0, 'BTAF1': 0, 'BTBD1': 0, 'BTBD2': 0, 'BTF3': 0, 'BTF3L4': 0, 'BTG2': 0, 'BTG3': 0, 'BUB1': 0, 'BUB1B': 0, 'BUB3': 0, 'BUD13': 0, 'BUD31': 0, 'BYSL': 0, 'BZW1': 0, 'BZW2': 0, 'C10orf2': 0, 'C11orf24': 0, 'C11orf57': 0, 'C11orf68': 0, 'C12orf10': 0, 'C12orf45': 0, 'C12orf65': 0, 'C14orf80': 0, 'C14orf93': 0, 'C15orf41': 0, 'C16orf72': 0, 'C16orf80': 0, 'C16orf86': 0, 'C17orf49': 0, 'C17orf58': 0, 'C17orf80': 0, 'C17orf85': 0, 'C18orf21': 0, 'C19orf25': 0, 'C19orf52': 0, 'C19orf53': 0, 'C1D': 0, 'C1QBP': 0, 'C1QTNF4': 0, 'C1orf109': 0, 'C1orf131': 0, 'C1orf174': 0, 'C1orf50': 0, 'C1orf53': 0, 'C20orf194': 0, 'C20orf27': 0, 'C21orf59': 0, 'C2orf49': 0, 'C3orf17': 0, 'C3orf38': 0, 'C4orf3': 0, 'C7orf26': 0, 'C8orf33': 0, 'C8orf37': 0, 'C8orf59': 0, 'C9orf114': 0, 'C9orf40': 0, 'C9orf78': 0, 'CAB39': 0, 'CACNB3': 0, 'CACTIN': 0, 'CACUL1': 0, 'CACYBP': 0, 'CAD': 0, 'CADM4': 0, 'CALM3': 0, 'CALML4': 0, 'CALR': 0, 'CALR3': 0, 'CAMLG': 0, 'CAMTA1': 0, 'CAND1': 0, 'CANX': 0, 'CAP1': 0, 'CAPN15': 0, 'CAPN2': 0, 'CAPRIN1': 0, 'CAPRIN2': 0, 'CARHSP1': 0, 'CARM1': 0, 'CARS': 0, 'CARS2': 0, 'CASC3': 0, 'CASC5': 0, 'CASP16': 0, 'CASZ1': 0, 'CBFB': 0, 'CBLL1': 0, 'CBR4': 0, 'CBX1': 0, 'CBX2': 0, 'CCAR1': 0, 'CCAR2': 0, 'CCDC101': 0, 'CCDC107': 0, 'CCDC115': 0, 'CCDC12': 0, 'CCDC125': 0, 'CCDC126': 0, 'CCDC130': 0, 'CCDC137': 0, 'CCDC144NL': 0, 'CCDC167': 0, 'CCDC174': 0, 'CCDC51': 0, 'CCDC58': 0, 'CCDC59': 0, 'CCDC64B': 0, 'CCDC71': 0, 'CCDC79': 0, 'CCDC84': 0, 'CCDC85C': 0, 'CCDC86': 0, 'CCDC9': 0, 'CCDC92': 0, 'CCDC93': 0, 'CCDC94': 0, 'CCNA2': 0, 'CCNB1': 0, 'CCNC': 0, 'CCND1': 0, 'CCND3': 0, 'CCNG1': 0, 'CCNH': 0, 'CCNK': 0, 'CCNL1': 0, 'CCNT1': 0, 'CCNT2': 0, 'CCRN4L': 0, 'CCT2': 0, 'CCT3': 0, 'CCT4': 0, 'CCT5': 0, 'CCT6A': 0, 'CCT7': 0, 'CCT8': 0, 'CD2BP2': 0, 'CD320': 0, 'CD36': 0, 'CD3D': 0, 'CD3EAP': 0, 'CD46': 0, 'CD55': 0, 'CD71': 0, 'CD81': 0, 'CDAN1': 0, 'CDC123': 0, 'CDC16': 0, 'CDC20': 0, 'CDC23': 0, 'CDC25A': 0, 'CDC25B': 0, 'CDC26': 0, 'CDC27': 0, 'CDC37': 0, 'CDC40': 0, 'CDC42': 0, 'CDC42SE1': 0, 'CDC45': 0, 'CDC5L': 0, 'CDC6': 0, 'CDC7': 0, 'CDC73': 0, 'CDCA3': 0, 'CDCA5': 0, 'CDCA7': 0, 'CDCA8': 0, 'CDIPT': 0, 'CDK1': 0, 'CDK12': 0, 'CDK2': 0, 'CDK4': 0, 'CDK5RAP1': 0, 'CDK5RAP3': 1, 'CDK6': 1, 'CDK7': 1, 'CDK9': 1, 'CDKN1A': 1, 'CDT1': 1, 'CDX1': 1, 'CDX2': 1, 'CDX4': 1, 'CEBPA': 1, 'CEBPB': 1, 'CEBPE': 1, 'CEBPG': 1, 'CEBPZ': 1, 'CELF1': 1, 'CELF3': 1, 'CELF4': 1, 'CELF5': 1, 'CELSR2': 1, 'CENPA': 1, 'CENPB': 1, 'CENPC': 1, 'CENPE': 1, 'CENPF': 1, 'CENPH': 1, 'CENPI': 1, 'CENPJ': 1, 'CENPK': 1, 'CENPL': 1, 'CENPM': 1, 'CENPN': 1, 'CENPO': 1, 'CENPP': 1, 'CENPQ': 1, 'CENPW': 1, 'CEP152': 1, 'CEP192': 1, 'CEP55': 1, 'CEP57': 1, 'CEP68': 1, 'CEP85': 1, 'CEPT1': 1, 'CERS2': 1, 'CFDP1': 1, 'CFL1': 1, 'CFL2': 1, 'CFLAR': 1, 'CHAF1A': 1, 'CHAF1B': 1, 'CHAMP1': 1, 'CHCHD1': 1, 'CHCHD2': 1, 'CHCHD3': 1, 'CHCHD4': 1, 'CHD2': 1, 'CHD3': 1, 'CHD4': 1, 'CHD6': 1, 'CHD7': 1, 'CHD8': 1, 'CHD9': 1, 'CHEK1': 1, 'CHEK2': 1, 'CHERP': 1, 'CHKA': 1, 'CHML': 1, 'CHMP1A': 1, 'CHMP2A': 1, 'CHMP3': 1, 'CHMP4B': 1, 'CHMP5': 1, 'CHMP6': 1, 'CHMP7': 1, 'CHORDC1': 1, 'CHPF': 1, 'CHRAC1': 1, 'CHTF18': 1, 'CHTF8': 1, 'CHTOP': 1, 'CIAO1': 1, 'CIAPIN1': 1, 'CIC': 1, 'CINP': 1, 'CIRBP': 1, 'CIRH1A': 1, 'CISD2': 1, 'CIT': 1, 'CITED4': 1, 'CIZ1': 1, 'CKAP4': 1, 'CKAP5': 1, 'CKB': 1, 'CKLF': 1, 'CKS2': 1, 'CLASRP': 1, 'CLCC1': 1, 'CLCF1': 1, 'CLDND1': 1, 'CLIC1': 1, 'CLK1': 1, 'CLK2': 1, 'CLK3': 1, 'CLK4': 1, 'CLN3': 1, 'CLNS1A': 1, 'CLOCK': 1, 'CLP1': 1, 'CLSPN': 1, 'CLSTN1': 1, 'CLTA': 1, 'CLTC': 1, 'CMC1': 1, 'CMC2': 1, 'CMIP': 1, 'CMPK1': 1, 'CMSS1': 1, 'CMTR1': 1, 'CMTR2': 1, 'CNBP': 1, 'CNDP2': 1, 'CNIH4': 1, 'CNN2': 1, 'CNOT1': 1, 'CNOT10': 1, 'CNOT11': 1, 'CNOT2': 1, 'CNOT3': 1, 'CNOT4': 1, 'CNOT6': 1, 'CNOT6L': 1, 'CNOT7': 1, 'CNOT8': 1, 'CNP': 1, 'CNPY3': 1, 'CNTNAP3B': 1, 'COA3': 1, 'COA5': 1, 'COA6': 1, 'COASY': 1, 'COG1': 1, 'COG2': 1, 'COG3': 1, 'COG4': 1, 'COG6': 1, 'COG7': 1, 'COG8': 1, 'COIL': 1, 'COMTD1': 1, 'COPA': 1, 'COPB1': 1, 'COPB2': 1, 'COPE': 1, 'COPG1': 1, 'COPS2': 1, 'COPS3': 1, 'COPS4': 1, 'COPS5': 1, 'COPS6': 1, 'COPS8': 1, 'COPZ1': 1, 'COQ2': 1, 'COQ4': 1, 'COQ5': 1, 'COX10': 1, 'COX11': 1, 'COX15': 1, 'COX17': 1, 'COX19': 1, 'COX20': 1, 'COX4I1': 1, 'COX5A': 1, 'COX5B': 1, 'COX6C': 1, 'COX7A2L': 1, 'COX7C': 1, 'CPAMD8': 1, 'CPEB2': 1, 'CPEB3': 1, 'CPEB4': 1, 'CPSF2': 1, 'CPSF3': 1, 'CPSF3L': 1, 'CPSF4': 1, 'CPSF4L': 1, 'CPSF6': 1, 'CPSF7': 1, 'CPXCR1': 1, 'CRCP': 1, 'CRCT1': 1, 'CREB1': 1, 'CREB3': 1, 'CREB3L2': 1, 'CREB3L3': 1, 'CREB3L4': 1, 'CREB5': 1, 'CREBL2': 1, 'CREBZF': 1, 'CREM': 1, 'CRIPT': 1, 'CRKL': 1, 'CRLS1': 1, 'CRNKL1': 1, 'CRTC1': 1, 'CRX': 1, 'CRYL1': 1, 'CRYZ': 1, 'CS': 1, 'CSDC2': 1, 'CSDE1': 1, 'CSE1L': 1, 'CSNK1A1': 1, 'CSNK1G3': 1, 'CSNK2B': 1, 'CSRP1': 1, 'CSRP2BP': 1, 'CSTF1': 1, 'CSTF2': 1, 'CSTF2T': 1, 'CSTF3': 1, 'CT62': 1, 'CTCF': 1, 'CTCFL': 1, 'CTDNEP1': 1, 'CTDP1': 1, 'CTIF': 1, 'CTNNB1': 1, 'CTNNBL1': 1, 'CTPS2': 1, 'CTR9': 1, 'CTSL': 1, 'CTSZ': 1, 'CTU1': 1, 'CTU2': 1, 'CUL1': 1, 'CUL3': 1, 'CUL4A': 1, 'CUL4B': 1, 'CUL5': 1, 'CUX1': 1, 'CUX2': 1, 'CWC15': 1, 'CWC27': 1, 'CWF19L1': 1, 'CWF19L2': 1, 'CXXC1': 1, 'CXXC5': 1, 'CXorf23': 1, 'CYB561A3': 1, 'CYB5B': 1, 'CYB5R1': 1, 'CYB5R4': 1, 'CYBA': 1, 'CYC1': 1, 'CYCS': 1, 'CYP2A13': 1, 'DAB2': 1, 'DACH2': 1, 'DAD1': 1, 'DALRD3': 1, 'DAP3': 1, 'DARS': 1, 'DAXX': 1, 'DAZAP1': 1, 'DAZL': 1, 'DBF4': 1, 'DBP': 1, 'DBR1': 1, 'DBX2': 1, 'DCAF11': 1, 'DCAF13': 1, 'DCAF6': 1, 'DCK': 1, 'DCP1B': 1, 'DCP2': 1, 'DCPS': 1, 'DCTD': 1, 'DCTN1': 1, 'DCTN3': 1, 'DCTN4': 1, 'DCTN5': 1, 'DCTN6': 1, 'DDA1': 1, 'DDB1': 1, 'DDHD2': 1, 'DDI2': 1, 'DDIT3': 1, 'DDN': 1, 'DDOST': 1, 'DDX1': 1, 'DDX10': 1, 'DDX11': 1, 'DDX18': 1, 'DDX19A': 1, 'DDX20': 1, 'DDX21': 1, 'DDX23': 1, 'DDX24': 1, 'DDX25': 1, 'DDX26B': 1, 'DDX28': 1, 'DDX31': 1, 'DDX39A': 1, 'DDX39B': 1, 'DDX3X': 1, 'DDX4': 1, 'DDX41': 1, 'DDX42': 1, 'DDX43': 1, 'DDX46': 1, 'DDX49': 1, 'DDX5': 1, 'DDX50': 1, 'DDX51': 1, 'DDX53': 1, 'DDX54': 1, 'DDX55': 1, 'DDX56': 1, 'DDX59': 1, 'DDX60': 1, 'DDX60L': 1, 'DEAF1': 1, 'DEFB129': 1, 'DENND1A': 1, 'DENR': 1, 'DEPDC1': 1, 'DEPDC1B': 1, 'DEPDC4': 1, 'DEPDC5': 1, 'DEPDC7': 1, 'DERL2': 1, 'DESI1': 1, 'DET1': 1, 'DEXI': 1, 'DGCR14': 1, 'DGCR8': 1, 'DGKH': 1, 'DGUOK': 1, 'DHDDS': 1, 'DHFR': 1, 'DHODH': 1, 'DHPS': 1, 'DHRS7': 1, 'DHX15': 1, 'DHX16': 1, 'DHX29': 1, 'DHX30': 1, 'DHX32': 1, 'DHX33': 1, 'DHX34': 1, 'DHX35': 1, 'DHX36': 1, 'DHX37': 1, 'DHX38': 1, 'DHX40': 1, 'DHX58': 1, 'DHX8': 1, 'DHX9': 1, 'DIAPH3': 1, 'DICER1': 1, 'DIDO1': 1, 'DIEXF': 1, 'DIMT1': 1, 'DIP2B': 1, 'DIS3': 1, 'DIS3L': 1, 'DIS3L2': 1, 'DKC1': 1, 'DLAT': 1, 'DLD': 1, 'DLST': 1, 'DLX1': 1, 'DLX2': 1, 'DLX3': 1, 'DLX4': 1, 'DLX6': 1, 'DMAP1': 1, 'DMBX1': 1, 'DMC1': 1, 'DMRT1': 1, 'DMRT2': 1, 'DMRT3': 1, 'DMRTA1': 1, 'DMRTA2': 1, 'DMRTB1': 1, 'DMRTC2': 1, 'DMTF1': 1, 'DNA2': 1, 'DNAAF2': 1, 'DNAJA1': 1, 'DNAJA3': 1, 'DNAJB6': 1, 'DNAJC1': 1, 'DNAJC11': 1, 'DNAJC17': 1, 'DNAJC19': 1, 'DNAJC2': 1, 'DNAJC8': 1, 'DNAJC9': 1, 'DND1': 1, 'DNLZ': 1, 'DNM1L': 1, 'DNM2': 1, 'DNMT1': 1, 'DNTTIP2': 1, 'DOLK': 1, 'DOLPP1': 1, 'DONSON': 1, 'DOT1L': 1, 'DPAGT1': 1, 'DPF1': 1, 'DPF2': 1, 'DPF3': 1, 'DPH1': 1, 'DPH3': 1, 'DPH5': 1, 'DPH6': 1, 'DPM1': 1, 'DPM2': 1, 'DPM3': 1, 'DPPA3': 1, 'DPRX': 1, 'DPY19L2': 1, 'DPY30': 1, 'DQX1': 1, 'DR1': 1, 'DRAP1': 1, 'DRG1': 1, 'DRG2': 1, 'DROSHA': 1, 'DSCC1': 1, 'DSN1': 1, 'DSP': 1, 'DSTN': 1, 'DTL': 1, 'DTX4': 1, 'DTYMK': 1, 'DUS1L': 1, 'DUS2': 1, 'DUS3L': 1, 'DUS4L': 1, 'DUSP11': 1, 'DUSP12': 1, 'DUT': 1, 'DUXA': 1, 'DVL1': 1, 'DVL2': 1, 'DVL3': 1, 'DXO': 1, 'DYNC1I2': 1, 'DYNLL1': 1, 'DYNLRB1': 1, 'DZIP1': 1, 'DZIP1L': 1, 'DZIP3': 1, 'E2F1': 1, 'E2F2': 1, 'E2F3': 1, 'E2F4': 1, 'E2F5': 1, 'E2F6': 1, 'E2F7': 1, 'E2F8': 1, 'E4F1': 1, 'EAF1': 1, 'EAPP': 1, 'EARS2': 1, 'EBF1': 1, 'EBF3': 1, 'EBNA1BP2': 1, 'EBPL': 1, 'ECD': 1, 'ECHS1': 1, 'ECT2': 1, 'EDC3': 1, 'EDC4': 1, 'EDF1': 1, 'EEA1': 1, 'EED': 1, 'EEF1A1': 1, 'EEF1A2': 1, 'EEF1B2': 1, 'EEF1D': 1, 'EEF1E1': 1, 'EEF2': 1, 'EEF2K': 1, 'EEFSEC': 1, 'EFR3A': 1, 'EFTUD1': 1, 'EFTUD2': 1, 'EGFL7': 1, 'EGFR': 1, 'EGLN1': 1, 'EGR1': 1, 'EGR2': 2, 'EGR3': 2, 'EGR4': 2, 'EHF': 2, 'EHMT2': 2, 'EID1': 2, 'EID2B': 2, 'EIF1': 2, 'EIF1AD': 2, 'EIF1AX': 2, 'EIF1B': 2, 'EIF2A': 2, 'EIF2AK1': 2, 'EIF2AK2': 2, 'EIF2AK4': 2, 'EIF2B1': 2, 'EIF2B2': 2, 'EIF2B3': 2, 'EIF2B4': 2, 'EIF2B5': 2, 'EIF2D': 2, 'EIF2S1': 2, 'EIF2S2': 2, 'EIF2S3': 2, 'EIF2S3L': 2, 'EIF3A': 2, 'EIF3B': 2, 'EIF3D': 2, 'EIF3E': 2, 'EIF3F': 2, 'EIF3G': 2, 'EIF3H': 2, 'EIF3I': 2, 'EIF3J': 2, 'EIF3K': 2, 'EIF3L': 2, 'EIF3M': 2, 'EIF4A2': 2, 'EIF4A3': 2, 'EIF4B': 2, 'EIF4E': 2, 'EIF4E1B': 2, 'EIF4E2': 2, 'EIF4E3': 2, 'EIF4ENIF1': 2, 'EIF4G1': 2, 'EIF4G2': 2, 'EIF4H': 2, 'EIF5': 2, 'EIF5A2': 2, 'EIF5B': 2, 'EIF6': 2, 'ELAC1': 2, 'ELAC2': 2, 'ELAVL1': 2, 'ELAVL2': 2, 'ELAVL3': 2, 'ELAVL4': 2, 'ELF1': 2, 'ELF2': 2, 'ELF3': 2, 'ELF4': 2, 'ELF5': 2, 'ELK1': 2, 'ELK3': 2, 'ELK4': 2, 'ELL': 2, 'ELMO2': 2, 'ELOF1': 2, 'ELOVL5': 2, 'ELP2': 2, 'ELP3': 2, 'ELP4': 2, 'ELP5': 2, 'ELP6': 2, 'EMC1': 2, 'EMC2': 2, 'EMC3': 2, 'EMC4': 2, 'EMC7': 2, 'EMG1': 2, 'EMP3': 2, 'EMX1': 2, 'EMX2': 2, 'EN1': 2, 'EN2': 2, 'ENDOG': 2, 'ENDOU': 2, 'ENDOV': 2, 'ENO1': 2, 'ENOPH1': 2, 'ENOX2': 2, 'ENSA': 2, 'ENTPD4': 2, 'ENY2': 2, 'EOMES': 2, 'EP300': 2, 'EPAS1': 2, 'EPC2': 2, 'EPHB4': 2, 'EPHX1': 2, 'EPOR': 2, 'EPRS': 2, 'ERAL1': 2, 'ERBB2': 2, 'ERCC1': 2, 'ERCC2': 2, 'ERCC3': 2, 'ERCC4': 2, 'ERCC6L': 2, 'ERF': 2, 'ERG': 2, 'ERGIC3': 2, 'ERH': 2, 'ERI1': 2, 'ERI2': 2, 'ERN1': 2, 'ERN2': 2, 'ESCO2': 2, 'ESD': 2, 'ESF1': 2, 'ESPL1': 2, 'ESR1': 2, 'ESR2': 2, 'ESRP1': 2, 'ESRP2': 2, 'ESRRA': 2, 'ESRRB': 2, 'ESRRG': 2, 'ESX1': 2, 'ESYT2': 2, 'ETF1': 2, 'ETFA': 2, 'ETS1': 2, 'ETS2': 2, 'ETV1': 2, 'ETV2': 2, 'ETV3': 2, 'ETV3L': 2, 'ETV4': 2, 'ETV5': 2, 'ETV6': 2, 'ETV7': 2, 'EVL': 2, 'EVX1': 2, 'EVX2': 2, 'EWSR1': 2, 'EXO1': 2, 'EXOC1': 2, 'EXOC2': 2, 'EXOC3': 2, 'EXOC4': 2, 'EXOC5': 2, 'EXOC7': 2, 'EXOC8': 2, 'EXOG': 2, 'EXOSC1': 2, 'EXOSC10': 2, 'EXOSC2': 2, 'EXOSC3': 2, 'EXOSC4': 2, 'EXOSC5': 2, 'EXOSC6': 2, 'EXOSC7': 2, 'EXOSC8': 2, 'EXOSC9': 2, 'EXT1': 2, 'EXTL2': 2, 'EZH1': 2, 'EZH2': 2, 'EZR': 2, 'FADD': 2, 'FAF2': 2, 'FAH': 2, 'FAM103A1': 2, 'FAM104B': 2, 'FAM120B': 2, 'FAM120C': 2, 'FAM133B': 2, 'FAM136A': 2, 'FAM160B1': 2, 'FAM168A': 2, 'FAM168B': 2, 'FAM171B': 2, 'FAM193A': 2, 'FAM207A': 2, 'FAM210A': 2, 'FAM216A': 2, 'FAM32A': 2, 'FAM46A': 2, 'FAM50A': 2, 'FAM89B': 2, 'FAM8A1': 2, 'FAM91A1': 2, 'FAM96B': 2, 'FAM98B': 2, 'FAM98C': 2, 'FANCF': 2, 'FANCI': 2, 'FANCM': 2, 'FAR2': 2, 'FARS2': 2, 'FARSA': 2, 'FARSB': 2, 'FASN': 2, 'FASTK': 2, 'FASTKD1': 2, 'FASTKD2': 2, 'FASTKD3': 2, 'FASTKD5': 2, 'FAU': 2, 'FBL': 2, 'FBLIM1': 2, 'FBLL1': 2, 'FBN1': 2, 'FBXL15': 2, 'FBXL20': 2, 'FBXL6': 2, 'FBXO17': 2, 'FBXO4': 2, 'FBXO41': 2, 'FBXO42': 2, 'FBXO5': 2, 'FBXW11': 2, 'FCF1': 2, 'FDPS': 2, 'FDX1L': 2, 'FDXACB1': 2, 'FDXR': 2, 'FECH': 2, 'FEN1': 2, 'FERD3L': 2, 'FERMT2': 2, 'FEV': 2, 'FEZF1': 2, 'FEZF2': 2, 'FGD1': 2, 'FGFR1': 2, 'FGFR1OP': 2, 'FGFRL1': 2, 'FIGLA': 2, 'FIP1L1': 2, 'FIS1': 2, 'FIZ1': 2, 'FKBP15': 2, 'FKBPL': 2, 'FLI1': 2, 'FLII': 2, 'FMNL2': 2, 'FMR1': 2, 'FNBP4': 2, 'FNDC3A': 2, 'FNIP2': 2, 'FNTA': 2, 'FNTB': 2, 'FOS': 2, 'FOSB': 2, 'FOSL1': 2, 'FOSL2': 2, 'FOXA1': 2, 'FOXA2': 2, 'FOXA3': 2, 'FOXB1': 2, 'FOXB2': 2, 'FOXD2': 2, 'FOXD3': 2, 'FOXE1': 2, 'FOXE3': 2, 'FOXF1': 2, 'FOXG1': 2, 'FOXH1': 2, 'FOXI1': 2, 'FOXI2': 2, 'FOXJ1': 2, 'FOXJ2': 2, 'FOXJ3': 2, 'FOXK1': 2, 'FOXK2': 2, 'FOXL1': 2, 'FOXM1': 2, 'FOXN1': 2, 'FOXN2': 2, 'FOXN3': 2, 'FOXN4': 2, 'FOXO1': 2, 'FOXO3': 2, 'FOXO4': 2, 'FOXO6': 2, 'FOXP1': 2, 'FOXP2': 2, 'FOXP3': 2, 'FOXP4': 2, 'FOXQ1': 2, 'FOXR1': 2, 'FOXR2': 2, 'FOXS1': 2, 'FPGS': 2, 'FRG1': 2, 'FRG1B': 2, 'FTH1': 2, 'FTO': 2, 'FTSJ1': 2, 'FTSJ2': 2, 'FTSJ3': 2, 'FUBP1': 2, 'FUBP3': 2, 'FUCA2': 2, 'FUNDC2': 2, 'FURIN': 2, 'FUS': 2, 'FXN': 2, 'FXR1': 2, 'FXR2': 2, 'FZD1': 2, 'G3BP1': 2, 'G3BP2': 2, 'G6PC3': 2, 'G6PD': 2, 'GABPA': 2, 'GABPB1': 2, 'GADD45GIP1': 2, 'GAK': 2, 'GALNT2': 2, 'GALNT4': 2, 'GALP': 2, 'GAPDH': 2, 'GAPVD1': 2, 'GAR1': 2, 'GARS': 2, 'GART': 2, 'GAS8': 2, 'GATA1': 2, 'GATA2': 2, 'GATA3': 2, 'GATA4': 2, 'GATA5': 2, 'GATA6': 2, 'GATAD1': 2, 'GATAD2A': 2, 'GATAD2B': 2, 'GATC': 2, 'GBF1': 2, 'GBX1': 2, 'GBX2': 2, 'GCFC2': 2, 'GCHFR': 2, 'GCM1': 2, 'GCM2': 2, 'GCN1L1': 2, 'GCSH': 2, 'GEMIN2': 2, 'GEMIN4': 2, 'GEMIN5': 2, 'GEMIN6': 2, 'GEMIN7': 2, 'GEMIN8': 2, 'GET4': 2, 'GFER': 2, 'GFI1': 2, 'GFM1': 2, 'GFM2': 2, 'GFPT1': 2, 'GGPS1': 2, 'GID8': 2, 'GINS1': 2, 'GINS2': 2, 'GINS3': 2, 'GINS4': 2, 'GJA3': 2, 'GLA': 2, 'GLE1': 2, 'GLI1': 2, 'GLI2': 2, 'GLI3': 2, 'GLI4': 2, 'GLIS1': 2, 'GLIS2': 2, 'GLMN': 2, 'GLO1': 2, 'GLRX3': 2, 'GLRX5': 2, 'GLT8D1': 2, 'GLTP': 2, 'GLTSCR2': 2, 'GM2A': 2, 'GMDS': 2, 'GMEB1': 2, 'GMEB2': 2, 'GMNN': 2, 'GMPS': 2, 'GNA12': 2, 'GNB1L': 2, 'GNB2L1': 2, 'GNL1': 2, 'GNL2': 2, 'GNL3': 2, 'GNL3L': 2, 'GNPAT': 2, 'GNPNAT1': 2, 'GOLT1B': 2, 'GON4L': 2, 'GOSR2': 2, 'GOT2': 2, 'GPAT2': 2, 'GPATCH1': 2, 'GPATCH4': 2, 'GPATCH8': 2, 'GPI': 2, 'GPKOW': 2, 'GPN1': 2, 'GPN2': 2, 'GPN3': 2, 'GPR155': 2, 'GPR61': 2, 'GPS1': 2, 'GPX4': 2, 'GRB2': 2, 'GRHL3': 2, 'GRM6': 2, 'GRPEL1': 2, 'GRWD1': 2, 'GSC': 2, 'GSC2': 2, 'GSDMA': 2, 'GSPT1': 2, 'GSPT2': 2, 'GSTK1': 2, 'GSX1': 2, 'GSX2': 2, 'GTF2A1': 2, 'GTF2A2': 2, 'GTF2B': 2, 'GTF2E1': 2, 'GTF2E2': 2, 'GTF2F1': 2, 'GTF2F2': 2, 'GTF2H1': 2, 'GTF2H3': 2, 'GTF2H4': 2, 'GTF3A': 2, 'GTF3C1': 2, 'GTF3C2': 2, 'GTF3C3': 2, 'GTF3C4': 2, 'GTF3C5': 2, 'GTF3C6': 2, 'GTPBP1': 2, 'GTPBP10': 2, 'GTPBP2': 2, 'GTPBP3': 2, 'GTPBP8': 2, 'GUF1': 2, 'GUK1': 2, 'GZF1': 2, 'H1F0': 2, 'H1FOO': 2, 'H1FX': 2, 'H2AFJ': 2, 'H2AFV': 2, 'H2AFZ': 2, 'HABP4': 2, 'HACE1': 2, 'HADHB': 2, 'HAMP': 2, 'HAND1': 2, 'HAND2': 2, 'HAP1': 2, 'HAPLN2': 2, 'HARS': 2, 'HARS2': 2, 'HAUS1': 2, 'HAUS2': 2, 'HAUS3': 2, 'HAUS4': 2, 'HAUS5': 2, 'HAUS6': 2, 'HAUS8': 2, 'HBP1': 2, 'HBS1L': 2, 'HCFC1': 2, 'HDAC1': 2, 'HDAC2': 2, 'HDAC3': 2, 'HDDC2': 2, 'HDGF': 2, 'HDLBP': 2, 'HDX': 2, 'HEATR1': 2, 'HEBP1': 2, 'HECTD1': 2, 'HECTD3': 2, 'HELT': 2, 'HELZ': 2, 'HELZ2': 3, 'HENMT1': 3, 'HERC2': 3, 'HERC4': 3, 'HES1': 3, 'HES2': 3, 'HES3': 3, 'HES4': 3, 'HES5': 3, 'HES6': 3, 'HES7': 3, 'HESX1': 3, 'HEXIM1': 3, 'HEXIM2': 3, 'HEY1': 3, 'HEY2': 3, 'HEYL': 3, 'HGS': 3, 'HHEX': 3, 'HHLA3': 3, 'HIC1': 3, 'HIC2': 3, 'HIF1A': 3, 'HIF3A': 3, 'HIGD2A': 3, 'HINFP': 3, 'HINT1': 3, 'HINT3': 3, 'HIPK2': 3, 'HIRA': 3, 'HIST1H1A': 3, 'HIST1H1B': 3, 'HIST1H1D': 3, 'HIST1H1E': 3, 'HIST1H1T': 3, 'HIST1H2AC': 3, 'HIST1H2AE': 3, 'HIST1H2AG': 3, 'HIST1H2AM': 3, 'HIST1H2BD': 3, 'HIST3H2A': 3, 'HIVEP1': 3, 'HIVEP2': 3, 'HIVEP3': 3, 'HJURP': 3, 'HK1': 3, 'HK2': 3, 'HKR1': 3, 'HLA-DQB1': 3, 'HLA-DQB2': 3, 'HLF': 3, 'HLX': 3, 'HM13': 3, 'HMBOX1': 3, 'HMG20A': 3, 'HMG20B': 3, 'HMGA1': 3, 'HMGA2': 3, 'HMGB1': 3, 'HMGB2': 3, 'HMGB3': 3, 'HMGB4': 3, 'HMGCR': 3, 'HMGCS1': 3, 'HMGN2': 3, 'HMX1': 3, 'HMX2': 3, 'HMX3': 3, 'HNF1A': 3, 'HNF4A': 3, 'HNF4G': 3, 'HNRNPA0': 3, 'HNRNPA1': 3, 'HNRNPA2B1': 3, 'HNRNPA3': 3, 'HNRNPAB': 3, 'HNRNPC': 3, 'HNRNPCL1': 3, 'HNRNPD': 3, 'HNRNPDL': 3, 'HNRNPF': 3, 'HNRNPH1': 3, 'HNRNPH2': 3, 'HNRNPH3': 3, 'HNRNPK': 3, 'HNRNPL': 3, 'HNRNPLL': 3, 'HNRNPM': 3, 'HNRNPR': 3, 'HNRNPU': 3, 'HNRNPUL1': 3, 'HNRNPUL2': 3, 'HOMEZ': 3, 'HOPX': 3, 'HOXA1': 3, 'HOXA10': 3, 'HOXA11': 3, 'HOXA2': 3, 'HOXA3': 3, 'HOXA4': 3, 'HOXA5': 3, 'HOXA6': 3, 'HOXA7': 3, 'HOXA9': 3, 'HOXB1': 3, 'HOXB13': 3, 'HOXB2': 3, 'HOXB3': 3, 'HOXB4': 3, 'HOXB5': 3, 'HOXB6': 3, 'HOXB7': 3, 'HOXB8': 3, 'HOXB9': 3, 'HOXC10': 3, 'HOXC11': 3, 'HOXC12': 3, 'HOXC13': 3, 'HOXC5': 3, 'HOXC6': 3, 'HOXC8': 3, 'HOXC9': 3, 'HOXD1': 3, 'HOXD10': 3, 'HOXD11': 3, 'HOXD12': 3, 'HOXD13': 3, 'HOXD4': 3, 'HOXD8': 3, 'HOXD9': 3, 'HP1BP3': 3, 'HPCAL1': 3, 'HRK': 3, 'HRSP12': 3, 'HS6ST1': 3, 'HSCB': 3, 'HSD17B10': 3, 'HSD17B12': 3, 'HSF1': 3, 'HSF2': 3, 'HSF4': 3, 'HSF5': 3, 'HSP90AB1': 3, 'HSP90B1': 3, 'HSPA14': 3, 'HSPA5': 3, 'HSPA8': 3, 'HSPA9': 3, 'HSPBP1': 3, 'HSPD1': 3, 'HSPE1': 3, 'HSPG2': 3, 'HTATSF1': 3, 'HUS1': 3, 'HUWE1': 3, 'HYAL2': 3, 'Hsp10': 3, 'IARS': 3, 'IARS2': 3, 'IBA57': 3, 'IBTK': 3, 'ICK': 3, 'ICT1': 3, 'ID1': 3, 'ID2': 3, 'ID3': 3, 'ID4': 3, 'IDH2': 3, 'IDH3A': 3, 'IDH3G': 3, 'IDS': 3, 'IER2': 3, 'IFI16': 3, 'IFI27L1': 3, 'IFIT1': 3, 'IFIT1B': 3, 'IFIT2': 3, 'IFIT3': 3, 'IFITM2': 3, 'IFITM3': 3, 'IFRD1': 3, 'IFT43': 3, 'IFT46': 3, 'IGBP1': 3, 'IGF2BP1': 3, 'IGF2BP2': 3, 'IGF2BP3': 3, 'IGF2R': 3, 'IGFBP4': 3, 'IGHM': 3, 'IGHMBP2': 3, 'IGSF3': 3, 'IK': 3, 'IKBKAP': 3, 'IKZF1': 3, 'IKZF2': 3, 'IKZF3': 3, 'IKZF4': 3, 'IKZF5': 3, 'IL10RB': 3, 'IL13RA1': 3, 'ILF2': 3, 'ILF3': 3, 'ILK': 3, 'ILKAP': 3, 'IMMP1L': 3, 'IMMP2L': 3, 'IMMT': 3, 'IMP3': 3, 'IMP4': 3, 'IMPA1': 3, 'IMPDH2': 3, 'INCENP': 3, 'INF2': 3, 'ING2': 3, 'ING3': 3, 'ING4': 3, 'INHBE': 3, 'INO80B': 3, 'INS': 3, 'INSM1': 3, 'INSM2': 3, 'INTS1': 3, 'INTS10': 3, 'INTS12': 3, 'INTS2': 3, 'INTS3': 3, 'INTS4': 3, 'INTS5': 3, 'INTS6': 3, 'INTS7': 3, 'INTS8': 3, 'INTS9': 3, 'IPO11': 3, 'IPO13': 3, 'IPO5': 3, 'IPO7': 3, 'IPO8': 3, 'IPO9': 3, 'IPPK': 3, 'IQSEC1': 3, 'IREB2': 3, 'IRF1': 3, 'IRF2': 3, 'IRF3': 3, 'IRF4': 3, 'IRF5': 3, 'IRF6': 3, 'IRF8': 3, 'IRS2': 3, 'IRX1': 3, 'IRX2': 3, 'IRX3': 3, 'IRX4': 3, 'IRX5': 3, 'IRX6': 3, 'ISCA1': 3, 'ISCA2': 3, 'ISCU': 3, 'ISG20': 3, 'ISG20L2': 3, 'ISL1': 3, 'ISL2': 3, 'IST1': 3, 'ISX': 3, 'ITGA2': 3, 'ITGAV': 3, 'ITGB1BP1': 3, 'ITGB5': 3, 'ITM2C': 3, 'IVNS1ABP': 3, 'JAKMIP1': 3, 'JAZF1': 3, 'JDP2': 3, 'JMJD6': 3, 'JMJD7': 3, 'JRKL': 3, 'JTB': 3, 'JUN': 3, 'JUNB': 3, 'JUND': 3, 'KANSL1': 3, 'KANSL2': 3, 'KANSL3': 3, 'KARS': 3, 'KAT2A': 3, 'KAT5': 3, 'KAT6B': 3, 'KAT7': 3, 'KAT8': 3, 'KATNB1': 3, 'KCMF1': 3, 'KCNA10': 3, 'KCNJ9': 3, 'KCTD10': 3, 'KCTD11': 3, 'KCTD15': 3, 'KCTD18': 3, 'KDELC2': 3, 'KDM2A': 3, 'KDM4B': 3, 'KDM8': 3, 'KEAP1': 3, 'KHDC1': 3, 'KHDRBS1': 3, 'KHDRBS2': 3, 'KHDRBS3': 3, 'KHNYN': 3, 'KHSRP': 3, 'KIAA0020': 3, 'KIAA0100': 3, 'KIAA0430': 3, 'KIAA0930': 3, 'KIAA0947': 3, 'KIAA1143': 3, 'KIAA1191': 3, 'KIAA1429': 3, 'KIAA1524': 3, 'KIAA1683': 3, 'KIAA2018': 3, 'KIF11': 3, 'KIF14': 3, 'KIF18A': 3, 'KIF18B': 3, 'KIF20A': 3, 'KIF23': 3, 'KIF2C': 3, 'KIF4A': 3, 'KIFC1': 3, 'KIFC3': 3, 'KLC2': 3, 'KLF1': 3, 'KLF10': 3, 'KLF11': 3, 'KLF12': 3, 'KLF13': 3, 'KLF15': 3, 'KLF16': 3, 'KLF17': 3, 'KLF2': 3, 'KLF3': 3, 'KLF4': 3, 'KLF5': 3, 'KLF6': 3, 'KLF7': 3, 'KLF8': 3, 'KLF9': 3, 'KLHDC3': 3, 'KLHDC8B': 3, 'KLHL11': 3, 'KLHL15': 3, 'KLHL20': 3, 'KMT2D': 3, 'KNTC1': 3, 'KPNA2': 3, 'KPNB1': 3, 'KRAS': 3, 'KRCC1': 3, 'KRI1': 3, 'KRR1': 3, 'KRT10': 3, 'KRT18': 3, 'KRT8': 3, 'KRTAP5-1': 3, 'KRTAP5-9': 3, 'KTI12': 3, 'L1TD1': 3, 'LAGE3': 3, 'LAMB1': 3, 'LAMTOR1': 3, 'LAMTOR2': 3, 'LAMTOR3': 3, 'LAMTOR4': 3, 'LAPTM4A': 3, 'LARP1': 3, 'LARP1B': 3, 'LARP4': 3, 'LARP4B': 3, 'LARP6': 3, 'LARP7': 3, 'LARS': 3, 'LARS2': 3, 'LAS1L': 3, 'LBX1': 3, 'LBX2': 3, 'LCE2A': 3, 'LCE3A': 3, 'LCE5A': 3, 'LCMT1': 3, 'LCMT2': 3, 'LCN1': 3, 'LCOR': 3, 'LCORL': 3, 'LDB1': 3, 'LDHA': 3, 'LDHB': 3, 'LEF1': 3, 'LEMD2': 3, 'LENG1': 3, 'LETM1': 3, 'LEUTX': 3, 'LGR4': 3, 'LHX2': 3, 'LHX3': 3, 'LHX4': 3, 'LHX5': 3, 'LHX6': 3, 'LHX8': 3, 'LIAS': 3, 'LIG1': 3, 'LIN28A': 3, 'LIN28B': 3, 'LIN52': 3, 'LIN54': 3, 'LIN7C': 3, 'LIN9': 3, 'LLPH': 3, 'LMAN2': 3, 'LMNA': 3, 'LMNB1': 3, 'LMX1A': 3, 'LONP1': 3, 'LPIN1': 3, 'LRPPRC': 3, 'LRR1': 3, 'LRRC37B': 3, 'LRRC47': 3, 'LRRC58': 3, 'LRRC8A': 3, 'LRRFIP1': 3, 'LRRFIP2': 3, 'LSG1': 3, 'LSM1': 3, 'LSM10': 3, 'LSM11': 3, 'LSM12': 3, 'LSM14A': 3, 'LSM14B': 3, 'LSM2': 3, 'LSM3': 3, 'LSM4': 3, 'LSM5': 3, 'LSM6': 3, 'LST1': 3, 'LTA4H': 3, 'LTV1': 3, 'LUC7L': 3, 'LUC7L3': 3, 'LUZP4': 3, 'LYL1': 3, 'LYRM4': 3, 'M6PR': 3, 'MACF1': 3, 'MAD2L1': 3, 'MAD2L1BP': 3, 'MAD2L2': 3, 'MAEA': 3, 'MAEL': 3, 'MAF': 3, 'MAFA': 3, 'MAFB': 3, 'MAFF': 3, 'MAFG': 3, 'MAFK': 3, 'MAGEH1': 3, 'MAGOH': 3, 'MAGOHB': 3, 'MAK16': 3, 'MALL': 3, 'MALSU1': 3, 'MANF': 3, 'MAP4': 3, 'MAPK1': 3, 'MAPK3': 3, 'MAPKAP1': 3, 'MAPRE1': 3, 'MARK2': 3, 'MARS': 3, 'MARS2': 3, 'MARVELD1': 3, 'MASTL': 3, 'MAT2A': 3, 'MAT2B': 3, 'MATR3': 3, 'MAU2': 3, 'MAX': 3, 'MAZ': 3, 'MBD1': 3, 'MBD2': 3, 'MBD3': 3, 'MBD4': 3, 'MBNL1': 3, 'MBNL2': 3, 'MBNL3': 4, 'MBOAT7': 4, 'MBTD1': 4, 'MBTPS1': 4, 'MBTPS2': 4, 'MCCD1': 4, 'MCFD2': 4, 'MCL1': 4, 'MCM10': 4, 'MCM2': 4, 'MCM3': 4, 'MCM3AP': 4, 'MCM4': 4, 'MCM5': 4, 'MCM6': 4, 'MCM7': 4, 'MCMBP': 4, 'MCRS1': 4, 'MCTS1': 4, 'MCU': 4, 'MDM4': 4, 'MEAF6': 4, 'MECP2': 4, 'MECR': 4, 'MED1': 4, 'MED10': 4, 'MED11': 4, 'MED12': 4, 'MED14': 4, 'MED15': 4, 'MED16': 4, 'MED17': 4, 'MED18': 4, 'MED19': 4, 'MED20': 4, 'MED21': 4, 'MED23': 4, 'MED24': 4, 'MED25': 4, 'MED26': 4, 'MED27': 4, 'MED28': 4, 'MED29': 4, 'MED30': 4, 'MED31': 4, 'MED4': 4, 'MED6': 4, 'MED7': 4, 'MED8': 4, 'MED9': 4, 'MEF2A': 4, 'MEF2C': 4, 'MEIS1': 4, 'MEIS3': 4, 'MEMO1': 4, 'MEN1': 4, 'MEOX1': 4, 'MEOX2': 4, 'MEPCE': 4, 'MESDC1': 4, 'MESP1': 4, 'MESP2': 4, 'MET': 4, 'METAP1': 4, 'METAP2': 4, 'METTL1': 4, 'METTL10': 4, 'METTL14': 4, 'METTL16': 4, 'METTL17': 4, 'METTL2A': 4, 'METTL3': 4, 'METTL5': 4, 'MEX3A': 4, 'MEX3B': 4, 'MEX3C': 4, 'MEX3D': 4, 'MFAP1': 4, 'MFN2': 4, 'MFSD12': 4, 'MGA': 4, 'MGAT1': 4, 'MGAT4B': 4, 'MGEA5': 4, 'MGMT': 4, 'MGST2': 4, 'MGST3': 4, 'MIB1': 4, 'MICU2': 4, 'MIER1': 4, 'MIER2': 4, 'MIER3': 4, 'MIF': 4, 'MIF4GD': 4, 'MINK1': 4, 'MIOS': 4, 'MIPEP': 4, 'MIS12': 4, 'MIS18A': 4, 'MIS18BP1': 4, 'MIXL1': 4, 'MKRN1': 4, 'MKRN2': 4, 'MKRN3': 4, 'MKX': 4, 'MLEC': 4, 'MLKL': 4, 'MLLT1': 4, 'MLLT3': 4, 'MLST8': 4, 'MLX': 4, 'MLXIP': 4, 'MLXIPL': 4, 'MMADHC': 4, 'MMGT1': 4, 'MMS19': 4, 'MMS22L': 4, 'MNAT1': 4, 'MNT': 4, 'MNX1': 4, 'MOAP1': 4, 'MOB1B': 4, 'MOB3C': 4, 'MOB4': 4, 'MOCS3': 4, 'MOG': 4, 'MON1A': 4, 'MORC2': 4, 'MORF4L2': 4, 'MOV10': 4, 'MOV10L1': 4, 'MPHOSPH10': 4, 'MPHOSPH6': 4, 'MPHOSPH8': 4, 'MPRIP': 4, 'MRFAP1L1': 4, 'MRGBP': 4, 'MROH6': 4, 'MRP63': 4, 'MRPL1': 4, 'MRPL10': 4, 'MRPL11': 4, 'MRPL12': 4, 'MRPL13': 4, 'MRPL14': 4, 'MRPL15': 4, 'MRPL16': 4, 'MRPL17': 4, 'MRPL18': 4, 'MRPL19': 4, 'MRPL2': 4, 'MRPL20': 4, 'MRPL21': 4, 'MRPL22': 4, 'MRPL23': 4, 'MRPL24': 4, 'MRPL27': 4, 'MRPL28': 4, 'MRPL3': 4, 'MRPL30': 4, 'MRPL32': 4, 'MRPL33': 4, 'MRPL34': 4, 'MRPL35': 4, 'MRPL36': 4, 'MRPL37': 4, 'MRPL38': 4, 'MRPL39': 4, 'MRPL4': 4, 'MRPL40': 4, 'MRPL41': 4, 'MRPL42': 4, 'MRPL43': 4, 'MRPL44': 4, 'MRPL46': 4, 'MRPL47': 4, 'MRPL48': 4, 'MRPL49': 4, 'MRPL51': 4, 'MRPL52': 4, 'MRPL54': 4, 'MRPL55': 4, 'MRPL9': 4, 'MRPS10': 4, 'MRPS11': 4, 'MRPS12': 4, 'MRPS14': 4, 'MRPS15': 4, 'MRPS16': 4, 'MRPS18A': 4, 'MRPS18B': 4, 'MRPS18C': 4, 'MRPS2': 4, 'MRPS22': 4, 'MRPS23': 4, 'MRPS25': 4, 'MRPS26': 4, 'MRPS27': 4, 'MRPS28': 4, 'MRPS30': 4, 'MRPS31': 4, 'MRPS33': 4, 'MRPS34': 4, 'MRPS35': 4, 'MRPS36': 4, 'MRPS5': 4, 'MRPS6': 4, 'MRPS7': 4, 'MRPS9': 4, 'MRRF': 4, 'MRTO4': 4, 'MSANTD4': 4, 'MSC': 4, 'MSGN1': 4, 'MSI1': 4, 'MSI2': 4, 'MSL1': 4, 'MSL3': 4, 'MSRB2': 4, 'MST1': 4, 'MST1R': 4, 'MSTO1': 4, 'MSX1': 4, 'MSX2': 4, 'MT1B': 4, 'MTA1': 4, 'MTA2': 4, 'MTA3': 4, 'MTBP': 4, 'MTERFD2': 4, 'MTF1': 4, 'MTFMT': 4, 'MTFR1L': 4, 'MTG1': 4, 'MTG2': 4, 'MTHFS': 4, 'MTHFSD': 4, 'MTIF2': 4, 'MTIF3': 4, 'MTMR14': 4, 'MTO1': 4, 'MTOR': 4, 'MTPAP': 4, 'MTRF1': 4, 'MTRF1L': 4, 'MUC20': 4, 'MUM1': 4, 'MVD': 4, 'MVK': 4, 'MVP': 4, 'MXD1': 4, 'MXD3': 4, 'MXD4': 4, 'MXI1': 4, 'MXRA7': 4, 'MYB': 4, 'MYBBP1A': 4, 'MYBL1': 4, 'MYBL2': 4, 'MYC': 4, 'MYCN': 4, 'MYEF2': 4, 'MYF5': 4, 'MYF6': 4, 'MYH10': 4, 'MYH9': 4, 'MYNN': 4, 'MYOD1': 4, 'MYOG': 4, 'MYSM1': 4, 'MYT1': 4, 'MYT1L': 4, 'MZF1': 4, 'MZT1': 4, 'N4BP1': 4, 'N6AMT1': 4, 'NAA10': 4, 'NAA15': 4, 'NAA20': 4, 'NAA25': 4, 'NAA30': 4, 'NAA35': 4, 'NAA38': 4, 'NAA50': 4, 'NACA': 4, 'NAE1': 4, 'NAF1': 4, 'NAIP': 4, 'NAMPT': 4, 'NANOG': 4, 'NANOS1': 4, 'NANOS2': 4, 'NANOS3': 4, 'NAPA': 4, 'NAPG': 4, 'NARFL': 4, 'NARS': 4, 'NARS2': 4, 'NASP': 4, 'NAT10': 4, 'NBAS': 4, 'NBPF3': 4, 'NCAPD2': 4, 'NCAPD3': 4, 'NCAPG': 4, 'NCAPG2': 4, 'NCAPH': 4, 'NCAPH2': 4, 'NCBP1': 4, 'NCBP2': 4, 'NCBP2L': 4, 'NCK1': 4, 'NCK2': 4, 'NCKAP1': 4, 'NCL': 4, 'NCOA1': 4, 'NCOA2': 4, 'NCOA3': 4, 'NCOR1': 4, 'NCOR2': 4, 'NDC80': 4, 'NDE1': 4, 'NDFIP2': 4, 'NDNL2': 4, 'NDOR1': 4, 'NDUFA1': 4, 'NDUFA10': 4, 'NDUFA11': 4, 'NDUFA2': 4, 'NDUFA4': 4, 'NDUFA5': 4, 'NDUFA8': 4, 'NDUFAB1': 4, 'NDUFAF1': 4, 'NDUFAF2': 4, 'NDUFAF3': 4, 'NDUFB10': 4, 'NDUFB11': 4, 'NDUFB3': 4, 'NDUFB4': 4, 'NDUFB5': 4, 'NDUFB6': 4, 'NDUFB7': 4, 'NDUFB8': 4, 'NDUFB9': 4, 'NDUFC2': 4, 'NDUFS1': 4, 'NDUFS2': 4, 'NDUFS3': 4, 'NDUFS5': 4, 'NDUFS7': 4, 'NDUFS8': 4, 'NDUFV1': 4, 'NECAP1': 4, 'NEDD1': 4, 'NEDD4L': 4, 'NEIL2': 4, 'NEIL3': 4, 'NELFA': 4, 'NELFB': 4, 'NELFE': 4, 'NENF': 4, 'NEUROD1': 4, 'NEUROD2': 4, 'NEUROD4': 4, 'NEUROD6': 4, 'NEUROG1': 4, 'NEUROG2': 4, 'NEUROG3': 4, 'NF1': 4, 'NF2': 4, 'NFAT5': 4, 'NFATC1': 4, 'NFATC2': 4, 'NFATC2IP': 4, 'NFATC3': 4, 'NFATC4': 4, 'NFE2': 4, 'NFE2L1': 4, 'NFE2L2': 4, 'NFE2L3': 4, 'NFIA': 4, 'NFIB': 4, 'NFIC': 4, 'NFIL3': 4, 'NFIX': 4, 'NFKB1': 4, 'NFKB2': 4, 'NFKBIE': 4, 'NFRKB': 4, 'NFS1': 4, 'NFX1': 4, 'NFXL1': 4, 'NFYA': 4, 'NFYB': 4, 'NFYC': 4, 'NGDN': 4, 'NHLH1': 4, 'NHLH2': 4, 'NHLRC2': 4, 'NHP2': 4, 'NIP7': 4, 'NIPBL': 4, 'NISCH': 4, 'NKAP': 4, 'NKRF': 4, 'NKX2-1': 4, 'NKX2-2': 4, 'NKX2-3': 4, 'NKX2-4': 4, 'NKX2-5': 4, 'NKX2-6': 4, 'NKX2-8': 4, 'NKX3-1': 4, 'NKX3-2': 4, 'NKX6-1': 4, 'NKX6-2': 4, 'NKX6-3': 4, 'NLE1': 4, 'NMD3': 4, 'NME4': 4, 'NME7': 4, 'NMNAT1': 4, 'NMT1': 4, 'NOA1': 4, 'NOB1': 4, 'NOBOX': 4, 'NOC2L': 4, 'NOC3L': 4, 'NOC4L': 4, 'NOL10': 4, 'NOL11': 4, 'NOL12': 4, 'NOL6': 4, 'NOL7': 4, 'NOL8': 4, 'NOL9': 4, 'NOLC1': 4, 'NOM1': 4, 'NONO': 4, 'NOP10': 4, 'NOP14': 4, 'NOP2': 4, 'NOP56': 4, 'NOP58': 4, 'NOP9': 4, 'NOTO': 4, 'NOVA1': 4, 'NOVA2': 4, 'NPAS1': 4, 'NPAS2': 4, 'NPAS3': 4, 'NPAS4': 4, 'NPAT': 4, 'NPC2': 4, 'NPEPPS': 4, 'NPLOC4': 4, 'NPM1': 4, 'NPM2': 4, 'NPM3': 4, 'NPTN': 4, 'NR0B1': 4, 'NR0B2': 4, 'NR1D1': 4, 'NR1D2': 4, 'NR1H2': 4, 'NR1H3': 4, 'NR1H4': 4, 'NR1I2': 4, 'NR1I3': 4, 'NR2C1': 4, 'NR2C2': 4, 'NR2C2AP': 4, 'NR2E1': 4, 'NR2F1': 4, 'NR2F2': 4, 'NR2F6': 4, 'NR3C1': 4, 'NR4A1': 4, 'NR4A2': 4, 'NR4A3': 4, 'NR5A1': 4, 'NR5A2': 4, 'NR6A1': 4, 'NRBP1': 4, 'NRDE2': 4, 'NRF1': 4, 'NRK': 4, 'NRL': 4, 'NSA2': 4, 'NSF': 4, 'NSL1': 4, 'NSMAF': 4, 'NSMCE1': 5, 'NSMCE2': 5, 'NSMCE4A': 5, 'NSRP1': 5, 'NSUN2': 5, 'NSUN3': 5, 'NSUN4': 5, 'NSUN5': 5, 'NSUN6': 5, 'NSUN7': 5, 'NT5C2': 5, 'NT5DC1': 5, 'NUBP1': 5, 'NUBP2': 5, 'NUDC': 5, 'NUDCD3': 5, 'NUDT16': 5, 'NUDT16L1': 5, 'NUDT21': 5, 'NUF2': 5, 'NUFIP1': 5, 'NUFIP2': 5, 'NUMA1': 5, 'NUMBL': 5, 'NUP107': 5, 'NUP133': 5, 'NUP153': 5, 'NUP155': 5, 'NUP160': 5, 'NUP205': 5, 'NUP214': 5, 'NUP35': 5, 'NUP43': 5, 'NUP50': 5, 'NUP54': 5, 'NUP85': 5, 'NUP88': 5, 'NUP93': 5, 'NUP98': 5, 'NUPL1': 5, 'NUPL2': 5, 'NUS1': 5, 'NUTF2': 5, 'NVL': 5, 'NXF1': 5, 'NXF3': 5, 'NXT1': 5, 'NXT2': 5, 'NYNRIN': 5, 'OAS1': 5, 'OAS2': 5, 'OAS3': 5, 'OASL': 5, 'OAZ2': 5, 'OBFC1': 5, 'ODC1': 5, 'OGDH': 5, 'OGT': 5, 'OIP5': 5, 'OLFML3': 5, 'OLIG2': 5, 'OLIG3': 5, 'ONECUT1': 5, 'ONECUT2': 5, 'ONECUT3': 5, 'OPA1': 5, 'OR8I2': 5, 'ORAI2': 5, 'ORAI3': 5, 'ORAOV1': 5, 'ORC1': 5, 'ORC2': 5, 'ORC3': 5, 'ORC5': 5, 'ORC6': 5, 'OSBP': 5, 'OSBPL11': 5, 'OSGEP': 5, 'OSGEPL1': 5, 'OSR1': 5, 'OSR2': 5, 'OSTC': 5, 'OTOP1': 5, 'OTOP3': 5, 'OTP': 5, 'OTUD1': 5, 'OTUD5': 5, 'OTX1': 5, 'OTX2': 5, 'OVOL1': 5, 'OVOL2': 5, 'OXSM': 5, 'P4HA1': 5, 'PA2G4': 5, 'PABPC1': 5, 'PABPC1L': 5, 'PABPC3': 5, 'PABPC4L': 5, 'PABPC5': 5, 'PABPN1': 5, 'PAF1': 5, 'PAFAH1B1': 5, 'PAICS': 5, 'PAIP1': 5, 'PAIP2': 5, 'PAIP2B': 5, 'PAK1IP1': 5, 'PAK2': 5, 'PAN2': 5, 'PAN3': 5, 'PANK2': 5, 'PAPD4': 5, 'PAPD5': 5, 'PAPD7': 5, 'PAPOLA': 5, 'PAPOLB': 5, 'PAPOLG': 5, 'PAPSS1': 5, 'PAQR6': 5, 'PARK7': 5, 'PARL': 5, 'PARN': 5, 'PARP1': 5, 'PARP12': 5, 'PARP4': 5, 'PARS2': 5, 'PARVB': 5, 'PATL1': 5, 'PATZ1': 5, 'PAWR': 5, 'PAX1': 5, 'PAX2': 5, 'PAX3': 5, 'PAX4': 5, 'PAX5': 5, 'PAX6': 5, 'PAX7': 5, 'PAX8': 5, 'PAX9': 5, 'PAXBP1': 5, 'PAXIP1': 5, 'PBK': 5, 'PBRM1': 5, 'PBX1': 5, 'PBX2': 5, 'PBX3': 5, 'PBX4': 5, 'PBXIP1': 5, 'PC': 5, 'PCBP1': 5, 'PCBP2': 5, 'PCBP3': 5, 'PCBP4': 5, 'PCCA': 5, 'PCF11': 5, 'PCGF6': 5, 'PCID2': 5, 'PCNA': 5, 'PCNP': 5, 'PCSK6': 5, 'PCYOX1': 5, 'PCYT1A': 5, 'PDAP1': 5, 'PDCD11': 5, 'PDCD2': 5, 'PDCD4': 5, 'PDCD5': 5, 'PDCD6': 5, 'PDCD6IP': 5, 'PDCD7': 5, 'PDCL': 5, 'PDCL3': 5, 'PDE12': 5, 'PDE4DIP': 5, 'PDHX': 5, 'PDIK1L': 5, 'PDLIM1': 5, 'PDPK1': 5, 'PDRG1': 5, 'PDS5A': 5, 'PDS5B': 5, 'PDSS1': 5, 'PDX1': 5, 'PDXK': 5, 'PDZD8': 5, 'PDZK1': 5, 'PEA15': 5, 'PECR': 5, 'PEG10': 5, 'PELI1': 5, 'PELO': 5, 'PELP1': 5, 'PER1': 5, 'PES1': 5, 'PET112': 5, 'PET117': 5, 'PEX16': 5, 'PFAS': 5, 'PFDN1': 5, 'PFDN2': 5, 'PFDN4': 5, 'PFDN5': 5, 'PFDN6': 5, 'PFKL': 5, 'PFKP': 5, 'PFN1': 5, 'PGAM1': 5, 'PGD': 5, 'PGGT1B': 5, 'PGLS': 5, 'PGM3': 5, 'PGR': 5, 'PGS1': 5, 'PHAX': 5, 'PHB': 5, 'PHB2': 5, 'PHF11': 5, 'PHF12': 5, 'PHF19': 5, 'PHF20': 5, 'PHF5A': 5, 'PHLDA2': 5, 'PHOX2A': 5, 'PHOX2B': 5, 'PHRF1': 5, 'PI4KB': 5, 'PIFO': 5, 'PIGC': 5, 'PIGH': 5, 'PIGS': 5, 'PIGV': 5, 'PIH1D1': 5, 'PIH1D2': 5, 'PIH1D3': 5, 'PIK3C2A': 5, 'PIK3C3': 5, 'PIK3CA': 5, 'PIK3R4': 5, 'PIN4': 5, 'PINK1': 5, 'PINX1': 5, 'PIP5K1A': 5, 'PISD': 5, 'PITHD1': 5, 'PITPNA': 5, 'PITRM1': 5, 'PITX1': 5, 'PITX3': 5, 'PIWIL1': 5, 'PIWIL2': 5, 'PIWIL3': 5, 'PIWIL4': 5, 'PKD1': 5, 'PKHD1': 5, 'PKHD1L1': 5, 'PKM': 5, 'PKMYT1': 5, 'PKN2': 5, 'PKNOX1': 5, 'PKNOX2': 5, 'PLAG1': 5, 'PLAGL1': 5, 'PLAGL2': 5, 'PLBD2': 5, 'PLD6': 5, 'PLEK': 5, 'PLEK2': 5, 'PLEKHA4': 5, 'PLEKHM1': 5, 'PLK1': 5, 'PLK4': 5, 'PLOD1': 5, 'PLOD3': 5, 'PLP2': 5, 'PLRG1': 5, 'PLSCR1': 5, 'PLXNA1': 5, 'PLXNA2': 5, 'PLXNA3': 5, 'PLXNA4': 5, 'PLXNB1': 5, 'PLXNB2': 5, 'PLXNB3': 5, 'PLXNC1': 5, 'PLXND1': 5, 'PMF1': 5, 'PMM1': 5, 'PMM2': 5, 'PMPCA': 5, 'PMPCB': 5, 'PMS1': 5, 'PMS2': 5, 'PMVK': 5, 'PNISR': 5, 'PNKP': 5, 'PNLDC1': 5, 'PNN': 5, 'PNO1': 5, 'PNP': 5, 'PNPT1': 5, 'PNRC2': 5, 'POC1A': 5, 'POGK': 5, 'POGZ': 5, 'POLA1': 5, 'POLA2': 5, 'POLD1': 5, 'POLD2': 5, 'POLD3': 5, 'POLDIP3': 5, 'POLE': 5, 'POLE2': 5, 'POLE3': 5, 'POLE4': 5, 'POLG': 5, 'POLG2': 5, 'POLR1A': 5, 'POLR1B': 5, 'POLR1C': 5, 'POLR1D': 5, 'POLR1E': 5, 'POLR2B': 5, 'POLR2C': 5, 'POLR2D': 5, 'POLR2E': 5, 'POLR2F': 5, 'POLR2G': 5, 'POLR2I': 5, 'POLR2J': 5, 'POLR2K': 5, 'POLR2L': 5, 'POLR3A': 5, 'POLR3B': 5, 'POLR3C': 5, 'POLR3D': 5, 'POLR3E': 5, 'POLR3F': 5, 'POLR3H': 5, 'POLR3K': 5, 'POLRMT': 5, 'POMGNT2': 5, 'POMP': 5, 'POP1': 5, 'POP4': 5, 'POP5': 5, 'POP7': 5, 'POT1': 5, 'POU1F1': 5, 'POU2F1': 5, 'POU2F2': 5, 'POU2F3': 5, 'POU3F1': 5, 'POU3F2': 5, 'POU3F3': 5, 'POU4F1': 5, 'POU4F2': 5, 'POU5F1': 5, 'POU6F2': 5, 'PPA1': 5, 'PPA2': 5, 'PPAN': 5, 'PPARD': 5, 'PPARGC1A': 5, 'PPARGC1B': 5, 'PPAT': 5, 'PPCS': 5, 'PPDPF': 5, 'PPIC': 5, 'PPID': 5, 'PPIE': 5, 'PPIF': 5, 'PPIH': 5, 'PPIL1': 5, 'PPIL2': 5, 'PPIL3': 5, 'PPIL4': 5, 'PPM1B': 5, 'PPM1D': 5, 'PPME1': 5, 'PPP1CA': 5, 'PPP1CB': 5, 'PPP1R10': 5, 'PPP1R11': 5, 'PPP1R12A': 5, 'PPP1R13L': 5, 'PPP1R14B': 5, 'PPP1R15B': 5, 'PPP1R35': 5, 'PPP1R7': 5, 'PPP1R8': 5, 'PPP2CA': 5, 'PPP2R1A': 5, 'PPP2R2A': 5, 'PPP2R3C': 5, 'PPP2R4': 5, 'PPP4C': 5, 'PPP6C': 5, 'PPP6R1': 5, 'PPP6R3': 5, 'PPRC1': 5, 'PPWD1': 5, 'PRAC1': 5, 'PRB3': 5, 'PRB4': 5, 'PRCC': 5, 'PRCP': 5, 'PRDM1': 5, 'PRDM10': 5, 'PRDM12': 5, 'PRDM13': 5, 'PRDM14': 5, 'PRDM15': 5, 'PRDM2': 5, 'PRDM4': 5, 'PRDM5': 5, 'PRDM6': 5, 'PRDM7': 5, 'PRDM8': 5, 'PRDM9': 5, 'PRDX1': 5, 'PRDX2': 5, 'PREB': 5, 'PRELID1': 5, 'PRIM1': 5, 'PRKAR1A': 5, 'PRKDC': 5, 'PRKRA': 5, 'PRKRIP1': 5, 'PRKRIR': 5, 'PRMT1': 5, 'PRMT3': 5, 'PRMT5': 5, 'PROP1': 5, 'PROX1': 5, 'PROX2': 5, 'PRPF18': 5, 'PRPF19': 5, 'PRPF3': 5, 'PRPF31': 5, 'PRPF38A': 5, 'PRPF38B': 5, 'PRPF39': 5, 'PRPF4': 5, 'PRPF40A': 5, 'PRPF40B': 5, 'PRPF4B': 5, 'PRPF6': 5, 'PRPF8': 5, 'PRPS2': 5, 'PRR12': 5, 'PRR13': 5, 'PRR3': 5, 'PRRC2A': 5, 'PRRX1': 5, 'PRRX2': 5, 'PRSS33': 5, 'PSAP': 5, 'PSEN2': 5, 'PSIP1': 5, 'PSMA1': 5, 'PSMA3': 5, 'PSMA4': 5, 'PSMA5': 5, 'PSMA7': 5, 'PSMB1': 5, 'PSMB2': 5, 'PSMB4': 5, 'PSMB5': 5, 'PSMB6': 5, 'PSMB7': 5, 'PSMC1': 5, 'PSMC2': 5, 'PSMC3': 5, 'PSMC4': 5, 'PSMC5': 5, 'PSMC6': 5, 'PSMD1': 5, 'PSMD11': 5, 'PSMD12': 5, 'PSMD13': 5, 'PSMD14': 5, 'PSMD2': 5, 'PSMD3': 5, 'PSMD4': 5, 'PSMD5': 5, 'PSMD6': 5, 'PSMD7': 5, 'PSMD8': 5, 'PSME1': 5, 'PSME2': 5, 'PSME3': 5, 'PSMG1': 5, 'PSMG2': 5, 'PSMG3': 6, 'PSMG4': 6, 'PSPC1': 6, 'PSTK': 6, 'PTAR1': 6, 'PTBP1': 6, 'PTBP2': 6, 'PTBP3': 6, 'PTCD1': 6, 'PTCD2': 6, 'PTCD3': 6, 'PTCHD2': 6, 'PTDSS1': 6, 'PTEN': 6, 'PTF1A': 6, 'PTGES3': 6, 'PTK2': 6, 'PTMA': 6, 'PTMS': 6, 'PTP4A1': 6, 'PTPMT1': 6, 'PTPN11': 6, 'PTPN18': 6, 'PTPN23': 6, 'PTRF': 6, 'PTRH1': 6, 'PTRH2': 6, 'PTRHD1': 6, 'PTTG1': 6, 'PTTG1IP': 6, 'PUF60': 6, 'PUM1': 6, 'PUM2': 6, 'PURA': 6, 'PURB': 6, 'PURG': 6, 'PUS10': 6, 'PUS3': 6, 'PUS7': 6, 'PUS7L': 6, 'PUSL1': 6, 'PVR': 6, 'PWP1': 6, 'PWP2': 6, 'PXN': 6, 'PYGL': 6, 'PYROXD1': 6, 'PYURF': 6, 'QARS': 6, 'QDPR': 6, 'QKI': 6, 'QRICH1': 6, 'QRSL1': 6, 'QSOX2': 6, 'QTRT1': 6, 'QTRTD1': 6, 'R3HCC1': 6, 'R3HCC1L': 6, 'R3HDM1': 6, 'R3HDM2': 6, 'RAB10': 6, 'RAB11B': 6, 'RAB18': 6, 'RAB1A': 6, 'RAB1B': 6, 'RAB22A': 6, 'RAB5B': 6, 'RAB6A': 6, 'RAB7A': 6, 'RABGGTA': 6, 'RABGGTB': 6, 'RABIF': 6, 'RABL6': 6, 'RAC1': 6, 'RAC3': 6, 'RACGAP1': 6, 'RAD1': 6, 'RAD17': 6, 'RAD21': 6, 'RAD51': 6, 'RAD51C': 6, 'RAD51D': 6, 'RAD54L2': 6, 'RAD9A': 6, 'RAE1': 6, 'RAF1': 6, 'RAG1': 6, 'RALB': 6, 'RALY': 6, 'RALYL': 6, 'RAN': 6, 'RANBP1': 6, 'RANBP17': 6, 'RANBP2': 6, 'RANBP6': 6, 'RANGAP1': 6, 'RAP1B': 6, 'RAPGEF2': 6, 'RAPGEF3': 6, 'RAPGEF4': 6, 'RAPGEF5': 6, 'RARA': 6, 'RARB': 6, 'RARG': 6, 'RARS': 6, 'RARS2': 6, 'RAVER2': 6, 'RAX': 6, 'RAX2': 6, 'RB1': 6, 'RBAK': 6, 'RBBP4': 6, 'RBBP5': 6, 'RBBP6': 6, 'RBBP8': 6, 'RBFOX1': 6, 'RBFOX2': 6, 'RBM10': 6, 'RBM11': 6, 'RBM12': 6, 'RBM12B': 6, 'RBM14': 6, 'RBM15': 6, 'RBM17': 6, 'RBM18': 6, 'RBM19': 6, 'RBM20': 6, 'RBM22': 6, 'RBM23': 6, 'RBM24': 6, 'RBM25': 6, 'RBM26': 6, 'RBM27': 6, 'RBM28': 6, 'RBM3': 6, 'RBM33': 6, 'RBM38': 6, 'RBM39': 6, 'RBM41': 6, 'RBM42': 6, 'RBM43': 6, 'RBM44': 6, 'RBM45': 6, 'RBM46': 6, 'RBM47': 6, 'RBM48': 6, 'RBM4B': 6, 'RBM5': 6, 'RBM6': 6, 'RBM7': 6, 'RBM8A': 6, 'RBMS1': 6, 'RBMS2': 6, 'RBMS3': 6, 'RBMX': 6, 'RBMX2': 6, 'RBMXL1': 6, 'RBMXL2': 6, 'RBMXL3': 6, 'RBPJ': 6, 'RBPJL': 6, 'RBPMS': 6, 'RBPMS2': 6, 'RBX1': 6, 'RC3H1': 6, 'RC3H2': 6, 'RCAN1': 6, 'RCC1': 6, 'RCL1': 6, 'RCN2': 6, 'RCOR2': 6, 'RCOR3': 6, 'RECQL': 6, 'REEP4': 6, 'REL': 6, 'RELA': 6, 'RELB': 6, 'RELL2': 6, 'REPIN1': 6, 'RERE': 6, 'REST': 6, 'REXO1': 6, 'REXO2': 6, 'REXO4': 6, 'RFC1': 6, 'RFC2': 6, 'RFC3': 6, 'RFC4': 6, 'RFC5': 6, 'RFK': 6, 'RFNG': 6, 'RFT1': 6, 'RFWD2': 6, 'RFWD3': 6, 'RFX1': 6, 'RFX2': 6, 'RFX3': 6, 'RFX4': 6, 'RFX5': 6, 'RFX6': 6, 'RFXANK': 6, 'RGP1': 6, 'RGS11': 6, 'RGS6': 6, 'RGS7': 6, 'RGS9': 6, 'RHEB': 6, 'RHOA': 6, 'RHOQ': 6, 'RHOXF1': 6, 'RHPN1': 6, 'RIC8A': 6, 'RICTOR': 6, 'RILP': 6, 'RINT1': 6, 'RIOK1': 6, 'RIOK2': 6, 'RIOK3': 6, 'RIPK1': 6, 'RLF': 6, 'RMI1': 6, 'RNASE1': 6, 'RNASE10': 6, 'RNASE12': 6, 'RNASE13': 6, 'RNASE2': 6, 'RNASE3': 6, 'RNASE6': 6, 'RNASE7': 6, 'RNASE8': 6, 'RNASE9': 6, 'RNASEH1': 6, 'RNASEH2A': 6, 'RNASEH2B': 6, 'RNASEH2C': 6, 'RNASEL': 6, 'RNASET2': 6, 'RNF113A': 6, 'RNF113B': 6, 'RNF114': 6, 'RNF125': 6, 'RNF138': 6, 'RNF166': 6, 'RNF167': 6, 'RNF168': 6, 'RNF17': 6, 'RNF19B': 6, 'RNF20': 6, 'RNF220': 6, 'RNF4': 6, 'RNF40': 6, 'RNF8': 6, 'RNFT1': 6, 'RNGTT': 6, 'RNH1': 6, 'RNMT': 6, 'RNMTL1': 6, 'RNPC3': 6, 'RNPS1': 6, 'ROMO1': 6, 'RORA': 6, 'RORB': 6, 'RORC': 6, 'RP9': 6, 'RPA1': 6, 'RPA2': 6, 'RPA3': 6, 'RPA4': 6, 'RPAIN': 6, 'RPAP1': 6, 'RPAP2': 6, 'RPAP3': 6, 'RPE': 6, 'RPF1': 6, 'RPF2': 6, 'RPIA': 6, 'RPL10': 6, 'RPL10A': 6, 'RPL10L': 6, 'RPL11': 6, 'RPL12': 6, 'RPL13': 6, 'RPL13A': 6, 'RPL14': 6, 'RPL15': 6, 'RPL18': 6, 'RPL18A': 6, 'RPL19': 6, 'RPL21': 6, 'RPL22': 6, 'RPL22L1': 6, 'RPL23': 6, 'RPL23A': 6, 'RPL24': 6, 'RPL26L1': 6, 'RPL27': 6, 'RPL27A': 6, 'RPL28': 6, 'RPL29': 6, 'RPL3': 6, 'RPL30': 6, 'RPL31': 6, 'RPL32': 6, 'RPL34': 6, 'RPL35': 6, 'RPL35A': 6, 'RPL36': 6, 'RPL36AL': 6, 'RPL37': 6, 'RPL37A': 6, 'RPL38': 6, 'RPL3L': 6, 'RPL4': 6, 'RPL41': 6, 'RPL5': 6, 'RPL6': 6, 'RPL7': 6, 'RPL7A': 6, 'RPL7L1': 6, 'RPL8': 6, 'RPL9': 6, 'RPLP0': 6, 'RPLP2': 6, 'RPN1': 6, 'RPN2': 6, 'RPP14': 6, 'RPP21': 6, 'RPP25L': 6, 'RPP30': 6, 'RPP38': 6, 'RPP40': 6, 'RPRD1B': 6, 'RPRD2': 6, 'RPS11': 6, 'RPS12': 6, 'RPS13': 6, 'RPS14': 6, 'RPS15': 6, 'RPS15A': 6, 'RPS16': 6, 'RPS18': 6, 'RPS19': 6, 'RPS19BP1': 6, 'RPS2': 6, 'RPS20': 6, 'RPS21': 6, 'RPS23': 6, 'RPS24': 6, 'RPS25': 6, 'RPS26': 6, 'RPS27A': 6, 'RPS27L': 6, 'RPS28': 6, 'RPS29': 6, 'RPS3': 6, 'RPS3A': 6, 'RPS4X': 6, 'RPS5': 6, 'RPS6': 6, 'RPS6KB1': 6, 'RPS7': 6, 'RPS8': 6, 'RPS9': 6, 'RPSA': 6, 'RPTOR': 6, 'RPUSD1': 6, 'RPUSD2': 6, 'RPUSD3': 6, 'RPUSD4': 6, 'RQCD1': 6, 'RRAGA': 6, 'RRAGC': 6, 'RRBP1': 6, 'RREB1': 6, 'RRM1': 6, 'RRM2': 6, 'RRN3': 6, 'RRNAD1': 6, 'RRP1': 6, 'RRP12': 6, 'RRP15': 6, 'RRP1B': 6, 'RRP36': 6, 'RRP7A': 6, 'RRP8': 6, 'RRP9': 6, 'RRS1': 6, 'RSAD1': 6, 'RSF1': 6, 'RSL1D1': 6, 'RSL24D1': 6, 'RSPH3': 6, 'RSPH9': 6, 'RSRC1': 6, 'RSRC2': 6, 'RTCA': 6, 'RTCB': 6, 'RTEL1': 6, 'RTF1': 6, 'RTFDC1': 6, 'RTTN': 6, 'RUNX1': 6, 'RUNX2': 6, 'RUNX3': 6, 'RUVBL1': 6, 'RUVBL2': 6, 'RXRA': 6, 'RXRB': 6, 'RXRG': 6, 'S100A13': 6, 'SACM1L': 6, 'SAE1': 6, 'SAFB': 6, 'SAFB2': 6, 'SALL1': 6, 'SALL3': 6, 'SALL4': 6, 'SAMD4A': 6, 'SAMD4B': 6, 'SAMHD1': 6, 'SAMM50': 6, 'SAP130': 6, 'SAP18': 6, 'SAP30BP': 6, 'SARS': 6, 'SARS2': 6, 'SART1': 6, 'SART3': 6, 'SASS6': 6, 'SATB1': 6, 'SATB2': 6, 'SBDS': 6, 'SCAF1': 6, 'SCAF11': 6, 'SCAF4': 6, 'SCAF8': 6, 'SCAMP3': 6, 'SCAP': 6, 'SCAPER': 6, 'SCARF1': 6, 'SCD': 6, 'SCFD1': 6, 'SCG5': 6, 'SCGB1C1': 6, 'SCMH1': 6, 'SCML4': 6, 'SCNM1': 6, 'SCO1': 6, 'SCO2': 6, 'SCRT2': 6, 'SCYL1': 6, 'SCYL2': 6, 'SDAD1': 6, 'SDC1': 6, 'SDC3': 6, 'SDE2': 6, 'SDF4': 6, 'SDHA': 6, 'SDHAF2': 6, 'SDHB': 6, 'SDHC': 6, 'SDHD': 6, 'SDSL': 6, 'SEC11A': 6, 'SEC13': 6, 'SEC16A': 6, 'SEC61A1': 6, 'SEC61B': 6, 'SEC61G': 6, 'SEC62': 6, 'SEC63': 6, 'SECISBP2': 6, 'SECISBP2L': 6, 'SEH1L': 6, 'SEL1L3': 6, 'SELRC1': 6, 'SEMA4A': 6, 'SEPSECS': 6, 'SERBP1': 6, 'SERGEF': 6, 'SERINC2': 6, 'SERPINB1': 6, 'SERPINB6': 6, 'SET': 6, 'SETD1A': 6, 'SETD2': 6, 'SETD5': 6, 'SETD7': 6, 'SETDB1': 6, 'SETDB2': 6, 'SETX': 6, 'SEZ6L2': 6, 'SF1': 6, 'SF3A1': 7, 'SF3A2': 7, 'SF3B1': 7, 'SF3B14': 7, 'SF3B2': 7, 'SF3B3': 7, 'SF3B4': 7, 'SF3B5': 7, 'SFPQ': 7, 'SFSWAP': 7, 'SFT2D1': 7, 'SGPL1': 7, 'SGPP1': 7, 'SH2B1': 7, 'SH3BGRL': 7, 'SH3PXD2B': 7, 'SHFM1': 7, 'SHOC2': 7, 'SHPRH': 7, 'SHQ1': 7, 'SHROOM4': 7, 'SIAH2': 7, 'SIDT1': 7, 'SIDT2': 7, 'SIGLEC11': 7, 'SIM1': 7, 'SIM2': 7, 'SIN3A': 7, 'SIX2': 7, 'SIX3': 7, 'SIX4': 7, 'SIX5': 7, 'SIX6': 7, 'SKA1': 7, 'SKA2': 7, 'SKA3': 7, 'SKI': 7, 'SKIL': 7, 'SKIV2L': 7, 'SKIV2L2': 7, 'SKP1': 7, 'SKP2': 7, 'SLBP': 7, 'SLC17A5': 7, 'SLC20A1': 7, 'SLC22A4': 7, 'SLC25A10': 7, 'SLC25A26': 7, 'SLC25A28': 7, 'SLC25A3': 7, 'SLC25A45': 7, 'SLC25A5': 7, 'SLC26A6': 7, 'SLC29A1': 7, 'SLC29A3': 7, 'SLC2A1': 7, 'SLC2A4RG': 7, 'SLC30A9': 7, 'SLC31A1': 7, 'SLC35B2': 7, 'SLC39A10': 7, 'SLC39A13': 7, 'SLC39A3': 7, 'SLC39A7': 7, 'SLC3A2': 7, 'SLC41A1': 7, 'SLC45A3': 7, 'SLC4A10': 7, 'SLC4A1AP': 7, 'SLC4A7': 7, 'SLC51B': 7, 'SLC7A1': 7, 'SLC7A5': 7, 'SLC7A6OS': 7, 'SLC9A1': 7, 'SLC9A3R2': 7, 'SLC9B1': 7, 'SLIRP': 7, 'SLMO2': 7, 'SLTM': 7, 'SLU7': 7, 'SMAD1': 7, 'SMAD2': 7, 'SMAD3': 7, 'SMAD4': 7, 'SMAD5': 7, 'SMAD7': 7, 'SMAD9': 7, 'SMARCA1': 7, 'SMARCA5': 7, 'SMARCAL1': 7, 'SMARCB1': 7, 'SMARCC1': 7, 'SMARCC2': 7, 'SMARCD1': 7, 'SMARCE1': 7, 'SMC1A': 7, 'SMC2': 7, 'SMC3': 7, 'SMC5': 7, 'SMC6': 7, 'SMG1': 7, 'SMG5': 7, 'SMG6': 7, 'SMG7': 7, 'SMG8': 7, 'SMG9': 7, 'SMIM12': 7, 'SMIM20': 7, 'SMNDC1': 7, 'SMR3B': 7, 'SMS': 7, 'SMYD2': 7, 'SNAI1': 7, 'SNAI2': 7, 'SNAI3': 7, 'SNAP23': 7, 'SNAPC1': 7, 'SNAPC2': 7, 'SNAPC3': 7, 'SNAPC4': 7, 'SNAPC5': 7, 'SNAPIN': 7, 'SNCB': 7, 'SND1': 7, 'SNF8': 7, 'SNIP1': 7, 'SNRNP200': 7, 'SNRNP25': 7, 'SNRNP27': 7, 'SNRNP35': 7, 'SNRNP40': 7, 'SNRNP48': 7, 'SNRNP70': 7, 'SNRPA': 7, 'SNRPA1': 7, 'SNRPB': 7, 'SNRPB2': 7, 'SNRPC': 7, 'SNRPD1': 7, 'SNRPD2': 7, 'SNRPD3': 7, 'SNRPE': 7, 'SNRPF': 7, 'SNRPG': 7, 'SNUPN': 7, 'SNW1': 7, 'SNX15': 7, 'SNX22': 7, 'SNX3': 7, 'SNX8': 7, 'SOCS3': 7, 'SOD1': 7, 'SOD2': 7, 'SOHLH1': 7, 'SOHLH2': 7, 'SON': 7, 'SORD': 7, 'SORT1': 7, 'SOS1': 7, 'SOWAHC': 7, 'SOX1': 7, 'SOX10': 7, 'SOX11': 7, 'SOX13': 7, 'SOX14': 7, 'SOX15': 7, 'SOX17': 7, 'SOX18': 7, 'SOX2': 7, 'SOX21': 7, 'SOX3': 7, 'SOX30': 7, 'SOX4': 7, 'SOX7': 7, 'SOX8': 7, 'SOX9': 7, 'SP1': 7, 'SP100': 7, 'SP110': 7, 'SP140': 7, 'SP2': 7, 'SP3': 7, 'SP4': 7, 'SP5': 7, 'SP6': 7, 'SP7': 7, 'SP8': 7, 'SPAG5': 7, 'SPAST': 7, 'SPATA2L': 7, 'SPATA33': 7, 'SPATA5': 7, 'SPATA5L1': 7, 'SPATS2': 7, 'SPATS2L': 7, 'SPC24': 7, 'SPC25': 7, 'SPCS2': 7, 'SPCS3': 7, 'SPDEF': 7, 'SPDL1': 7, 'SPDYE1': 7, 'SPEN': 7, 'SPI1': 7, 'SPIB': 7, 'SPIC': 7, 'SPPL2A': 7, 'SPR': 7, 'SPRR2G': 7, 'SPRTN': 7, 'SPRYD4': 7, 'SPTLC1': 7, 'SPTSSA': 7, 'SRA1': 7, 'SRBD1': 7, 'SRCAP': 7, 'SREBF1': 7, 'SREBF2': 7, 'SREK1': 7, 'SRF': 7, 'SRFBP1': 7, 'SRP14': 7, 'SRP19': 7, 'SRP54': 7, 'SRP68': 7, 'SRP72': 7, 'SRP9': 7, 'SRPK2': 7, 'SRPR': 7, 'SRPRB': 7, 'SRRM1': 7, 'SRRM2': 7, 'SRRM4': 7, 'SRSF1': 7, 'SRSF10': 7, 'SRSF11': 7, 'SRSF12': 7, 'SRSF2': 7, 'SRSF3': 7, 'SRSF4': 7, 'SRSF5': 7, 'SRSF6': 7, 'SRSF7': 7, 'SRSF9': 7, 'SS18L2': 7, 'SSB': 7, 'SSBP1': 7, 'SSBP3': 7, 'SSBP4': 7, 'SSH1': 7, 'SSH2': 7, 'SSH3': 7, 'SSR2': 7, 'SSRP1': 7, 'SSSCA1': 7, 'SSU72': 7, 'SSX5': 7, 'ST13': 7, 'ST18': 7, 'ST20': 7, 'ST6GALNAC4': 7, 'STAG1': 7, 'STAG2': 7, 'STAMBP': 7, 'STARD3NL': 7, 'STARD7': 7, 'STAT1': 7, 'STAT2': 7, 'STAT3': 7, 'STAT4': 7, 'STAT5B': 7, 'STAT6': 7, 'STAU1': 7, 'STAU2': 7, 'STIL': 7, 'STK11': 7, 'STK25': 7, 'STK32C': 7, 'STOML1': 7, 'STRAP': 7, 'STRBP': 7, 'STRIP1': 7, 'STRN3': 7, 'STX18': 7, 'STX3': 7, 'STX4': 7, 'STX5': 7, 'STX6': 7, 'STXBP3': 7, 'SUB1': 7, 'SUCLG2': 7, 'SUDS3': 7, 'SUGP1': 7, 'SUGP2': 7, 'SUGT1': 7, 'SUMO1': 7, 'SUPT16H': 7, 'SUPT20H': 7, 'SUPT4H1': 7, 'SUPT5H': 7, 'SUPT6H': 7, 'SUPV3L1': 7, 'SURF1': 7, 'SURF6': 7, 'SUV420H1': 7, 'SUZ12': 7, 'SWT1': 7, 'SYF2': 7, 'SYMPK': 7, 'SYNCRIP': 7, 'SYPL1': 7, 'SYS1': 7, 'SYT15': 7, 'SYVN1': 7, 'T': 7, 'TACC3': 7, 'TACO1': 7, 'TADA1': 7, 'TADA2B': 7, 'TADA3': 7, 'TAF1': 7, 'TAF10': 7, 'TAF12': 7, 'TAF13': 7, 'TAF1A': 7, 'TAF1B': 7, 'TAF1C': 7, 'TAF1D': 7, 'TAF2': 7, 'TAF4': 7, 'TAF5': 7, 'TAF5L': 7, 'TAF6': 7, 'TAF6L': 7, 'TAF7': 7, 'TAF8': 7, 'TAF9': 7, 'TAL1': 7, 'TAL2': 7, 'TAMM41': 7, 'TANGO6': 7, 'TAP1': 7, 'TARBP1': 7, 'TARBP2': 7, 'TARDBP': 7, 'TARS': 7, 'TARS2': 7, 'TARSL2': 7, 'TAX1BP1': 7, 'TBC1D14': 7, 'TBC1D24': 7, 'TBC1D9B': 7, 'TBCA': 7, 'TBCB': 7, 'TBCC': 7, 'TBCD': 7, 'TBCE': 7, 'TBL1XR1': 7, 'TBL3': 7, 'TBP': 7, 'TBR1': 7, 'TBRG4': 7, 'TBX1': 7, 'TBX10': 7, 'TBX15': 7, 'TBX18': 7, 'TBX19': 7, 'TBX2': 7, 'TBX20': 7, 'TBX21': 7, 'TBX22': 7, 'TBX3': 7, 'TBX5': 7, 'TBX6': 7, 'TCEA2': 7, 'TCEAL8': 7, 'TCERG1': 7, 'TCF12': 7, 'TCF15': 7, 'TCF19': 7, 'TCF20': 7, 'TCF21': 7, 'TCF23': 7, 'TCF24': 7, 'TCF25': 7, 'TCF3': 7, 'TCF4': 7, 'TCF7': 7, 'TCF7L1': 7, 'TCF7L2': 7, 'TCFL5': 7, 'TCOF1': 7, 'TCP1': 7, 'TCTA': 7, 'TDGF1': 7, 'TDRD1': 7, 'TDRD10': 7, 'TDRD12': 7, 'TDRD15': 7, 'TDRD3': 7, 'TDRD5': 7, 'TDRD6': 7, 'TDRD7': 7, 'TDRD9': 7, 'TDRKH': 7, 'TEAD2': 7, 'TEAD3': 7, 'TEAD4': 7, 'TEF': 7, 'TEFM': 7, 'TELO2': 7, 'TEP1': 7, 'TERF1': 7, 'TERF2IP': 7, 'TERT': 7, 'TEX10': 7, 'TFAM': 7, 'TFAP2A': 7, 'TFAP2B': 7, 'TFAP2C': 7, 'TFAP2D': 7, 'TFAP2E': 7, 'TFAP4': 7, 'TFB1M': 7, 'TFB2M': 7, 'TFCP2': 7, 'TFCP2L1': 7, 'TFDP1': 7, 'TFDP2': 7, 'TFDP3': 7, 'TFE3': 7, 'TFEB': 7, 'TFEC': 7, 'TFIP11': 7, 'TFRC': 7, 'TGFBRAP1': 7, 'TGIF1': 7, 'TGIF2': 7, 'TGIF2LX': 7, 'TGS1': 7, 'THAP1': 7, 'THAP10': 7, 'THAP11': 7, 'THAP2': 7, 'THAP3': 7, 'THAP4': 7, 'THAP5': 7, 'THAP6': 7, 'THAP7': 7, 'THAP8': 7, 'THAP9': 7, 'THG1L': 7, 'THOC1': 7, 'THOC2': 7, 'THOC3': 7, 'THOC5': 7, 'THOC6': 7, 'THOC7': 7, 'THRA': 7, 'THRAP3': 7, 'THRB': 7, 'THUMPD1': 7, 'THUMPD2': 7, 'THUMPD3': 7, 'TIA1': 7, 'TIAL1': 7, 'TICRR': 7, 'TIGD1': 7, 'TIGD3': 7, 'TIGD4': 7, 'TIGD5': 7, 'TIGD6': 7, 'TIMELESS': 7, 'TIMM10': 7, 'TIMM13': 7, 'TIMM17A': 7, 'TIMM17B': 7, 'TIMM22': 7, 'TIMM44': 7, 'TIMM50': 7, 'TIMM8A': 7, 'TIMM9': 7, 'TIMMDC1': 7, 'TINF2': 7, 'TIPARP': 7, 'TIPIN': 7, 'TIPRL': 7, 'TKT': 7, 'TLCD1': 7, 'TLDC2': 7, 'TLN1': 7, 'TLR3': 7, 'TLR7': 8, 'TLR8': 8, 'TLX1': 8, 'TLX2': 8, 'TLX3': 8, 'TM7SF3': 8, 'TMA16': 8, 'TMED10': 8, 'TMED2': 8, 'TMED4': 8, 'TMEM106B': 8, 'TMEM123': 8, 'TMEM138': 8, 'TMEM14B': 8, 'TMEM160': 8, 'TMEM167A': 8, 'TMEM177': 8, 'TMEM183A': 8, 'TMEM184B': 8, 'TMEM19': 8, 'TMEM199': 8, 'TMEM214': 8, 'TMEM218': 8, 'TMEM222': 8, 'TMEM230': 8, 'TMEM237': 8, 'TMEM246': 8, 'TMEM258': 8, 'TMEM260': 8, 'TMEM39A': 8, 'TMEM41B': 8, 'TMEM50A': 8, 'TMEM59': 8, 'TMX2': 8, 'TNFAIP1': 8, 'TNNT2': 8, 'TNPO1': 8, 'TNPO3': 8, 'TNRC6A': 8, 'TNRC6B': 8, 'TOE1': 8, 'TOM1L1': 8, 'TOMM20': 8, 'TOMM22': 8, 'TOMM34': 8, 'TOMM40': 8, 'TONSL': 8, 'TOP1': 8, 'TOP2A': 8, 'TOP3A': 8, 'TOP3B': 8, 'TOPBP1': 8, 'TOR1AIP2': 8, 'TOR2A': 8, 'TOX': 8, 'TOX2': 8, 'TOX3': 8, 'TP53': 8, 'TP53RK': 8, 'TP63': 8, 'TP73': 8, 'TPGS2': 8, 'TPI1': 8, 'TPM4': 8, 'TPMT': 8, 'TPP2': 8, 'TPR': 8, 'TPRKB': 8, 'TPRX1': 8, 'TPT1': 8, 'TPX2': 8, 'TRA2A': 8, 'TRA2B': 8, 'TRABD': 8, 'TRAF2': 8, 'TRAF3IP1': 8, 'TRAFD1': 8, 'TRAIP': 8, 'TRAM1': 8, 'TRAP1': 8, 'TRAPPC1': 8, 'TRAPPC11': 8, 'TRAPPC12': 8, 'TRAPPC13': 8, 'TRAPPC2L': 8, 'TRAPPC3': 8, 'TRAPPC4': 8, 'TRAPPC8': 8, 'TRDMT1': 8, 'TRERF1': 8, 'TRIAP1': 8, 'TRIM14': 8, 'TRIM21': 8, 'TRIM23': 8, 'TRIM25': 8, 'TRIM26': 8, 'TRIM28': 8, 'TRIM3': 8, 'TRIM32': 8, 'TRIM35': 8, 'TRIM37': 8, 'TRIM56': 8, 'TRIM61': 8, 'TRIM71': 8, 'TRIOBP': 8, 'TRIP11': 8, 'TRIP13': 8, 'TRIT1': 8, 'TRMT1': 8, 'TRMT10A': 8, 'TRMT10B': 8, 'TRMT10C': 8, 'TRMT11': 8, 'TRMT112': 8, 'TRMT12': 8, 'TRMT13': 8, 'TRMT1L': 8, 'TRMT2A': 8, 'TRMT2B': 8, 'TRMT44': 8, 'TRMT5': 8, 'TRMT6': 8, 'TRMT61A': 8, 'TRMT61B': 8, 'TRMU': 8, 'TRNAU1AP': 8, 'TRNT1': 8, 'TROVE2': 8, 'TRPS1': 8, 'TRPT1': 8, 'TRRAP': 8, 'TRUB1': 8, 'TRUB2': 8, 'TSC2': 8, 'TSC22D1': 8, 'TSC22D2': 8, 'TSC22D3': 8, 'TSC22D4': 8, 'TSEN15': 8, 'TSEN2': 8, 'TSEN34': 8, 'TSEN54': 8, 'TSFM': 8, 'TSHZ1': 8, 'TSHZ2': 8, 'TSHZ3': 8, 'TSN': 8, 'TSNAX': 8, 'TSPAN17': 8, 'TSPAN4': 8, 'TSPYL1': 8, 'TSPYL2': 8, 'TSPYL5': 8, 'TSR1': 8, 'TSR2': 8, 'TSR3': 8, 'TST': 8, 'TTC1': 8, 'TTC13': 8, 'TTC17': 8, 'TTC27': 8, 'TTC4': 8, 'TTF1': 8, 'TTF2': 8, 'TTI1': 8, 'TTI2': 8, 'TTK': 8, 'TTPAL': 8, 'TUB': 8, 'TUBB': 8, 'TUBB4B': 8, 'TUBD1': 8, 'TUBE1': 8, 'TUBG1': 8, 'TUBG2': 8, 'TUBGCP2': 8, 'TUBGCP3': 8, 'TUBGCP4': 8, 'TUBGCP6': 8, 'TUFM': 8, 'TUT1': 8, 'TWF1': 8, 'TWIST1': 8, 'TWISTNB': 8, 'TXN': 8, 'TXN2': 8, 'TXNDC11': 8, 'TXNDC9': 8, 'TXNL4A': 8, 'TXNL4B': 8, 'TYW1': 8, 'TYW3': 8, 'TYW5': 8, 'U2AF1': 8, 'U2AF1L4': 8, 'U2AF2': 8, 'U2SURP': 8, 'UAP1': 8, 'UBA1': 8, 'UBA2': 8, 'UBA3': 8, 'UBA5': 8, 'UBA52': 8, 'UBA6': 8, 'UBAC2': 8, 'UBAP1': 8, 'UBAP2': 8, 'UBAP2L': 8, 'UBE2C': 8, 'UBE2D1': 8, 'UBE2D3': 8, 'UBE2G2': 8, 'UBE2H': 8, 'UBE2I': 8, 'UBE2K': 8, 'UBE2L3': 8, 'UBE2M': 8, 'UBE2Q1': 8, 'UBE2S': 8, 'UBE3C': 8, 'UBIAD1': 8, 'UBL4A': 8, 'UBL5': 8, 'UBLCP1': 8, 'UBP1': 8, 'UBQLN2': 8, 'UBQLN4': 8, 'UBR4': 8, 'UBR5': 8, 'UBTF': 8, 'UCHL3': 8, 'UCHL5': 8, 'UCKL1': 8, 'UCP2': 8, 'UFC1': 8, 'UFD1L': 8, 'UFL1': 8, 'UFM1': 8, 'UFSP1': 8, 'UGP2': 8, 'UHMK1': 8, 'UMPS': 8, 'UNC119': 8, 'UNC45A': 8, 'UNC50': 8, 'UNCX': 8, 'UNK': 8, 'UPF1': 8, 'UPF2': 8, 'UPF3A': 8, 'UPF3B': 8, 'UQCRB': 8, 'UQCRC1': 8, 'UQCRC2': 8, 'UQCRFS1': 8, 'UQCRQ': 8, 'URB1': 8, 'URB2': 8, 'URI1': 8, 'URM1': 8, 'UROD': 8, 'USB1': 8, 'USE1': 8, 'USF1': 8, 'USF2': 8, 'USP1': 8, 'USP10': 8, 'USP18': 8, 'USP31': 8, 'USP36': 8, 'USP37': 8, 'USP39': 8, 'USP5': 8, 'USP7': 8, 'USP8': 8, 'USP9X': 8, 'USPL1': 8, 'UTP11L': 8, 'UTP14A': 8, 'UTP14C': 8, 'UTP15': 8, 'UTP18': 8, 'UTP20': 8, 'UTP3': 8, 'UTP6': 8, 'UVRAG': 8, 'UXS1': 8, 'UXT': 8, 'VAMP3': 8, 'VARS': 8, 'VARS2': 8, 'VAX1': 8, 'VAX2': 8, 'VBP1': 8, 'VCL': 8, 'VCP': 8, 'VDAC1': 8, 'VDAC2': 8, 'VDR': 8, 'VENTX': 8, 'VEZF1': 8, 'VHL': 8, 'VMA21': 8, 'VMP1': 8, 'VPRBP': 8, 'VPS13D': 8, 'VPS16': 8, 'VPS18': 8, 'VPS25': 8, 'VPS28': 8, 'VPS29': 8, 'VPS33A': 8, 'VPS33B': 8, 'VPS35': 8, 'VPS36': 8, 'VPS37A': 8, 'VPS41': 8, 'VPS51': 8, 'VPS52': 8, 'VPS53': 8, 'VPS54': 8, 'VPS72': 8, 'VRK1': 8, 'VSX1': 8, 'VSX2': 8, 'VTI1B': 8, 'VWA9': 8, 'WAC': 8, 'WARS': 8, 'WARS2': 8, 'WBP1': 8, 'WBP11': 8, 'WBP2': 8, 'WBP4': 8, 'WBSCR22': 8, 'WDR1': 8, 'WDR11': 8, 'WDR12': 8, 'WDR18': 8, 'WDR24': 8, 'WDR25': 8, 'WDR3': 8, 'WDR33': 8, 'WDR35': 8, 'WDR4': 8, 'WDR43': 8, 'WDR46': 8, 'WDR48': 8, 'WDR55': 8, 'WDR61': 8, 'WDR7': 8, 'WDR70': 8, 'WDR73': 8, 'WDR74': 8, 'WDR75': 8, 'WDR77': 8, 'WDR82': 8, 'WEE1': 8, 'WHSC1': 8, 'WIBG': 8, 'WIZ': 8, 'WNK1': 8, 'WNT8B': 8, 'WRAP53': 8, 'WRB': 8, 'WT1': 8, 'WTAP': 8, 'WWTR1': 8, 'XAB2': 8, 'XBP1': 8, 'XPA': 8, 'XPNPEP1': 8, 'XPO1': 8, 'XPO4': 8, 'XPO5': 8, 'XPO6': 8, 'XPO7': 8, 'XPOT': 8, 'XRCC2': 8, 'XRCC3': 8, 'XRCC5': 8, 'XRCC6': 8, 'XRN1': 8, 'XRN2': 8, 'XYLT2': 8, 'YAE1D1': 8, 'YAP1': 8, 'YARS': 8, 'YARS2': 8, 'YBEY': 8, 'YBX1': 8, 'YBX2': 8, 'YBX3': 8, 'YEATS2': 8, 'YEATS4': 8, 'YIF1B': 8, 'YKT6': 8, 'YME1L1': 8, 'YOD1': 8, 'YPEL1': 8, 'YPEL5': 8, 'YRDC': 8, 'YTHDC1': 8, 'YTHDC2': 8, 'YTHDF1': 8, 'YTHDF2': 8, 'YTHDF3': 8, 'YWHAE': 8, 'YWHAZ': 8, 'YY1': 8, 'YY2': 8, 'ZBED2': 8, 'ZBED3': 8, 'ZBED4': 8, 'ZBTB10': 8, 'ZBTB11': 8, 'ZBTB12': 8, 'ZBTB16': 8, 'ZBTB17': 8, 'ZBTB2': 8, 'ZBTB22': 8, 'ZBTB24': 8, 'ZBTB25': 8, 'ZBTB26': 8, 'ZBTB3': 8, 'ZBTB32': 8, 'ZBTB33': 8, 'ZBTB34': 8, 'ZBTB37': 8, 'ZBTB38': 8, 'ZBTB39': 8, 'ZBTB4': 8, 'ZBTB40': 8, 'ZBTB41': 8, 'ZBTB43': 8, 'ZBTB44': 8, 'ZBTB45': 8, 'ZBTB46': 8, 'ZBTB48': 8, 'ZBTB5': 8, 'ZBTB6': 8, 'ZBTB7A': 8, 'ZBTB7B': 8, 'ZBTB7C': 8, 'ZBTB8OS': 8, 'ZBTB9': 8, 'ZC3H10': 8, 'ZC3H11A': 8, 'ZC3H12A': 8, 'ZC3H12C': 8, 'ZC3H13': 8, 'ZC3H14': 8, 'ZC3H15': 8, 'ZC3H18': 8, 'ZC3H3': 8, 'ZC3H4': 8, 'ZC3H6': 8, 'ZC3H7A': 8, 'ZC3H8': 8, 'ZC3HAV1': 8, 'ZC3HAV1L': 8, 'ZC3HC1': 8, 'ZCCHC11': 8, 'ZCCHC13': 8, 'ZCCHC14': 8, 'ZCCHC17': 8, 'ZCCHC2': 8, 'ZCCHC24': 8, 'ZCCHC4': 8, 'ZCCHC5': 8, 'ZCCHC6': 8, 'ZCCHC7': 8, 'ZCCHC9': 8, 'ZCRB1': 8, 'ZDHHC11': 8, 'ZDHHC19': 8, 'ZDHHC3': 8, 'ZDHHC6': 8, 'ZEB2': 8, 'ZFAT': 8, 'ZFC3H1': 8, 'ZFHX2': 8, 'ZFHX3': 8, 'ZFP1': 8, 'ZFP14': 8, 'ZFP2': 8, 'ZFP28': 8, 'ZFP3': 8, 'ZFP30': 8, 'ZFP36L1': 8, 'ZFP36L2': 8, 'ZFP37': 8, 'ZFP42': 8, 'ZFP57': 9, 'ZFP62': 9, 'ZFP64': 9, 'ZFP69B': 9, 'ZFP82': 9, 'ZFP90': 9, 'ZFP91': 9, 'ZFP92': 9, 'ZFPM1': 9, 'ZFPM2': 9, 'ZFR': 9, 'ZFR2': 9, 'ZFX': 9, 'ZFYVE19': 9, 'ZFYVE20': 9, 'ZFYVE26': 9, 'ZGPAT': 9, 'ZHX1': 9, 'ZHX2': 9, 'ZHX3': 9, 'ZIC1': 9, 'ZIC2': 9, 'ZIC3': 9, 'ZIC4': 9, 'ZIC5': 9, 'ZIK1': 9, 'ZIM2': 9, 'ZIM3': 9, 'ZKSCAN1': 9, 'ZKSCAN2': 9, 'ZKSCAN3': 9, 'ZKSCAN4': 9, 'ZKSCAN5': 9, 'ZMAT1': 9, 'ZMAT2': 9, 'ZMAT3': 9, 'ZMAT4': 9, 'ZMAT5': 9, 'ZMIZ2': 9, 'ZMYND8': 9, 'ZNF100': 9, 'ZNF101': 9, 'ZNF106': 9, 'ZNF107': 9, 'ZNF114': 9, 'ZNF117': 9, 'ZNF12': 9, 'ZNF121': 9, 'ZNF124': 9, 'ZNF131': 9, 'ZNF132': 9, 'ZNF134': 9, 'ZNF135': 9, 'ZNF136': 9, 'ZNF138': 9, 'ZNF14': 9, 'ZNF140': 9, 'ZNF141': 9, 'ZNF142': 9, 'ZNF143': 9, 'ZNF146': 9, 'ZNF148': 9, 'ZNF154': 9, 'ZNF155': 9, 'ZNF157': 9, 'ZNF16': 9, 'ZNF160': 9, 'ZNF165': 9, 'ZNF169': 9, 'ZNF17': 9, 'ZNF174': 9, 'ZNF175': 9, 'ZNF18': 9, 'ZNF180': 9, 'ZNF181': 9, 'ZNF182': 9, 'ZNF184': 9, 'ZNF189': 9, 'ZNF19': 9, 'ZNF195': 9, 'ZNF197': 9, 'ZNF200': 9, 'ZNF202': 9, 'ZNF205': 9, 'ZNF207': 9, 'ZNF208': 9, 'ZNF211': 9, 'ZNF212': 9, 'ZNF213': 9, 'ZNF214': 9, 'ZNF215': 9, 'ZNF217': 9, 'ZNF219': 9, 'ZNF22': 9, 'ZNF221': 9, 'ZNF222': 9, 'ZNF224': 9, 'ZNF225': 9, 'ZNF226': 9, 'ZNF227': 9, 'ZNF23': 9, 'ZNF230': 9, 'ZNF232': 9, 'ZNF233': 9, 'ZNF234': 9, 'ZNF235': 9, 'ZNF236': 9, 'ZNF239': 9, 'ZNF24': 9, 'ZNF248': 9, 'ZNF25': 9, 'ZNF250': 9, 'ZNF251': 9, 'ZNF253': 9, 'ZNF254': 9, 'ZNF256': 9, 'ZNF257': 9, 'ZNF259': 9, 'ZNF26': 9, 'ZNF260': 9, 'ZNF264': 9, 'ZNF266': 9, 'ZNF268': 9, 'ZNF274': 9, 'ZNF276': 9, 'ZNF277': 9, 'ZNF28': 9, 'ZNF280A': 9, 'ZNF280C': 9, 'ZNF280D': 9, 'ZNF281': 9, 'ZNF283': 9, 'ZNF287': 9, 'ZNF292': 9, 'ZNF296': 9, 'ZNF3': 9, 'ZNF30': 9, 'ZNF300': 9, 'ZNF302': 9, 'ZNF304': 9, 'ZNF311': 9, 'ZNF317': 9, 'ZNF318': 9, 'ZNF319': 9, 'ZNF32': 9, 'ZNF320': 9, 'ZNF324': 9, 'ZNF324B': 9, 'ZNF326': 9, 'ZNF329': 9, 'ZNF331': 9, 'ZNF333': 9, 'ZNF334': 9, 'ZNF335': 9, 'ZNF337': 9, 'ZNF33A': 9, 'ZNF33B': 9, 'ZNF34': 9, 'ZNF341': 9, 'ZNF343': 9, 'ZNF345': 9, 'ZNF346': 9, 'ZNF347': 9, 'ZNF35': 9, 'ZNF350': 9, 'ZNF354A': 9, 'ZNF354B': 9, 'ZNF354C': 9, 'ZNF362': 9, 'ZNF365': 9, 'ZNF366': 9, 'ZNF367': 9, 'ZNF37A': 9, 'ZNF382': 9, 'ZNF383': 9, 'ZNF384': 9, 'ZNF385A': 9, 'ZNF385B': 9, 'ZNF385C': 9, 'ZNF385D': 9, 'ZNF391': 9, 'ZNF394': 9, 'ZNF396': 9, 'ZNF404': 9, 'ZNF407': 9, 'ZNF408': 9, 'ZNF41': 9, 'ZNF410': 9, 'ZNF414': 9, 'ZNF415': 9, 'ZNF416': 9, 'ZNF417': 9, 'ZNF418': 9, 'ZNF419': 9, 'ZNF420': 9, 'ZNF423': 9, 'ZNF425': 9, 'ZNF426': 9, 'ZNF428': 9, 'ZNF429': 9, 'ZNF43': 9, 'ZNF430': 9, 'ZNF431': 9, 'ZNF432': 9, 'ZNF433': 9, 'ZNF436': 9, 'ZNF439': 9, 'ZNF44': 9, 'ZNF440': 9, 'ZNF441': 9, 'ZNF442': 9, 'ZNF443': 9, 'ZNF444': 9, 'ZNF445': 9, 'ZNF446': 9, 'ZNF449': 9, 'ZNF45': 9, 'ZNF451': 9, 'ZNF454': 9, 'ZNF460': 9, 'ZNF461': 9, 'ZNF462': 9, 'ZNF467': 9, 'ZNF468': 9, 'ZNF470': 9, 'ZNF471': 9, 'ZNF473': 9, 'ZNF474': 9, 'ZNF479': 9, 'ZNF48': 9, 'ZNF480': 9, 'ZNF483': 9, 'ZNF484': 9, 'ZNF485': 9, 'ZNF486': 9, 'ZNF487': 9, 'ZNF490': 9, 'ZNF491': 9, 'ZNF493': 9, 'ZNF496': 9, 'ZNF497': 9, 'ZNF500': 9, 'ZNF501': 9, 'ZNF502': 9, 'ZNF503': 9, 'ZNF506': 9, 'ZNF507': 9, 'ZNF510': 9, 'ZNF511': 9, 'ZNF512': 9, 'ZNF512B': 9, 'ZNF513': 9, 'ZNF514': 9, 'ZNF516': 9, 'ZNF517': 9, 'ZNF518B': 9, 'ZNF519': 9, 'ZNF521': 9, 'ZNF524': 9, 'ZNF526': 9, 'ZNF527': 9, 'ZNF528': 9, 'ZNF529': 9, 'ZNF530': 9, 'ZNF532': 9, 'ZNF536': 9, 'ZNF540': 9, 'ZNF543': 9, 'ZNF544': 9, 'ZNF546': 9, 'ZNF547': 9, 'ZNF548': 9, 'ZNF549': 9, 'ZNF550': 9, 'ZNF551': 9, 'ZNF552': 9, 'ZNF554': 9, 'ZNF555': 9, 'ZNF556': 9, 'ZNF557': 9, 'ZNF558': 9, 'ZNF560': 9, 'ZNF561': 9, 'ZNF562': 9, 'ZNF563': 9, 'ZNF565': 9, 'ZNF566': 9, 'ZNF567': 9, 'ZNF568': 9, 'ZNF569': 9, 'ZNF57': 9, 'ZNF570': 9, 'ZNF571': 9, 'ZNF572': 9, 'ZNF574': 9, 'ZNF575': 9, 'ZNF576': 9, 'ZNF578': 9, 'ZNF579': 9, 'ZNF580': 9, 'ZNF581': 9, 'ZNF582': 9, 'ZNF583': 9, 'ZNF584': 9, 'ZNF585B': 9, 'ZNF586': 9, 'ZNF587': 9, 'ZNF589': 9, 'ZNF592': 9, 'ZNF593': 9, 'ZNF594': 9, 'ZNF595': 9, 'ZNF596': 9, 'ZNF597': 9, 'ZNF599': 9, 'ZNF600': 9, 'ZNF605': 9, 'ZNF606': 9, 'ZNF607': 9, 'ZNF608': 9, 'ZNF609': 9, 'ZNF610': 9, 'ZNF611': 9, 'ZNF613': 9, 'ZNF615': 9, 'ZNF616': 9, 'ZNF618': 9, 'ZNF619': 9, 'ZNF620': 9, 'ZNF621': 9, 'ZNF622': 9, 'ZNF623': 9, 'ZNF624': 9, 'ZNF626': 9, 'ZNF627': 9, 'ZNF628': 9, 'ZNF629': 9, 'ZNF638': 9, 'ZNF639': 9, 'ZNF641': 9, 'ZNF644': 9, 'ZNF645': 9, 'ZNF646': 9, 'ZNF648': 9, 'ZNF649': 9, 'ZNF652': 9, 'ZNF653': 9, 'ZNF655': 9, 'ZNF660': 9, 'ZNF664': 9, 'ZNF665': 9, 'ZNF667': 9, 'ZNF668': 9, 'ZNF669': 9, 'ZNF670': 9, 'ZNF671': 9, 'ZNF672': 9, 'ZNF674': 9, 'ZNF675': 9, 'ZNF676': 9, 'ZNF677': 9, 'ZNF678': 9, 'ZNF679': 9, 'ZNF680': 9, 'ZNF681': 9, 'ZNF682': 9, 'ZNF683': 9, 'ZNF684': 9, 'ZNF687': 9, 'ZNF689': 9, 'ZNF69': 9, 'ZNF691': 9, 'ZNF692': 9, 'ZNF695': 9, 'ZNF696': 9, 'ZNF697': 9, 'ZNF699': 9, 'ZNF70': 9, 'ZNF700': 9, 'ZNF701': 9, 'ZNF703': 9, 'ZNF704': 9, 'ZNF706': 9, 'ZNF707': 9, 'ZNF708': 9, 'ZNF71': 9, 'ZNF710': 9, 'ZNF711': 9, 'ZNF713': 9, 'ZNF716': 9, 'ZNF717': 9, 'ZNF720': 9, 'ZNF721': 9, 'ZNF738': 9, 'ZNF74': 9, 'ZNF740': 9, 'ZNF746': 9, 'ZNF747': 9, 'ZNF749': 9, 'ZNF750': 9, 'ZNF75A': 9, 'ZNF75D': 9, 'ZNF76': 9, 'ZNF764': 9, 'ZNF765': 9, 'ZNF766': 9, 'ZNF768': 9, 'ZNF77': 9, 'ZNF770': 9, 'ZNF771': 9, 'ZNF772': 9, 'ZNF773': 9, 'ZNF774': 9, 'ZNF775': 9, 'ZNF776': 9, 'ZNF777': 9, 'ZNF778': 9, 'ZNF780A': 9, 'ZNF780B': 9, 'ZNF781': 9, 'ZNF782': 9, 'ZNF783': 9, 'ZNF784': 9, 'ZNF786': 9, 'ZNF789': 9, 'ZNF79': 9, 'ZNF790': 9, 'ZNF791': 9, 'ZNF792': 9, 'ZNF793': 9, 'ZNF80': 9, 'ZNF800': 9, 'ZNF804A': 9, 'ZNF804B': 9, 'ZNF805': 9, 'ZNF808': 9, 'ZNF81': 9, 'ZNF821': 9, 'ZNF823': 9, 'ZNF827': 9, 'ZNF829': 9, 'ZNF83': 9, 'ZNF830': 9, 'ZNF831': 9, 'ZNF835': 9, 'ZNF836': 9, 'ZNF837': 9, 'ZNF839': 9, 'ZNF84': 9, 'ZNF841': 9, 'ZNF843': 9, 'ZNF845': 9, 'ZNF846': 9, 'ZNF85': 9, 'ZNF90': 9, 'ZNF91': 9, 'ZNF92': 9, 'ZNF93': 9, 'ZNFX1': 9, 'ZNHIT1': 9, 'ZNHIT2': 9, 'ZNHIT6': 9, 'ZNRD1': 9, 'ZRANB1': 9, 'ZRANB2': 9, 'ZRSR1': 9, 'ZRSR2': 9, 'ZSCAN1': 9, 'ZSCAN10': 9, 'ZSCAN12': 9, 'ZSCAN16': 9, 'ZSCAN18': 9, 'ZSCAN2': 9, 'ZSCAN20': 9, 'ZSCAN21': 9, 'ZSCAN22': 9, 'ZSCAN23': 9, 'ZSCAN4': 9, 'ZSCAN5A': 9, 'ZSCAN5B': 9, 'ZSCAN5C': 9, 'ZUFSP': 9, 'ZWILCH': 9, 'ZWINT': 9, 'ZXDA': 9, 'ZXDB': 9, 'ZXDC': 9, 'ZZZ3': 9}
'''

def global_init(seed=1024):
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    print('working with pytorch version {}'.format(torch.__version__))
    print('with cuda version {}'.format(torch.version.cuda))
    print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
    print('cudnn version: {}'.format(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
        torch.backends.cudnn.benckmark = True
    else:
        raise Exception('CUDA is not available!!!!')

def get_scheduler(optimizer, conf):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if conf.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch) / float(conf.max_epoch + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif conf.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=conf.lr_decay_iters, gamma=0.5)
    elif conf.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, threshold=1e-5, patience=4, min_lr=5e-6)
    elif conf.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.max_epoch, eta_min=0)
    elif conf.lr_policy == 'WarmUp':
        scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=float(conf.max_epoch), after_scheduler=None)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', conf.lr_policy)
    return scheduler

def train(train_loader, test_loader, cfg, device, fold, params):
    # save path
    ckpt_path = 'ckpt/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    
    loss_path = 'loss/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/'
    if not os.path.exists(loss_path):
        os.mkdir(loss_path)
    
    output_path = 'output/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    train_log_path = 'train_log/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/'
    if not os.path.exists(train_log_path):
        os.mkdir(train_log_path)

    model = getattr(models, cfg.model)(des=cfg.model_des)
    criterion = getattr(loss_functions, cfg.loss_function)()
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = get_scheduler(optimizer, cfg)

    met = metrics.MLMetrics(objective='binary')
    best_auc = 0
    best_epoch = 0
    all_losses = []

    for epoch in range(cfg.max_epoch):
        losses = []
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(sample[:22], device)
            lbl = sample[9].to(device)
            loss = criterion(output, lbl)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        losses = np.array(losses)
        all_losses.append(losses.mean())

        probss = []
        ys = []
        val_loss = []
        # csv
        csv_target_genes, csv_labels, csv_models, csv_gRNA_ids, csv_ms_match_idts, csv_gRNA_seqs, \
        csv_tRNA_seqs, csv_g_folds, csv_icshapes, csv_binding_ps, \
        csv_rpkms, csv_g_mfes, csv_h_mfes, \
        csv_st_relate_lens, csv_ed_relate_lens, \
        csv_utr5_rates, csv_cds_rates, csv_utr3_rates, \
        csv_read_depths, csv_norm_labels = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        csv_predicts = []
        model.eval()
        for batch_idx, sample in enumerate(test_loader):
            output = model(sample[:22], device)
            lbl = sample[9].to(device)

            loss = criterion(output, lbl)
            p_np = output.to(device='cpu').detach().numpy()
            y_np = lbl.to(device='cpu').detach().numpy()

            # y_np = (y_np>0.75)*1.0
            probss.append(p_np)
            ys.append(y_np)
            val_loss.append(float(loss.item()))

            # save csv file
            csv_target_gene = sample[20]
            csv_label, csv_model,csv_gRNA_id, csv_ms_match_idt, csv_gRNA_seq, \
            csv_tRNA_seq, csv_g_fold, csv_icshape, csv_binding_p, \
            csv_rpkm, csv_g_mfe, csv_h_mfe, \
            csv_st_relate_len, csv_ed_relate_len, \
            csv_utr5_rate, csv_cds_rate, csv_utr3_rate, \
            csv_read_depth, csv_norm_label = sample[22:]
            csv_labels.append(csv_label)
            csv_target_genes.append(csv_target_gene)
            csv_models.append(csv_model)
            csv_gRNA_ids.append(csv_gRNA_id)
            csv_ms_match_idts.append(csv_ms_match_idt)
            csv_gRNA_seqs.append(csv_gRNA_seq)
            csv_tRNA_seqs.append(csv_tRNA_seq)
            csv_g_folds.append(csv_g_fold)
            csv_icshapes.append(csv_icshape)
            csv_binding_ps.append(csv_binding_p)
            csv_rpkms.append(csv_rpkm)
            csv_g_mfes.append(csv_g_mfe)
            csv_h_mfes.append(csv_h_mfe)
            csv_st_relate_lens.append(csv_st_relate_len)
            csv_ed_relate_lens.append(csv_ed_relate_len)
            csv_utr5_rates.append(csv_utr5_rate)
            csv_cds_rates.append(csv_cds_rate)
            csv_utr3_rates.append(csv_utr3_rate)
            csv_read_depths.append(csv_read_depth)
            csv_norm_labels.append(csv_norm_label)
            csv_predicts.append(p_np.squeeze())

        # csv
        csv_models = np.concatenate(csv_models)
        csv_gRNA_ids = np.concatenate(csv_gRNA_ids)
        csv_ms_match_idts = np.concatenate(csv_ms_match_idts)
        csv_gRNA_seqs = np.concatenate(csv_gRNA_seqs)
        csv_tRNA_seqs = np.concatenate(csv_tRNA_seqs)
        csv_g_folds = np.concatenate(csv_g_folds)
        csv_icshapes = np.concatenate(csv_icshapes)
        csv_binding_ps = np.concatenate(csv_binding_ps)
        csv_rpkms = np.concatenate(csv_rpkms)
        csv_g_mfes = np.concatenate(csv_g_mfes)
        csv_h_mfes = np.concatenate(csv_h_mfes)
        csv_st_relate_lens = np.concatenate(csv_st_relate_lens)
        csv_ed_relate_lens = np.concatenate(csv_ed_relate_lens)
        csv_utr5_rates = np.concatenate(csv_utr5_rates)
        csv_cds_rates = np.concatenate(csv_cds_rates)
        csv_utr3_rates = np.concatenate(csv_utr3_rates)
        csv_read_depths = np.concatenate(csv_read_depths)
        csv_norm_labels = np.concatenate(csv_norm_labels)
        csv_predicts = np.concatenate(csv_predicts)
        csv_target_genes = np.concatenate(csv_target_genes)
        csv_labels = np.concatenate(csv_labels)

        l_all = np.array(val_loss)
        probss = np.concatenate(probss)
        ys = np.concatenate(ys)
        probss = np.squeeze(probss, axis=1)
        ys = np.squeeze(ys, axis=1)

        # auc, cur = AUC_SC(probss, ys)
        # met = metrics.MLMetrics(objective='binary')
        # met.update(ys, probss, [l_all.mean()])
        lr = optimizer.param_groups[0]['lr']
        
        
        scheduler.step(epoch)
        # best
        
        out_csv = []
        # out_csv.append(['models', 'gRNA_ids', 'ms_match_idts', 'target_gene', 'gRNA_seqs', \
        #                 'tRNA_seqs', 'g_folds', 'icshapes', 'binding_ps', \
        #                 'rpkms', 'g_mfes', 'h_mfes', \
        #                 'st_relate_lens', 'ed_relate_lens', \
        #                 'utr5_rates', 'cds_rates', 'utr3_rates', \
        #                 'read_depths', 'norm_labels', 'norm_predicts', 'label', 'predicts'])

        for i in range(len(csv_predicts)):
            out_csv.append([csv_models[i], csv_gRNA_ids[i], csv_ms_match_idts[i], csv_target_genes[i], csv_gRNA_seqs[i], \
                            csv_tRNA_seqs[i], csv_g_folds[i], csv_icshapes[i], csv_binding_ps[i], \
                            csv_rpkms[i], csv_g_mfes[i], csv_h_mfes[i], \
                            csv_st_relate_lens[i], csv_ed_relate_lens[i], \
                            csv_utr5_rates[i], csv_cds_rates[i], csv_utr3_rates[i], \
                            csv_read_depths[i], csv_norm_labels[i], csv_predicts[i], csv_labels[i], i])
        out_data = pd.DataFrame(out_csv, columns=['models', 'gRNA_ids', 'ms_match_idts',            'target_gene', 'gRNA_seqs', \
                        'tRNA_seqs', 'g_folds', 'icshapes', 'binding_ps', \
                        'rpkms', 'g_mfes', 'h_mfes', \
                        'st_relate_lens', 'ed_relate_lens', \
                        'utr5_rates', 'cds_rates', 'utr3_rates', \
                        'read_depths', 'norm_labels', 'norm_predicts', 'label', 'predicts'])
        for gene in out_data['target_gene'].unique():
            col = 'norm_predicts'
            tar = 'predicts'
            denormalized_lfc = out_data.loc[out_data['target_gene'] == gene, col].to_numpy()
            denormalized_lfc = denormalized_lfc * params.loc[gene, 'scale'] + params.loc[gene, 'location']
            out_data.loc[out_data['target_gene'] == gene, tar] = denormalized_lfc
        
        lb = out_data['label'].to_numpy()
        pret = out_data['predicts'].to_numpy()
        # print('lb + pret: ', lb.shape, pret.shape)
        lb = (lb <= -0.5)*1.0

        (fpr, tpr), (precision, recall) = roc_and_prc_from_lfc(lb, pret)
        auroc = auc(fpr, tpr)
        auprc = auc(recall, precision)

        # print('auc: ', auroc, auprc)


        color_best = 'green'
        if auroc > best_auc:
            best_auc = auroc
            best_epoch = epoch
            color_best = 'red'
            
            torch.save(model.state_dict(), '{}/Fold:{}.pth'.format(ckpt_path, fold))
            out_data.to_csv('{}/Fold:{}.csv'.format(output_path, fold), sep=',', index=False, header=True)

            # with open('{}/Fold:{}.csv'.format(output_path, fold), 'w') as res:
            #     writer = csv.writer(res)
            #     writer.writerows(out_csv)
                
            line = '{} \t Test Epoch: {}     avg.loss: {:.4f} , AUROC: {:4f}, AUPRC: {:.4f}'.format(cfg.model, epoch, l_all.mean(), auroc, auprc)

            cprint(line, color=color_best, attrs=['bold'])
            with open('{}/Fold:{}.txt'.format(train_log_path, fold), 'a') as file:
                print(line, file=file)

    # save loss
    all_losses = np.array(all_losses)
    loss_pathh = loss_path + str(fold) + '.npy'
    np.save(loss_pathh, all_losses)
    torch.cuda.empty_cache()
    model = None
    optimizer = None
    scheduler = None
    criterion = None

def test(test_loader, cfg, device, ckpt_options, params):
    output_path = 'predicts/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '|' + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model = getattr(models, cfg.model)(des=cfg.model_des)
    criterion = getattr(loss_functions, cfg.loss_function)()
    model = model.to(device)

    model.load_state_dict(torch.load(ckpt_options))

    probss = []
    ys = []
    val_loss = []
    # csv
    csv_target_genes, csv_labels = [], []
    csv_models, csv_gRNA_ids, csv_ms_match_idts, csv_gRNA_seqs, \
    csv_tRNA_seqs, csv_g_folds, csv_icshapes, csv_binding_ps, \
    csv_rpkms, csv_g_mfes, csv_h_mfes, \
    csv_st_relate_lens, csv_ed_relate_lens, \
    csv_utr5_rates, csv_cds_rates, csv_utr3_rates, \
    csv_read_depths, csv_norm_labels = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    csv_predicts = []

    model.eval()
    for batch_idx, sample in enumerate(test_loader):
        output = model(sample[:22], device)
        lbl = sample[9].to(device)

        p_np = output.to(device='cpu').detach().numpy()

        # save csv file
        csv_target_gene = sample[20]
        csv_label, csv_model, csv_gRNA_id, csv_ms_match_idt, csv_gRNA_seq, \
        csv_tRNA_seq, csv_g_fold, csv_icshape, csv_binding_p, \
        csv_rpkm, csv_g_mfe, csv_h_mfe, \
        csv_st_relate_len, csv_ed_relate_len, \
        csv_utr5_rate, csv_cds_rate, csv_utr3_rate, \
        csv_read_depth, csv_norm_label = sample[22:]
        csv_labels.append(csv_label)
        csv_target_genes.append(csv_target_gene)
        csv_models.append(csv_model)
        csv_gRNA_ids.append(csv_gRNA_id)
        csv_ms_match_idts.append(csv_ms_match_idt)
        csv_gRNA_seqs.append(csv_gRNA_seq)
        csv_tRNA_seqs.append(csv_tRNA_seq)
        csv_g_folds.append(csv_g_fold)
        csv_icshapes.append(csv_icshape)
        csv_binding_ps.append(csv_binding_p)
        csv_rpkms.append(csv_rpkm)
        csv_g_mfes.append(csv_g_mfe)
        csv_h_mfes.append(csv_h_mfe)
        csv_st_relate_lens.append(csv_st_relate_len)
        csv_ed_relate_lens.append(csv_ed_relate_len)
        csv_utr5_rates.append(csv_utr5_rate)
        csv_cds_rates.append(csv_cds_rate)
        csv_utr3_rates.append(csv_utr3_rate)
        csv_read_depths.append(csv_read_depth)
        csv_norm_labels.append(csv_norm_label)
        csv_predicts.append(p_np.squeeze() if p_np.shape[0] != 1 else p_np[0])

    # csv
    csv_models = np.concatenate(csv_models)
    csv_gRNA_ids = np.concatenate(csv_gRNA_ids)
    csv_ms_match_idts = np.concatenate(csv_ms_match_idts)
    csv_gRNA_seqs = np.concatenate(csv_gRNA_seqs)
    csv_tRNA_seqs = np.concatenate(csv_tRNA_seqs)
    csv_g_folds = np.concatenate(csv_g_folds)
    csv_icshapes = np.concatenate(csv_icshapes)
    csv_binding_ps = np.concatenate(csv_binding_ps)
    csv_rpkms = np.concatenate(csv_rpkms)
    csv_g_mfes = np.concatenate(csv_g_mfes)
    csv_h_mfes = np.concatenate(csv_h_mfes)
    csv_st_relate_lens = np.concatenate(csv_st_relate_lens)
    csv_ed_relate_lens = np.concatenate(csv_ed_relate_lens)
    csv_utr5_rates = np.concatenate(csv_utr5_rates)
    csv_cds_rates = np.concatenate(csv_cds_rates)
    csv_utr3_rates = np.concatenate(csv_utr3_rates)
    csv_read_depths = np.concatenate(csv_read_depths)
    csv_norm_labels = np.concatenate(csv_norm_labels)
    csv_predicts = np.concatenate(csv_predicts)
    csv_target_genes = np.concatenate(csv_target_genes)
    csv_labels = np.concatenate(csv_labels)

    # best
    out_csv = []
    for i in range(len(csv_predicts)):
        out_csv.append([csv_models[i], csv_gRNA_ids[i], csv_ms_match_idts[i], csv_target_genes[i], csv_gRNA_seqs[i], \
                        csv_tRNA_seqs[i], csv_g_folds[i], csv_icshapes[i], csv_binding_ps[i], \
                        csv_rpkms[i], csv_g_mfes[i], csv_h_mfes[i], \
                        csv_st_relate_lens[i], csv_ed_relate_lens[i], \
                        csv_utr5_rates[i], csv_cds_rates[i], csv_utr3_rates[i], \
                        csv_read_depths[i], csv_norm_labels[i], csv_predicts[i], csv_labels[i], i])
    out_data = pd.DataFrame(out_csv, columns=['models', 'gRNA_ids', 'ms_match_idts',            'target_gene', 'gRNA_seqs', \
                    'tRNA_seqs', 'g_folds', 'icshapes', 'binding_ps', \
                    'rpkms', 'g_mfes', 'h_mfes', \
                    'st_relate_lens', 'ed_relate_lens', \
                    'utr5_rates', 'cds_rates', 'utr3_rates', \
                    'read_depths', 'norm_labels', 'norm_predicts', 'label', 'predicts'])
    for gene in out_data['target_gene'].unique():
        col = 'norm_predicts'
        tar = 'predicts'
        denormalized_lfc = out_data.loc[out_data['target_gene'] == gene, col].to_numpy()
        denormalized_lfc = denormalized_lfc * params.loc[gene, 'scale'] + params.loc[gene, 'location']
        out_data.loc[out_data['target_gene'] == gene, tar] = denormalized_lfc

    name = cfg.data_path.split('/')[-1][:-4]
    out_data.to_csv('{}/{}.csv'.format(output_path, name), sep=',', index=False, header=True)

def control(**kwargs):
    global_init()
    cfg._parse(kwargs)

    # features setting, Params `des` option.
    features_groups = {
        'seq': [None], # seq only
        'seq_fold': ['g_fold'],
        'seq_mfe1': ['g_mfe'],
        'seq_mfe2': ['h_mfe'],
        'seq_icshape': ['icshape'],
        'seq_binding': ['binding_p'],
        'seq_rpkm': ['rpkm'],
        'seq_relatelen': ['st_relate_len', 'ed_relate_len'],
        'seq_utrrate': ['utr5_rate', 'cds_rate', 'utr3_rate'],
        'seq_bert': ['bert_embedding'] # seq & bert only
        # all: seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate
    }

    # dataset validation setting, Params `data_des` option.
    dataset_groups = {
        'cell_line': ['K562', 'HEK293FT', 'A375', 'HAP1'],
        'target_gene': [None], # 4925 gene
        'match': [
            ['na', 'SM', 'CD', 'CI', 'DD', 'DI', 'DM', 'RDM', 'RTM', 'SD', 'SI', 'TM'], # 0 all
            ['na', 'CD', 'CI', 'DD', 'DI', 'SD', 'SI'],# 1
            ['na', 'SM', 'DM', 'RDM', 'TM', 'RTM'],# 2
            ['na'], # 3
            ['CD', 'CI', 'DD', 'DI', 'SD', 'SI', 'SM', 'DM', 'RDM', 'TM', 'RTM'], # 4
            ['SM', 'DM', 'RDM', 'TM', 'RTM'], # 5
            ['CD', 'CI', 'DD', 'DI', 'SD', 'SI'] # 6
        ],
        'random': [None]
    }

    # path = '/home/postphd/SJD/val_random.csv'
    path = '/home/postphd/SJD/cas13d_data/final_data/new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    path = cfg.data_path
    
    if cfg.data_des[:3] == 'abc':
        path = '/home/postphd/SJD/cas13d_data/labeled-new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'

    # if cfg.data_des == 'random_all':
    #     path = '/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'

    origin = '/home/postphd/SJD/cas13d_data/final_data/new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'

    data = pd.read_csv(path, sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.)) # h_mfe
    data[24] = data[24].replace('na', float(0.)) # RPKM
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    origin = pd.read_csv(origin, sep=',', encoding='GBK', header=None)
    origin[11] = origin[11].replace('na', float(0.)) # h_mfe
    origin[24] = origin[24].replace('na', float(0.)) # RPKM
    origin = origin.iloc[1:]
    origin[14] = origin[14].astype(float)

    # norm
    # df = df[df.guide_type == 'PM'].groupby('gene')['observed_lfc']
    # tmp = data[data[20] == 'na'].groupby(3)[14] # MisMatch
    tmp = origin.groupby(3)[14]
    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']
    name = path.split('/')[-1][:-4]
    # params.to_csv('params/name.csv'.format(name), sep=',', index=False, header=True)
    for gene in data[3].unique():
        col = 14 # lfc column
        tar = 21 # norm lfc column
        lfc = data.loc[data[3] == gene, col].to_numpy()
        lfc = (lfc - params.loc[gene, 'location']+1e-5) / (params.loc[gene, 'scale']+1e-5)
        data.loc[data[3] == gene, tar] = lfc


    device = torch.device('cuda:{}'.format(cfg.gpu_ids[0]))

    print(device)
    
    dset = getattr(dataset, cfg.dataset)

    fold = 0
    train_data, val_data = None, None
    if cfg.data_des[:4] == 'cell':
        cline = cfg.data_des.split(':')[1]
        train_data = data[~data[2].isin([cline])]
        val_data = data[data[2].isin([cline])]
        tdata = dset(train_data, max_len=30)
        vdata = dset(val_data, max_len=30)
        train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        train(train_loader, test_loader, cfg, device, fold, params)
    elif cfg.data_des[:4] == 'targ':
        # Group by target gene
        tmp = data
        sorted_categories = sorted(tmp[3].unique())
        category_groups = {category: idx//493 for idx, category in enumerate(sorted_categories)}
        tmp['Group'] = tmp[3].map(category_groups)
        group_kfold = GroupKFold(n_splits=10)
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(tmp, tmp[21], groups=tmp['Group'])):
            tmp1 = tmp.iloc[train_idx]
            tmp2 = tmp.iloc[val_idx]
            tdata = dset(tmp1, max_len=30)
            vdata = dset(tmp2, max_len=30)
            train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
            test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
            train(train_loader, test_loader, cfg, device, fold, params)
    elif cfg.data_des == 'random':
        tmp = data
        tdata = dset(tmp, max_len=30)
        kf = KFold(n_splits=10, shuffle=True)
        for fold, (train_idx, test_idx) in enumerate(kf.split(tdata)):
            train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(train_idx), drop_last=True)
            test_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(test_idx), drop_last=True)
            train(train_loader, test_loader, cfg, device, fold, params)
    elif cfg.data_des == 'random_all':
        tmp = data
        tdata = dset(tmp, max_len=30)
        train_loader = DataLoader(tdata, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)

        # vpath = '/data3/SJD/Ca13TransformerDesigned3/shape_data/ours/val_random:full_icshape.csv'
        vpath = '/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_random.csv'
        valdata = pd.read_csv(vpath, sep=',', encoding='GBK', header=None)
        valdata[11] = valdata[11].replace('na', float(0.)) # h_mfe
        valdata[24] = valdata[24].replace('na', float(0.)) # RPKM
        valdata = valdata.iloc[1:]
        valdata[14] = valdata[14].astype(float)
        for gene in valdata[3].unique():
            col = 14 # lfc column
            tar = 21 # norm lfc column
            lfc = valdata.loc[valdata[3] == gene, col].to_numpy()
            lfc = (lfc - params.loc[gene, 'location']+1e-5) / (params.loc[gene, 'scale']+1e-5)
            valdata.loc[valdata[3] == gene, tar] = lfc

        valdata = dset(valdata, max_len=30)
        test_loader = DataLoader(valdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        train(train_loader, test_loader, cfg, device, fold, params)  
    elif cfg.data_des[:11] == 'random_all:':
        tmp = data
        cline = cfg.data_des.split(':')[1]
        tmp_ = tmp[tmp[2].isin([cline])]
        print('All data len is : ', len(tmp_))
        tdata = dset(tmp_, max_len=30)
        train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        test_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        train(train_loader, test_loader, cfg, device, fold, params)  
    elif cfg.data_des[:4] == 'matc':
        idd = int(cfg.data_des.split(':')[1])
        tmp = data
        tmp = tmp[tmp[20].isin(dataset_groups['match'][idd])]
        tdata = dset(tmp, max_len=30)
        kf = KFold(n_splits=10, shuffle=True)
        for fold, (train_idx, test_idx) in enumerate(kf.split(tdata)):
            train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(train_idx), drop_last=True)
            test_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(test_idx), drop_last=True)
            train(train_loader, test_loader, cfg, device, fold, params)
    elif cfg.data_des[:3] == 'abc':
        abc = cfg.data_des[-1]
        train_data = data[data[0].isin([abc])]
        val_data = data[~data[0].isin([abc])]
        print('ABC, train data size: ', len(train_data))
        print('ABC, val data size: ', len(val_data))
        tdata = dset(train_data, max_len=30)
        vdata = dset(val_data, max_len=30)
        train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        train(train_loader, test_loader, cfg, device, fold, params)
    else:
        assert Exception('Invalid data_des!')

def add_single_celline(**kwargs):
    global_init()
    cfg._parse(kwargs)

    # features setting, Params `des` option.
    features_groups = {
        'seq': [None], # seq only
        'seq_fold': ['g_fold'],
        'seq_mfe1': ['g_mfe'],
        'seq_mfe2': ['h_mfe'],
        'seq_icshape': ['icshape'],
        'seq_binding': ['binding_p'],
        'seq_rpkm': ['rpkm'],
        'seq_relatelen': ['st_relate_len', 'ed_relate_len'],
        'seq_utrrate': ['utr5_rate', 'cds_rate', 'utr3_rate'],
        'seq_bert': ['bert_embedding'] # seq & bert only
        # all: seq_bert_fold_mfe1_mfe2_icshape_binding_rpkm_relatelen_utrrate
    }

    # dataset validation setting, Params `data_des` option.
    dataset_groups = {
        'cell_line': ['K562', 'HEK293FT', 'A375', 'HAP1'],
        'target_gene': [None], # 4925 gene
        'match': [
            ['na', 'SM', 'CD', 'CI', 'DD', 'DI', 'DM', 'RDM', 'RTM', 'SD', 'SI', 'TM'], # 0 all
            ['na', 'CD', 'CI', 'DD', 'DI', 'SD', 'SI'],# 1
            ['na', 'SM', 'DM', 'RDM', 'TM', 'RTM'],# 2
            ['na'], # 3
            ['CD', 'CI', 'DD', 'DI', 'SD', 'SI', 'SM', 'DM', 'RDM', 'TM', 'RTM'], # 4
            ['SM', 'DM', 'RDM', 'TM', 'RTM'], # 5
            ['CD', 'CI', 'DD', 'DI', 'SD', 'SI'] # 6
        ],
        'random': [None]
    }

    # path = '/home/postphd/SJD/val_random.csv'
    path = '/home/postphd/SJD/cas13d_data/final_data/new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    
    if cfg.data_des[:4] == 'abca':
        path = '/home/ai/suiru_lu/Cas13d_gRNA_data/final/labeled_source/a_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo.csv'
    elif cfg.data_des[:4] == 'abcb':
        path = '/home/ai/suiru_lu/Cas13d_gRNA_data/final/labeled_source/b_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo.csv'
    elif cfg.data_des[:4] == 'abcc':
        path = '/home/ai/suiru_lu/Cas13d_gRNA_data/final/labeled_source/c_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo.csv'

    data = pd.read_csv(path, sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.)) # h_mfe
    data[24] = data[24].replace('na', float(0.)) # RPKM
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    # norm
    # df = df[df.guide_type == 'PM'].groupby('gene')['observed_lfc']
    # tmp = data[data[20] == 'na'].groupby(3)[14] # MisMatch
    tmp = data.groupby(3)[14]
    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']
    name = path.split('/')[-1][:-4]
    # params.to_csv('params/name.csv'.format(name), sep=',', index=False, header=True)
    for gene in data[3].unique():
        col = 14 # lfc column
        tar = 21 # norm lfc column
        lfc = data.loc[data[3] == gene, col].to_numpy()
        lfc = (lfc - params.loc[gene, 'location']) / params.loc[gene, 'scale']
        data.loc[data[3] == gene, tar] = lfc


    device = torch.device('cuda:{}'.format(cfg.gpu_ids[0]))

    print(device)
    
    dset = getattr(dataset, cfg.dataset)

    fold = 0
    train_data, val_data = None, None
    if cfg.data_des[:4] == 'cell':
        cline = cfg.data_des.split(':')[1]
        tmp = data[data[2].isin([cline])]
        tdata = dset(tmp, max_len=30)
        kf = KFold(n_splits=10, shuffle=True)
        for fold, (train_idx, test_idx) in enumerate(kf.split(tdata)):
            train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(train_idx), drop_last=True)
            test_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(test_idx), drop_last=True)
            train(train_loader, test_loader, cfg, device, fold, params)
    elif cfg.data_des == 'random':
        tmp = data
        tdata = dset(tmp, max_len=30)
        kf = KFold(n_splits=10, shuffle=True)
        for fold, (train_idx, test_idx) in enumerate(kf.split(tdata)):
            train_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(train_idx), drop_last=True)
            test_loader = DataLoader(tdata, batch_size=cfg.batch_size, num_workers=0, sampler=torch.utils.data.SubsetRandomSampler(test_idx), drop_last=True)
            train(train_loader, test_loader, cfg, device, fold, params)
    else:
        assert Exception('Invalid data_des!')

def val(**kwargs):
    global_init()
    cfg._parse(kwargs)

    path = cfg.data_path
    data = pd.read_csv(path, sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.)) # h_mfe
    data[12] = data[12].replace(np.nan, 'n,a') # icshape
    data[24] = data[24].replace('na', float(0.)) # RPKM
    data = data.iloc[1:]
    data[14] = data[14].astype(float)


    origin = path
    origin = '/home/postphd/SJD/cas13d_data/final_data/new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    origin_data = pd.read_csv(origin, sep=',', encoding='GBK', header=None)

    

    origin_data[11] = origin_data[11].replace('na', float(0.)) # h_mfe
    origin_data[24] = origin_data[24].replace('na', float(0.)) # RPKM
    origin_data = origin_data.iloc[1:]
    origin_data[14] = origin_data[14].astype(float)
    tmp = origin_data.groupby(3)[14]


    loc = tmp.apply(lambda x: np.nanquantile(x, 0.5)).rename('location')
    neg_scale = tmp.apply(lambda x: np.nanquantile(x, 0.1)).rename('negative scale')
    pos_scale = tmp.apply(lambda x: np.nanquantile(x, 0.9)).rename('positive scale')
    params = pd.DataFrame([loc, neg_scale, pos_scale]).T
    params['scale'] = params['positive scale'] - params['negative scale']

    # print(params)
    # params.to_csv('./params.csv', sep=',')

    # assert False



    device = torch.device('cuda:{}'.format(cfg.gpu_ids[0]))
    print('device: ', device)
    dset = getattr(dataset, cfg.dataset)

    ckpt_options = None
    if cfg.data_des[:4] == 'cell':
        ckpt_options = 'ckpt/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/Fold:3.pth'
        val_data = data
        vdata = dset(val_data, max_len=30)
        test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=False)
        test(test_loader, cfg, device, ckpt_options, params)
    elif cfg.data_des[:4] == 'targ':
        assert False
        pass
    elif cfg.data_des[:4] == 'matc':
        ckpt_options = 'ckpt/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/Fold:{}.pth'.format(0)
        val_data = data
        vdata = dset(val_data, max_len=30)
        test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=False)
        test(test_loader, cfg, device, ckpt_options, params)
    elif cfg.data_des == 'random':
        # for fold in range(len(10)):
        fold = 3
        ckpt_options = 'ckpt/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/Fold:{}.pth'.format(fold)
        val_data = data
        vdata = dset(val_data, max_len=30)
        test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=False)
        test(test_loader, cfg, device, ckpt_options, params)
    elif cfg.data_des[:10] == 'random_all':
        fold = 0
        ckpt_options = 'ckpt/' + cfg.model_des + '|' + cfg.data_des + '|' + cfg.model + '/Fold:{}.pth'.format(fold)
        val_data = data
        vdata = dset(val_data, max_len=30)
        test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=False)
        test(test_loader, cfg, device, ckpt_options, params)
    elif cfg.data_des[:4] == 'mism':
        # best model in 208 server,/data3/SJD/Ca13TransformerDesigned3
        ckpt_options = 'ckpt/' + cfg.model_des + '|match:3|' + cfg.model + '/Fold:{}.pth'.format(3)
        # match
        match = cfg.data_des.split(':')[1]
        val_data = data[data[20].isin([match])]

        print('len::  ', len(val_data))
        vdata = dset(val_data, max_len=30)

        test_loader = DataLoader(vdata, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
        test(test_loader, cfg, device, ckpt_options, params)
    else:
        assert Exception('Invalid data_des!')


# python3 -u logicArchi.py generate_other_model_data --data_des='cell_line:K562'
# python3 -u logicArchi.py generate_other_model_data --data_des='random'
# python3 -u logicArchi.py generate_other_model_data --data_des='random:full_icshape' --data_path='/home/postphd/SJD/cas13d_data/icshape_filtered_data/full-icshape-new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
def generate_other_model_data(**kwargs):
    global_init()
    cfg._parse(kwargs)

    # tiger data shape
    def tiger_like(data):
        gene = data[3].copy()
        guide_id = data[1].copy()
        guide_type = data[20].copy()
        lfc_r1, lfc_r2, lfc_r3 = data[14].copy(), data[14].copy(), data[14].copy()
        p5_context, p3_context = data[7].copy(), data[7].copy()
        guide_seq = data[4].copy()
        target_seq = data[5].copy()
        loc_utr_5p = -1
        loc_cds = -1
        loc_utr_3p = -1
        log_gene_len = -1
        junction_dist_5p = -1
        junction_dist_3p = -1
        direct_repeat = -1
        g_quad = -1
        mfe = data[9].copy()
        hybrid_mfe_1_23 = data[11].copy()
        hybrid_mfe_15_9 = -1
        hybrid_mfe_3_12 = -1
        log_unpaired = -1
        log_unpaired_11 = -1
        log_unpaired_19 = -1
        log_unpaired_25 = -1
        guide_fold = 0
        target_fold = 0
        fold = 0
        for i in range(len(data)):
            p1, p2 = data[15].iloc[i], data[16].iloc[i]
            p5_context.iloc[i] = str(p5_context.iloc[i])[:int(p1)]
            p3_context.iloc[i] = str(p3_context.iloc[i])[int(p2):]
        
        tiger_data = pd.DataFrame()
        tiger_data[0] = gene
        tiger_data[1] = guide_id
        tiger_data[2] = guide_type.replace('na', 'PM')
        tiger_data[3] = lfc_r1
        tiger_data[4] = lfc_r2
        tiger_data[5] = lfc_r3
        tiger_data[6] = guide_seq
        tiger_data[7] = target_seq
        tiger_data[8] = p5_context
        tiger_data[9] = p3_context
        tiger_data[10] = -1
        tiger_data[11] = -1
        tiger_data[12] = -1
        tiger_data[13] = -1
        tiger_data[14] = -1
        tiger_data[15] = -1
        tiger_data[16] = -1
        tiger_data[17] = -1
        tiger_data[18] = mfe
        tiger_data[19] = hybrid_mfe_1_23
        tiger_data[20] = -1
        tiger_data[21] = -1
        tiger_data[22] = -1
        tiger_data[23] = -1
        tiger_data[24] = -1
        tiger_data[25] = -1
        tiger_data[26] = -1
        tiger_data[27] = -1
        tiger_data[28] = -1

        tmp = data.copy()
        sorted_categories = sorted(tmp[6].unique())
        category_groups = {category: (idx//493)+1 for idx, category in enumerate(sorted_categories)}
        tmp['Group'] = tmp[6].map(category_groups)
        for i in range(len(tiger_data)):
            tiger_data[26].iloc[i] = random.randint(1, 10)
            tiger_data[27].iloc[i] = tmp['Group'].iloc[i]
            tiger_data[28].iloc[i] = tmp['Group'].iloc[i]
        
        tiger_data.columns = ['gene', 'guide_id', 'guide_type', 'lfc_r1', 'lfc_r2', 'lfc_r3', 'guide_seq', 'target_seq', '5p_context', '3p_context', 'loc_utr_5p', 'loc_cds', 'loc_utr_3p', 'log_gene_len', 'junction_dist_5p', 'junction_dist_3p', 'direct_repeat', 'g_quad', 'mfe', 'hybrid_mfe_1_23', 'hybrid_mfe_15_9', 'hybrid_mfe_3_12', 'log_unpaired', 'log_unpaired_11', 'log_unpaired_19', 'log_unpaired_25', 'guide_fold', 'target_fold', 'fold']

        tiger_data_nt = pd.DataFrame()
        tiger_data_nt[1] = lfc_r1
        tiger_data_nt[2] = lfc_r2
        tiger_data_nt[3] = lfc_r3
        tiger_data_nt.columns = ['lfc_r1', 'lfc_r2', 'lfc_r3']
        # tiger_data_nt.to_csv('./other_model_data/tiger/val_nt_random.csv', index=True)

        return tiger_data, tiger_data_nt

    # deepcas13
    def deepcas13_like(data):
        lfc_r1 = data[14].copy()
        target_seq = data[5].copy()
        target_extent = data[7].copy()
        for i in range(len(data)):
            p1, p2 = int(data[15].iloc[i]), int(data[16].iloc[i])
            if p2 - p1 >= 33:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
            elif p2 - 33 < 0:
                target_extent.iloc[i] = str(target_extent.iloc[i])[0:33]
            elif p1 + 33 >= len(target_extent.iloc[i]):
                target_extent.iloc[i] = str(target_extent.iloc[i])[-33:]
            else:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
        deepcas13_data = pd.DataFrame()
        deepcas13_data[0] = target_extent
        deepcas13_data[1] = lfc_r1
        return deepcas13_data

    # deepcas13 nolabel
    def deepcas13_nolabel_like(data):
        lfc_r1 = data[14].copy()
        target_seq = data[5].copy()
        target_extent = data[7].copy()
        for i in range(len(data)):
            p1, p2 = int(data[15].iloc[i]), int(data[16].iloc[i])
            if p2 - p1 >= 33:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
            elif p2 - 33 < 0:
                target_extent.iloc[i] = str(target_extent.iloc[i])[0:33]
            elif p1 + 33 >= len(target_extent.iloc[i]):
                target_extent.iloc[i] = str(target_extent.iloc[i])[-33:]
            else:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
        deepcas13_data = pd.DataFrame()
        deepcas13_data[0] = target_extent
        # deepcas13_data[1] = lfc_r1
        return deepcas13_data

    # CasRx 
    def CasRx_like(data):
        # guide,gene,refseq,raw ratio,relative_ratio,binary_relative_ratio_061f,binary_relative_ratio_075f,ratio045_cutoff_binary_relative_ratio,ratio,old_relative_ratio,old_binary_relative_ratio_gene20,position,is_5UTR,UTR5_position,is_CDS,CDS_position,is_3UTR,UTR3_position,RNAseq2_relative,RNAseq3_relative,RNAseq7_relative,RNAseq8_relative,np_vivo_ic_has_data,np_vivo_ic_sum,blast_f24_mis3_e1_20_match_num,pos,vienna_2_T37,vienna_2_T60,vienna_2_T70,contrafold_2,eternafold,refseq_target_transcript_percent,ensembl_target_transcript_percent,absolute_position_start,target_seq,nearby_seq_all_5,nearby_seq_all_10,nearby_seq_all_15,nearby_seq_all_20,GC_content,linearfold_dr_flag,linearfold_vals,linearfold_vals_7win,linearfold_vals_23win,target unfold energy,target unfold energy_7win,target unfold energy_23win,bad guide_bottom20%,bad guide_bottom10%,bad guide_90th_pct,bad guide_95th_pct,bins 
        # = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

        target_extent = data[7].copy()
        for i in range(len(data)):
            p1, p2 = int(data[15].iloc[i]), int(data[16].iloc[i])
            if p2 - p1 >= 30:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-30:p2]
            elif p2 - 30 < 0:
                target_extent.iloc[i] = str(target_extent.iloc[i])[0:30]
            elif p1 + 30 >= len(target_extent.iloc[i]):
                target_extent.iloc[i] = str(target_extent.iloc[i])[-30:]
            else:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-30:p2]

        relative_ratio = data[14].copy() # unnormalized
        # tmp = 1 / (1 + np.exp(-relative_ratio))
        tmp = 2 ** (relative_ratio)

        relative_ratio = tmp
        
        guide = target_extent
        gene = data[3].copy()
        classes = None
        linearfold_vals = 0
        is_5UTR = data[31].copy()
        is_CDS = data[32].copy()
        is_3UTR = data[33].copy()
        refseq_target_transcript_percent = 0
        target_unfold_energy = 0
        UTR5_position = data[34].copy()
        CDS_position = data[35].copy()
        UTR3_position = data[36].copy()


        casrx_data= pd.DataFrame()
        casrx_data[0] = 0
        casrx_data[1] = guide
        casrx_data[2] = gene
        casrx_data[3] = 0
        casrx_data[4] = 0
        casrx_data[5] = relative_ratio
        casrx_data[6] = 0
        casrx_data[7] = 0
        casrx_data[8] = 0
        casrx_data[9] = 0
        casrx_data[10] = 0
        casrx_data[11] = 0
        casrx_data[12] = 0
        casrx_data[13] = is_5UTR
        casrx_data[14] = UTR5_position
        casrx_data[15] = is_CDS
        casrx_data[16] = CDS_position
        casrx_data[17] = is_3UTR
        casrx_data[18] = UTR3_position
        for i in range(19, 53):
            casrx_data[i] = 0
        
        casrx_data.columns = ['', 'guide','gene','refseq','raw ratio','relative_ratio','binary_relative_ratio_061f','binary_relative_ratio_075f','ratio045_cutoff_binary_relative_ratio','ratio','old_relative_ratio','old_binary_relative_ratio_gene20','position','is_5UTR','UTR5_position','is_CDS','CDS_position','is_3UTR','UTR3_position','RNAseq2_relative','RNAseq3_relative','RNAseq7_relative','RNAseq8_relative','np_vivo_ic_has_data','np_vivo_ic_sum','blast_f24_mis3_e1_20_match_num','pos','vienna_2_T37','vienna_2_T60','vienna_2_T70','contrafold_2','eternafold','refseq_target_transcript_percent','ensembl_target_transcript_percent','absolute_position_start','target_seq','nearby_seq_all_5','nearby_seq_all_10','nearby_seq_all_15','nearby_seq_all_20','GC_content','linearfold_dr_flag','linearfold_vals','linearfold_vals_7win','linearfold_vals_23win','target unfold energy','target unfold energy_7win','target unfold energy_23win','bad guide_bottom20%','bad guide_bottom10%','bad guide_90th_pct','bad guide_95th_pct','bins']
        '''
        classes = dataframe['binary_relative_ratio_075f'].values
        outputs = dataframe['relative_ratio'].values if args.regression else classes.astype(np.float32)

        other_single_value_inputs = np.empty((9, num_examples))
        other_single_value_inputs[0, :] = dataframe['linearfold_vals'].values
        other_single_value_inputs[1, :] = dataframe['is_5UTR'].values
        other_single_value_inputs[2, :] = dataframe['is_CDS'].values
        other_single_value_inputs[3, :] = dataframe['is_3UTR'].values
        other_single_value_inputs[4, :] = dataframe['refseq_target_transcript_percent'].values
        other_single_value_inputs[5, :] = dataframe['target unfold energy'].values
        other_single_value_inputs[6, :] = dataframe['UTR5_position'].values
        other_single_value_inputs[7, :] = dataframe['CDS_position'].values
        other_single_value_inputs[8, :] = dataframe['UTR3_position'].values
        '''

        return casrx_data
        

    def our_method(data):
        return data


    path = cfg.data_path
    data = pd.read_csv(path, sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.)) # h_mfe
    data[24] = data[24].replace('na', float(0.)) # RPKM
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    # path = './other_model_data/'
    path = './shape_data/'
    tiger_path = path + 'tiger'
    deepcas13_path = path + 'deepcas13'
    casrx_path = path + 'casrx'
    ours_path = path + 'ours'
    if not os.path.exists(tiger_path):
        os.mkdir(tiger_path)
    if not os.path.exists(deepcas13_path):
        os.mkdir(deepcas13_path)
    if not os.path.exists(casrx_path):
        os.mkdir(casrx_path)
    if not os.path.exists(ours_path):
        os.mkdir(ours_path)

    # selected_columns = df[['column1', 'column2']]

    # ##################
    train_data, val_data = None, None
    if cfg.data_des[:4] == 'cell':
        cline = cfg.data_des.split(':')[1]
        train_data = data[~data[2].isin([cline])]
        val_data = data[data[2].isin([cline])]
        tiger_train_data = tiger_like(train_data)
        tiger_val_data = tiger_like(val_data)
        print(tiger_train_data)
        print(tiger_val_data)

    elif cfg.data_des[:4] == 'targ':
        # Group by target gene
        tmp = data
        sorted_categories = sorted(tmp[3].unique())
        category_groups = {category: idx//493 for idx, category in enumerate(sorted_categories)}
        tmp['Group'] = tmp[3].map(category_groups)
        group_kfold = GroupKFold(n_splits=10)
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(tmp, tmp[21], groups=tmp['Group'])):
            tmp1 = tmp.iloc[train_idx]
            tmp2 = tmp.iloc[val_idx]
            # tdata = dset(tmp1, max_len=30)
            # vdata = dset(tmp2, max_len=30)

    elif cfg.data_des[:6] == 'random':
        train_data = data.sample(frac=0.9, random_state=1024)
        val_data = data.drop(train_data.index)
        # tiger
        tiger_train_data, tiger_train_nt_data = tiger_like(train_data)
        tiger_val_data, tiger_val_nt_data = tiger_like(val_data)
        tiger_train_data.to_csv(tiger_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        tiger_val_data.to_csv(tiger_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        
        tiger_train_nt_data.to_csv(tiger_path+'/train_nt_{}.csv'.format(cfg.data_des), index=True)
        tiger_val_nt_data.to_csv(tiger_path+'/val_nt_{}.csv'.format(cfg.data_des), index=True)

        # deepcas13
        deepcas13_train_data = deepcas13_like(train_data)
        deepcas13_val_data = deepcas13_like(val_data)
        deepcas13_val_data_nolabel = deepcas13_nolabel_like(val_data)
        deepcas13_train_data.to_csv(deepcas13_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        deepcas13_val_data.to_csv(deepcas13_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        deepcas13_val_data_nolabel.to_csv(deepcas13_path+'/val_{}_nolabel.csv'.format(cfg.data_des), index=False)
        

        # Ours
        ours_train_data = our_method(train_data)
        ours_val_data = our_method(val_data)
        ours_train_data.to_csv(ours_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        ours_val_data.to_csv(ours_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        

        # CasRx_guide
        casrx_train_data = CasRx_like(train_data)
        casrx_val_data = CasRx_like(val_data)
        casrx_train_data.to_csv(casrx_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        casrx_val_data.to_csv(casrx_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        
    elif cfg.data_des[:4] == 'all_':
        val_data = data
        # tiger
        tiger_val_data, tiger_val_nt_data = tiger_like(val_data)
        tiger_val_data.to_csv(tiger_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        tiger_val_nt_data.to_csv(tiger_path+'/val_nt_{}.csv'.format(cfg.data_des), index=True)


        # deepcas13
        deepcas13_val_data = deepcas13_like(val_data)
        deepcas13_val_data_nolabel = deepcas13_nolabel_like(val_data)
        deepcas13_val_data.to_csv(deepcas13_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        deepcas13_val_data_nolabel.to_csv(deepcas13_path+'/val_{}_nolabel.csv'.format(cfg.data_des), index=False)
        

        # Ours
        ours_val_data = our_method(val_data)
        ours_val_data.to_csv(ours_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        

        # CasRx_guide
        casrx_val_data = CasRx_like(val_data)
        casrx_val_data.to_csv(casrx_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        

    elif cfg.data_des[:4] == 'matc':
        idd = int(cfg.data_des.split(':')[1])
        tmp = data
        tmp = tmp[tmp[20].isin(dataset_groups['match'][idd])]
        # tdata = dset(tmp, max_len=30)


    else:
        assert Exception('Invalid data_des!')

# python3 -u logicArchi.py generate_other_model_data --data_des='cell_line:K562'
def generate_cellline_data(**kwargs):
    global_init()
    cfg._parse(kwargs)

    # tiger data shape
    def tiger_like(data):
        gene = data[3].copy()
        guide_id = data[1].copy()
        guide_type = data[20].copy()
        lfc_r1, lfc_r2, lfc_r3 = data[14].copy(), data[14].copy(), data[14].copy()
        p5_context, p3_context = data[7].copy(), data[7].copy()
        guide_seq = data[4].copy()
        target_seq = data[5].copy()
        loc_utr_5p = -1
        loc_cds = -1
        loc_utr_3p = -1
        log_gene_len = -1
        junction_dist_5p = -1
        junction_dist_3p = -1
        direct_repeat = -1
        g_quad = -1
        mfe = data[9].copy()
        hybrid_mfe_1_23 = data[11].copy()
        hybrid_mfe_15_9 = -1
        hybrid_mfe_3_12 = -1
        log_unpaired = -1
        log_unpaired_11 = -1
        log_unpaired_19 = -1
        log_unpaired_25 = -1
        guide_fold = 0
        target_fold = 0
        fold = 0
        for i in range(len(data)):
            p1, p2 = data[15].iloc[i], data[16].iloc[i]
            p5_context.iloc[i] = str(p5_context.iloc[i])[:int(p1)]
            p3_context.iloc[i] = str(p3_context.iloc[i])[int(p2):]
        
        tiger_data = pd.DataFrame()
        tiger_data[0] = gene
        tiger_data[1] = guide_id
        tiger_data[2] = guide_type.replace('na', 'PM')
        tiger_data[3] = lfc_r1
        tiger_data[4] = lfc_r2
        tiger_data[5] = lfc_r3
        tiger_data[6] = guide_seq
        tiger_data[7] = target_seq
        tiger_data[8] = p5_context
        tiger_data[9] = p3_context
        tiger_data[10] = -1
        tiger_data[11] = -1
        tiger_data[12] = -1
        tiger_data[13] = -1
        tiger_data[14] = -1
        tiger_data[15] = -1
        tiger_data[16] = -1
        tiger_data[17] = -1
        tiger_data[18] = mfe
        tiger_data[19] = hybrid_mfe_1_23
        tiger_data[20] = -1
        tiger_data[21] = -1
        tiger_data[22] = -1
        tiger_data[23] = -1
        tiger_data[24] = -1
        tiger_data[25] = -1
        tiger_data[26] = -1
        tiger_data[27] = -1
        tiger_data[28] = -1

        tmp = data.copy()
        sorted_categories = sorted(tmp[6].unique())
        category_groups = {category: (idx//493)+1 for idx, category in enumerate(sorted_categories)}
        tmp['Group'] = tmp[6].map(category_groups)
        for i in range(len(tiger_data)):
            tiger_data[26].iloc[i] = random.randint(1, 10)
            tiger_data[27].iloc[i] = tmp['Group'].iloc[i]
            tiger_data[28].iloc[i] = tmp['Group'].iloc[i]
        
        tiger_data.columns = ['gene', 'guide_id', 'guide_type', 'lfc_r1', 'lfc_r2', 'lfc_r3', 'guide_seq', 'target_seq', '5p_context', '3p_context', 'loc_utr_5p', 'loc_cds', 'loc_utr_3p', 'log_gene_len', 'junction_dist_5p', 'junction_dist_3p', 'direct_repeat', 'g_quad', 'mfe', 'hybrid_mfe_1_23', 'hybrid_mfe_15_9', 'hybrid_mfe_3_12', 'log_unpaired', 'log_unpaired_11', 'log_unpaired_19', 'log_unpaired_25', 'guide_fold', 'target_fold', 'fold']

        tiger_data_nt = pd.DataFrame()
        tiger_data_nt[1] = lfc_r1
        tiger_data_nt[2] = lfc_r2
        tiger_data_nt[3] = lfc_r3
        tiger_data_nt.columns = ['lfc_r1', 'lfc_r2', 'lfc_r3']
        # tiger_data_nt.to_csv('./other_model_data/tiger/val_nt_random.csv', index=True)

        return tiger_data, tiger_data_nt

    # deepcas13
    def deepcas13_like(data):
        lfc_r1 = data[14].copy()
        target_seq = data[5].copy()
        target_extent = data[7].copy()
        for i in range(len(data)):
            p1, p2 = int(data[15].iloc[i]), int(data[16].iloc[i])
            if p2 - p1 >= 33:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
            elif p2 - 33 < 0:
                target_extent.iloc[i] = str(target_extent.iloc[i])[0:33]
            elif p1 + 33 >= len(target_extent.iloc[i]):
                target_extent.iloc[i] = str(target_extent.iloc[i])[-33:]
            else:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
        deepcas13_data = pd.DataFrame()
        deepcas13_data[0] = target_extent
        deepcas13_data[1] = lfc_r1
        return deepcas13_data

    # deepcas13 nolabel
    def deepcas13_nolabel_like(data):
        lfc_r1 = data[14].copy()
        target_seq = data[5].copy()
        target_extent = data[7].copy()
        for i in range(len(data)):
            p1, p2 = int(data[15].iloc[i]), int(data[16].iloc[i])
            if p2 - p1 >= 33:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
            elif p2 - 33 < 0:
                target_extent.iloc[i] = str(target_extent.iloc[i])[0:33]
            elif p1 + 33 >= len(target_extent.iloc[i]):
                target_extent.iloc[i] = str(target_extent.iloc[i])[-33:]
            else:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-33:p2]
        deepcas13_data = pd.DataFrame()
        deepcas13_data[0] = target_extent
        # deepcas13_data[1] = lfc_r1
        return deepcas13_data

    # CasRx 
    def CasRx_like(data):
        
        # = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

        target_extent = data[7].copy()
        for i in range(len(data)):
            p1, p2 = int(data[15].iloc[i]), int(data[16].iloc[i])
            if p2 - p1 >= 30:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-30:p2]
            elif p2 - 30 < 0:
                target_extent.iloc[i] = str(target_extent.iloc[i])[0:30]
            elif p1 + 30 >= len(target_extent.iloc[i]):
                target_extent.iloc[i] = str(target_extent.iloc[i])[-30:]
            else:
                target_extent.iloc[i] = str(target_extent.iloc[i])[p2-30:p2]

        relative_ratio = data[14].copy() # unnormalized
        # tmp = 1 / (1 + np.exp(-relative_ratio))
        tmp = 2 ** (relative_ratio)

        relative_ratio = tmp
        
        guide = target_extent
        gene = data[3].copy()
        classes = None
        linearfold_vals = 0
        is_5UTR = data[31].copy()
        is_CDS = data[32].copy()
        is_3UTR = data[33].copy()
        refseq_target_transcript_percent = 0
        target_unfold_energy = 0
        UTR5_position = data[34].copy()
        CDS_position = data[35].copy()
        UTR3_position = data[36].copy()


        casrx_data= pd.DataFrame()
        casrx_data[0] = 0
        casrx_data[1] = guide
        casrx_data[2] = gene
        casrx_data[3] = 0
        casrx_data[4] = 0
        casrx_data[5] = relative_ratio
        casrx_data[6] = 0
        casrx_data[7] = 0
        casrx_data[8] = 0
        casrx_data[9] = 0
        casrx_data[10] = 0
        casrx_data[11] = 0
        casrx_data[12] = 0
        casrx_data[13] = is_5UTR
        casrx_data[14] = UTR5_position
        casrx_data[15] = is_CDS
        casrx_data[16] = CDS_position
        casrx_data[17] = is_3UTR
        casrx_data[18] = UTR3_position
        for i in range(19, 53):
            casrx_data[i] = 0
        
        casrx_data.columns = ['', 'guide','gene','refseq','raw ratio','relative_ratio','binary_relative_ratio_061f','binary_relative_ratio_075f','ratio045_cutoff_binary_relative_ratio','ratio','old_relative_ratio','old_binary_relative_ratio_gene20','position','is_5UTR','UTR5_position','is_CDS','CDS_position','is_3UTR','UTR3_position','RNAseq2_relative','RNAseq3_relative','RNAseq7_relative','RNAseq8_relative','np_vivo_ic_has_data','np_vivo_ic_sum','blast_f24_mis3_e1_20_match_num','pos','vienna_2_T37','vienna_2_T60','vienna_2_T70','contrafold_2','eternafold','refseq_target_transcript_percent','ensembl_target_transcript_percent','absolute_position_start','target_seq','nearby_seq_all_5','nearby_seq_all_10','nearby_seq_all_15','nearby_seq_all_20','GC_content','linearfold_dr_flag','linearfold_vals','linearfold_vals_7win','linearfold_vals_23win','target unfold energy','target unfold energy_7win','target unfold energy_23win','bad guide_bottom20%','bad guide_bottom10%','bad guide_90th_pct','bad guide_95th_pct','bins']
        '''
        classes = dataframe['binary_relative_ratio_075f'].values
        outputs = dataframe['relative_ratio'].values if args.regression else classes.astype(np.float32)

        other_single_value_inputs = np.empty((9, num_examples))
        other_single_value_inputs[0, :] = dataframe['linearfold_vals'].values
        other_single_value_inputs[1, :] = dataframe['is_5UTR'].values
        other_single_value_inputs[2, :] = dataframe['is_CDS'].values
        other_single_value_inputs[3, :] = dataframe['is_3UTR'].values
        other_single_value_inputs[4, :] = dataframe['refseq_target_transcript_percent'].values
        other_single_value_inputs[5, :] = dataframe['target unfold energy'].values
        other_single_value_inputs[6, :] = dataframe['UTR5_position'].values
        other_single_value_inputs[7, :] = dataframe['CDS_position'].values
        other_single_value_inputs[8, :] = dataframe['UTR3_position'].values
        '''

        return casrx_data
        

    def our_method(data):
        return data


    path = cfg.data_path
    data = pd.read_csv(path, sep=',', encoding='GBK', header=None)
    data[11] = data[11].replace('na', float(0.)) # h_mfe
    data[24] = data[24].replace('na', float(0.)) # RPKM
    data = data.iloc[1:]
    data[14] = data[14].astype(float)

    path = './other_model_data/'
    tiger_path = path + 'tiger'
    deepcas13_path = path + 'deepcas13'
    casrx_path = path + 'casrx'
    ours_path = path + 'ours'
    if not os.path.exists(tiger_path):
        os.mkdir(tiger_path)
    if not os.path.exists(deepcas13_path):
        os.mkdir(deepcas13_path)
    if not os.path.exists(casrx_path):
        os.mkdir(casrx_path)
    if not os.path.exists(ours_path):
        os.mkdir(ours_path)

    # selected_columns = df[['column1', 'column2']]

    # ##################
    train_data, val_data = None, None
    if cfg.data_des[:4] == 'NOUSED':
        cline = cfg.data_des.split(':')[1]
        train_data = data[~data[2].isin([cline])]
        val_data = data[data[2].isin([cline])]
        tiger_train_data = tiger_like(train_data)
        tiger_val_data = tiger_like(val_data)
        print(tiger_train_data)
        print(tiger_val_data)

    elif cfg.data_des[:4] == 'cell':
        cline = cfg.data_des.split(':')[1]
        data = data[data[2].isin([cline])]
        print('len:  ', len(data))
        # val_data = data[data[2].isin([cline])]
        
        train_data = data.sample(frac=0.9, random_state=1024)
        val_data = data.drop(train_data.index)
        # tiger
        tiger_train_data, tiger_train_nt_data = tiger_like(train_data)
        tiger_val_data, tiger_val_nt_data = tiger_like(val_data)
        tiger_train_data.to_csv(tiger_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        tiger_val_data.to_csv(tiger_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        
        tiger_train_nt_data.to_csv(tiger_path+'/train_nt_{}.csv'.format(cfg.data_des), index=True)
        tiger_val_nt_data.to_csv(tiger_path+'/val_nt_{}.csv'.format(cfg.data_des), index=True)

        '''
        # deepcas13
        deepcas13_train_data = deepcas13_like(train_data)
        deepcas13_val_data = deepcas13_like(val_data)
        deepcas13_val_data_nolabel = deepcas13_nolabel_like(val_data)
        deepcas13_train_data.to_csv(deepcas13_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        deepcas13_val_data.to_csv(deepcas13_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        deepcas13_val_data_nolabel.to_csv(deepcas13_path+'/val_{}_nolabel.csv'.format(cfg.data_des), index=False)
        

        # Ours
        ours_train_data = our_method(train_data)
        ours_val_data = our_method(val_data)
        ours_train_data.to_csv(ours_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        ours_val_data.to_csv(ours_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        

        # CasRx_guide
        casrx_train_data = CasRx_like(train_data)
        casrx_val_data = CasRx_like(val_data)
        casrx_train_data.to_csv(casrx_path+'/train_{}.csv'.format(cfg.data_des), index=False)
        casrx_val_data.to_csv(casrx_path+'/val_{}.csv'.format(cfg.data_des), index=False)
        '''
        


    elif cfg.data_des[:4] == 'matc':
        idd = int(cfg.data_des.split(':')[1])
        tmp = data
        tmp = tmp[tmp[20].isin(dataset_groups['match'][idd])]
        # tdata = dset(tmp, max_len=30)


    else:
        assert Exception('Invalid data_des!')


if __name__ == '__main__':
    import fire
    fire.Fire()
