# Import
import matplotlib.pyplot as plt
from tueplots import bundles
import tueplots
# plt.rcParams.update(bundles.iclr2024())
import numpy as np
import math
import os

# Constant
TABLE_NAME = "generaltable7_fewshot"
ECOLOR ='orange'
BAR_COLOR = tueplots.constants.color.rgb.tue_blue
BAR_WIDTH = 0.93

# Data
MODELS = ['Our 70B', 'Our 13B', 'Our 7B', 'Llama-2 13B', 'Llama-2 7B', 'Vietcuna 7B', 'MixSUra 7x8B', 'Gemini Pro', 'GPT-3.5-turbo', 'GPT-4']
MODELS.remove('Gemini Pro')
EXCLUDE_MODELS = ['Gemini Pro','GPT-3.5-turbo', 'GPT-4']

if not os.path.exists(TABLE_NAME):
    os.mkdir(TABLE_NAME)

sent_task = {
    "VLSP 2016": {
        "AC": {
            "mean": [0.66, 0.59, 0.57, 0.51, 0.45, 0.04, 0.62, 0.65, 0.75],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.01, 0.01],
        },
        "F1": {
            "mean": [0.49, 0.57, 0.42, 0.41, 0.32, 0.05, 0.63, 0.59, 0.74],
            "std": [0.01, 0.01, 0.05, 0.06, 0.01, 0.01, 0.00, 0.1, 0.01],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.72, 0.67, 0.69, 0.66, 0.59, 0.45, 0.59, 0.00, 0.00],
            "std": [0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.00, 0.32, 0.41],
        },
        "ECE": {
            "mean": [0.13, 0.09, 0.07, 0.32, 0.26, 0.71, 0.30, 0.32, 0.41],
            "std": [0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.00, 0.01, 0.01],
        },
        "A@10": {
            "mean": [0.77, 0.82, 0.77, 0.80, 0.50, 0.05, 0.59, 0.65, 0.74],
            "std": [0.04, 0.04, 0.04, 0.04, 0.05, 0.02, 0.00, 0.05, 0.04],
        },
    },
    "UiT-VSFC": {
        "AC": {
            "mean": [0.75, 0.74, 0.72, 0.63, 0.50, 0.03, 0.74, 0.86, 0.85],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01, 0.01],
        },
        "F1": {
            "mean": [0.48, 0.52, 0.43, 0.46, 0.34, 0.03, 0.46, 0.73, 0.59],
            "std": [0.01, 0.08, 0.01, 0.07, 0.01, 0.00, 0.00, 0.01, 0.09],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.81, 0.83, 0.78, 0.71, 0.69, 0.53, 0.63, 0.00, 0.00],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.52, 0.52],
        },
        "ECE": {
            "mean": [0.16, 0.10, 0.13, 0.13, 0.23, 0.50, 0.23, 0.52, 0.52],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01, 0.01],
        },
        "A@10": {
            "mean": [0.71, 0.87, 0.95, 0.88, 0.62, 0.01, 0.655, 0.86, 0.85],
            "std": [0.02, 0.02, 0.03, 0.02, 0.03, 0.00, 0.00, 0.02, 0.02],
        },
    },
}

tcl_task = {
    "UiT-VSMEC": {
        "AC": {
            "mean": [0.25, 0.32, 0.29, 0.18, 0.25, 0.15, 0.40, 0.42, 0.49],
            "std": [0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.00, 0.02, 0.02],
        },
        "F1": {
            "mean": [0.15, 0.12, 0.11, 0.08, 0.12, 0.22, 0.36, 0.40, 0.48],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.00, 0.02, 0.02],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.56, 0.58, 0.60, 0.55, 0.57, 0.83, 0.72, 0.00, 0.00],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.28, 0.35],
        },
        "ECE": {
            "mean": [0.25, 0.22, 0.12, 0.45, 0.21, 0.81, 0.53, 0.28, 0.35],
            "std": [0.02, 0.02, 0.02, 0.01, 0.02, 0.01, 0.00, 0.02, 0.02],
        },
        "A@10": {
            "mean": [0.37, 0.57, 0.43, 0.49, 0.54, 0.13, 0.79, 0.42, 0.49],
            "std": [0.06, 0.07, 0.06, 0.07, 0.06, 0.04, 0.00, 0.06, 0.06],
        },
    },
    "PhoATIS": {
        "AC": {
            "mean": [0.15, 0.01, 0.06, 0.02, 0.03, 0.04, 0.81, 0.69, 0.85],
            "std": [0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.00, 0.02, 0.01],
        },
        "F1": {
            "mean": [0.22, 0.06, 0.01, 0.06, 0.02, 0.01, 0.58, 0.67, 0.78],
            "std": [0.03, 0.02, 0.00, 0.02, 0.01, 0.00, 0.00, 0.03, 0.03],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.83, 0.47, 0.57, 0.56, 0.56, 0.63, 0.96, 0.00, 0.00],
            "std": [0.00, 0.00, 0.01, 0.01, 0.01, 0.00, 0.00, 0.63, 0.79],
        },
        "ECE": {
            "mean": [0.81, 0.84, 0.24, 0.90, 0.54, 0.21, 0.14, 0.63, 0.79],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.02, 0.01],
        },
        "A@10": {
            "mean": [0.13, 0.00, 0.08, 0.01, 0.01, 0.07, 0.91, 0.69, 0.88],
            "std": [0.04, 0.01, 0.03, 0.01, 0.01, 0.03, 0.00, 0.05, 0.04],
        },
    },
}

kn_task = {
    "ZaloE2E": {
        "EM": {
            "mean": [0.34, 0.26, 0.14, 0.22, 0.07, 0.07, 0.19, 0.49, 0.49],
            "std": [0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02],
        },
        "F1": {
            "mean": [0.50, 0.40, 0.25, 0.36, 0.15, 0.19, 0.34, 0.64, 0.64],
            "std": [0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.02],
        },
    },
    "ViMMRC": {
        "AC": {
            "mean": [0.78, 0.62, 0.42, 0.58, 0.30, 0.31, 0.65, 0.90, 0.91],
            "std": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0, 0.01, 0.01],
        },
        "F1": {
            "mean": [0.63, 0.50, 0.33, 0.46, 0.23, 0.18, 0.64, 0.73, 0.73],
            "std": [0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0, 0.03, 0.04],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.90, 0.69, 0.61, 0.62, 0.56, 0.50, 0.54, 0.00, 0.00],
            "std": [0.01, 0.02, 0.02, 0.02, 0.02, 0.00, 0, 0.66, 0.66],
        },
        "ECE": {
            "mean": [0.13, 0.18, 0.13, 0.28, 0.43, 0.06, 0.29, 0.66, 0.66],
            "std": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0, 0.01, 0.01],
        },
        "A@10": {
            "mean": [0.96, 0.65, 0.39, 0.77, 0.16, 0.31, 0.65, 0.91, 0.91],
            "std": [0.03, 0.07, 0.07, 0.06, 0.05, 0.06, 0, 0.04, 0.04],
        },
    },
}

td_task = {
    "UiT-ViCTSD": {
        "AC": {
            "mean": [0.44, 0.44, 0.43, 0.28, 0.16, 0.08, 0.70, 0.63, 0.89],
            "std": [0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.02, 0.00],
        },
        "F1": {
            "mean": [0.27, 0.30, 0.40, 0.19, 0.12, 0.10, 0.39, 0.54, 0.71],
            "std": [0.01, 0.05, 0.01, 0.00, 0.01, 0.01, 0.00, 0.02, 0.01],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.75, 0.67, 0.60, 0.67, 0.61, 0.50, 0.00, 0.0, 0.00],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0, 0.13, 0.39],
        },
        "ECE": {
            "mean": [0.52, 0.33, 0.29, 0.52, 0.66, 0.42, 0.29, 0.13, 0.39],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.02, 0.00],
        },
        "A@10": {
            "mean": [0.37, 0.41, 0.71, 0.63, 0.08, 0.08, 0.80, 0.63, 0.89],
            "std": [0.02, 0.03, 0.02, 0.03, 0.02, 0.03, 0.00, 0.05, 0.03],
        },
    },
    "UiT-ViHSD": {
        "AC": {
            "mean": [0.17, 0.26, 0.16, 0.17, 0.01, 0.61, 0.58, 0.63, 0.77],
            "std": [0.00, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.01, 0.01],
        },
        "F1": {
            "mean": [0.15, 0.16, 0.10, 0.11, 0.01, 0.21, 0.31, 0.47, 0.57],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.01, 0.01],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.64, 0.61, 0.67, 0.62, 0.56, 0.50, 0.68, 0.00, 0.00],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0, 0.29, 0.44],
        },
        "ECE": {
            "mean": [0.57, 0.42, 0.32, 0.58, 0.71, 0.28, 0.30, 0.29, 0.44],
            "std": [0.00, 0.01, 0.00, 0.00, 0.00, 0.01, 0, 0.01, 0.01],
        },
        "A@10": {
            "mean": [0.27, 0.21, 0.28, 0.44, 0.01, 0.61, 0.93, 0.63, 0.77],
            "std": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.00, 0.02, 0.02],
        },
    },
}

lm_task = {
    "MLQA-MLM": {
        "EM": {
            "mean": [0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.04, 0.08],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "CER": {
            "mean": [0.54, 0.45, 0.40, 0.74, 0.81, 1.04, 0.55, 0.28, 0.23],
            "std": [0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.0, 0.01, 0.01],
        },
        "WER": {
            "mean": [0.66, 0.61, 0.55, 0.87, 0.98, 1.06, 0.63, 0.44, 0.40],
            "std": [0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.0, 0.01, 0.01],
        },
        "CED": {
            "mean": [669.74, 559.64, 498.36, 760.98, 769.36, 935.65, 526.79, 387.37, 336.53],
            "std": [10.38, 11.23, 11.01, 11.91, 10.51, 12.47, 0.0, 10.86, 10.18],
        },
        "WED": {
            "mean": [153.04, 136.97, 118.11, 186.90, 198.53, 204.98, 131.02, 92.78, 83.55],
            "std": [2.33, 2.68, 2.58, 2.85, 2.57, 2.79, 0.0, 2.46, 2.34],
        },
        "PLX": {
            "mean": [1.32, 1.49, 1.24, 1.24, 1.74, 1.40, 1.00, 0.0, 0.0],
            "std": [0.05, 0.10, 0.01, 0.03, 0.19, 0.00, 0.0, 0.00, 0.00],
        },
    },
    "VSEC": {
        "EM": {
            "mean": [0.33, 0.35, 0.22, 0.16, 0.12, 0.00, 0.08, 0.66, 0.75],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "CER": {
            "mean": [0.11, 0.02, 0.32, 0.03, 0.36, 8.00, 0.19, 0.01, 0.01],
            "std": [0.00, 0.00, 0.01, 0.00, 0.01, 0.07, 0.0, 0.00, 0.00],
        },
        "WER": {
            "mean": [0.13, 0.04, 0.33, 0.05, 0.39, 8.01, 0.28, 0.02, 0.01],
            "std": [0.00, 0.00, 0.01, 0.00, 0.01, 0.07, 0.0, 0.00, 0.00],
        },
        "CED": {
            "mean": [15.09, 2.81, 41.89, 3.38, 47.50, 1063.93, 25.13, 1.63, 0.89],
            "std": [0.42, 0.12, 1.54, 0.16, 0.86, 7.64, 0.0, 0.08, 0.04],
        },
        "WED": {
            "mean": [4.05, 1.18, 10.10, 1.51, 11.80, 241.74, 8.58, 0.61, 0.37],
            "std": [0.11, 0.03, 0.34, 0.04, 0.19, 1.74, 0.0, 0.02, 0.01],
        },
        "PLX": {
            "exclude": EXCLUDE_MODELS,
            "mean": [1.13, 1.15, 1.07, 1.01, 1.06, 1.46, 1.00, 0.0, 0.0],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
}

reasoning_task = {
    "SR - Natural": {
        "EM": {
            "mean": [0.14, 0.08, 0.04, 0.03, 0.00, 0.00, 0.07, 0.15, 0.37],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.48, 0.42, 0.38, 0.24, 0.01, 0.00, 0.41, 0.50, 0.74],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "Equivalent": {
            "mean": [0.15, 0.08, 0.04, 0.04, 0.00, 0.00, 0.07, 0.16, 0.42],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
    "SR - Abstract": {
        "EM": {
            "mean": [0.27, 0.20, 0.11, 0.19, 0.06, 0.14, 0.22, 0.26, 0.37],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.85, 0.70, 0.61, 0.69, 0.44, 0.71, 0.78, 0.83, 0.87],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "Equivalent": {
            "mean": [0.30, 0.17, 0.10, 0.18, 0.06, 0.10, 0.23, 0.29, 0.44],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
    "MATH": {
        "EM": {
            "mean": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "Equivalent": {
            "mean": [0.12, 0.00, 0.07, 0.16, 0.11, 0.01, 0.00, 0.62, 0.65],
            "std": [0.02, 0.01, 0.01, 0.02, 0.01, 0.00, 0.00, 0.02, 0.02],
        },
    },
}

ir_task = {
    "mMARCO": {
        "M@10": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.05, 0.04, 0.04, 0.07, 0.05, 0.00, 0.01],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "M@10B": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.11, 0.10, 0.11, 0.15, 0.11, 0.00, 0.07],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "N@10": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.06, 0.06, 0.06, 0.09, 0.07, 0.00, 0.04],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "N@10B": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.14, 0.14, 0.16, 0.21, 0.16, 0.00, 0.11],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
    },
    "mRobust04": {
        "M@10": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.04, 0.03, 0.03, 0.05, 0.02, 0.00, 0.04],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "M@10B": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.04, 0.05, 0.03, 0.04, 0.03, 0.00, 0.04],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "N@10": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.03, 0.04, 0.02, 0.04, 0.03, 0.00, 0.02],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "N@10B": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.04, 0.04, 0.02, 0.04, 0.02, 0.00, 0.02],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
    },
}

trans_task = {
    "PhoMT en→vi": {
        "BLEU": {
            "mean": [0.28, 0.25, 0.19, 0.23, 0.18, 0.15, 0.15, 0.33, 0.33],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "hLEPOR": {
            "mean": [0.59, 0.55, 0.50, 0.53, 0.47, 0.35, 0.51, 0.65, 0.66],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
    "PhoMT vi→en": {
        "BLEU": {
            "mean": [0.27, 0.15, 0.22, 0.23, 0.21, 0.03, 0.16, 0.33, 0.34],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "hLEPOR": {
            "mean": [0.58, 0.56, 0.54, 0.54, 0.52, 0.11, 0.52, 0.63, 0.65],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
    "OPUS100 en→vi": {
        "BLEU": {
            "mean": [0.10, 0.10, 0.08, 0.09, 0.07, 0.00, 0.07, 0.16, 0.17],
            "std": [0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.0, 0.01, 0.01],
        },
        "hLEPOR": {
            "mean": [0.44, 0.41, 0.38, 0.39, 0.34, 0.00, 0.37, 0.50, 0.51],
            "std": [0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.0, 0.01, 0.01],
        },
    },
    "OPUS100 vi→en": {
        "BLEU": {
            "mean": [0.14, 0.17, 0.14, 0.14, 0.11, 0.05, 0.09, 0.24, 0.25],
            "std": [0.00, 0.01, 0.01, 0.01, 0.01, 0.00, 0.0, 0.01, 0.01],
        },
        "hLEPOR": {
            "mean": [0.41, 0.43, 0.39, 0.40, 0.36, 0.16, 0.36, 0.51, 0.53],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.0, 0.00, 0.00],
        },
    },
}

drawing_tasks = {
    # "Question-Answering": qa_task, 
    # "Summarization": sum_task, 
    "Sentiment Analysis": sent_task, 
    "Text Classification": tcl_task, 
    "Knowledge": kn_task, 
    "Toxic Detection": td_task, 
    "Language Modeling": lm_task, 
    "Reasoning": reasoning_task,
    "Information Retrieval": ir_task,
    "Translation": trans_task
}

num_of_samples = {
    "XQuAD": 1190, 
    "MLQA": 5495, 
    "VietNews": 22498, 
    "WikiLingua": 27489, 
    "VLSP 2016": 1050, 
    "UiT-VSFC": 3166, 
    "UiT-VSMEC": 693, 
    "PhoATIS": 893,
    "ZaloE2E": 600, 
    "ViMMRC": 514, 
    "UiT-ViCTSD": 1000,
    "UiT-ViHSD": 6680,
    "MLQA-MLM": 5495,
    "VSEC": 9341, 
    "SR - Natural": 5000,
    "SR - Abstract": 15000, 
    "MATH": 5000, 
    "PhoMT en→vi": 19151,
    "PhoMT vi→en": 19151,
    "OPUS100 en→vi": 2000,
    "OPUS100 vi→en": 2000,
    "mMARCO": 6980,
    "mRobust04": 250,
}

selected_metrics = ["F1", "F1", "F1", "F1", "WER", "Equivalent", "N@10", "BLEU"]
plt.figure(figsize=(12, 7))
for idx, (task_name, task) in enumerate(drawing_tasks.items()):
    datasets = task.keys()
    metric = selected_metrics[idx]
    average_mean = None
    average_std = None
    tmp_model = None
    for dataset in datasets:
        try:
            exclude_flag = task[dataset][metric]["exclude"] if "exclude" in task[dataset][metric].keys() else []
            tmp_model = list(filter(lambda x: x not in exclude_flag, MODELS))
            mean = task[dataset][metric]["mean"][:len(tmp_model)]
            std = task[dataset][metric]["std"][:len(tmp_model)]
            # print("Yes")
            if average_mean is None:
                average_mean = np.array(mean)*num_of_samples[dataset]
                average_std = np.array(std)*num_of_samples[dataset]
            else:
                average_mean += np.array(mean)*num_of_samples[dataset]
                average_std += np.array(std)*num_of_samples[dataset]
    
            # plt.figure(figsize=(7, 10))  # Adjust figure size as needed
            # # print(avg_std_F1_qa)
            # # plt.figure(figsize=(6, 10))  # Adjust figure size as needed
            # # Create horizontal bar plot
            # y_pos = np.arange(len(tmp_model))#0069aa

            # plt.barh(y_pos, list(reversed(mean)), align='center', color=BAR_COLOR, ecolor=ECOLOR, xerr=list((reversed(std))), error_kw=dict(lw=2, capsize=3, capthick=1, color='#fff'),  height=BAR_WIDTH)
            # # plt.errorbar(y_pos, accuracies, xerr=std, lw=2, capsize=5, capthick=2, color='#fff')
            # plt.yticks(y_pos, reversed(tmp_model), fontsize=15)
            # plt.xticks(fontsize=15)
            # # plt.xlabel(task_name, fontsize=15)
            # plt.title(f"{dataset}\n{metric}", fontsize=15)

            # # Add grid and limit y-axis to 1.0
            # plt.grid(axis='x', linestyle='--', alpha=0.8)
            # plt.xlim(math.floor(min(0, min(mean))), math.ceil(max(mean)))

            # plt.tight_layout()  # Ensures labels are not cut off
            # plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}_{task_name}_{dataset}_{metric}.pdf")  # Save to a file (optional)
            # plt.close()
        except Exception as e:
            print(str(e))
            print(task_name)
            print(dataset)
            print(metric)
            exit(0)
    total_samples = sum([num_of_samples[i] for i in datasets])
    average_mean = average_mean/total_samples
    average_std = average_std/total_samples

    ax = plt.subplot(2, 4, idx+1)
    y_pos = np.arange(len(tmp_model))#0069aa

    plt.barh(y_pos, list(reversed(average_mean.tolist())), align='center', color=BAR_COLOR, ecolor=ECOLOR, xerr=list(reversed(average_std.tolist())), error_kw=dict(lw=2, capsize=3, capthick=1, color='#fff'), height=BAR_WIDTH)
    plt.yticks(y_pos, reversed(tmp_model), fontsize=10)
    plt.xticks(fontsize=10)
    plt.title(f'{task_name}\n{metric}', fontsize=12)
    max_number = max(average_mean)
    max_limit = 0
    if max_number < 0.05:
        max_limit = 0.05
    elif max_number < 0.1:
        max_limit = 0.1
    elif max_number < 0.15:
        max_limit = 0.15
    elif max_number < 0.25:
        max_limit = 0.25
    elif max_number < 0.5:
        max_limit = 0.5
    elif max_number < 0.75:
        max_limit = 0.75
    else:
        max_limit = math.ceil(max_number)

    major_ticks = np.arange(0, max_limit+0.1, round(max_limit/3, 2))
    # minor_ticks = np.arange(0, max_limit, max_limit/10)

    ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)

    ax.grid(which='both', linestyle='--', alpha=0.8)
    plt.xlim(0, max_limit)
# plt.show()

plt.tight_layout()  # Ensures labels are not cut off
plt.subplots_adjust(hspace=.4)
plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}.pdf")
plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}.png", dpi=500)  # Save to a file (optional)
# plt.show()