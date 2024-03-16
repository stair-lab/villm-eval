# Import
import matplotlib.pyplot as plt
from tueplots import bundles
import tueplots
# plt.rcParams.update(bundles.iclr2024())
import numpy as np
import math
import os

# Constant
TABLE_NAME = "generaltable10_typo"
ECOLOR ='orange'
BAR_COLOR = tueplots.constants.color.rgb.tue_blue
BAR_WIDTH = 0.93

# Data
MODELS = ['Our 70B', 'Our 13B', 'Our 7B', 'Llama-2 13B', 'Llama-2 7B', 'Vietcuna 7B', 'MixSUra 7x8B', 'Gemini Pro', 'GPT-3.5-turbo', 'GPT-4']
MODELS.remove('Gemini Pro')
EXCLUDE_MODELS = ['Gemini Pro','GPT-3.5-turbo', 'GPT-4']

if not os.path.exists(TABLE_NAME):
    os.mkdir(TABLE_NAME)

qa_task = {
    "XQuAD": {
        "EM": {
            "mean": [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.17, 0.09, 0.09, 0.02, 0.02, 0.06, 0.11, 0.19, 0.24],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
    "MLQA": {
        "EM": {
            "mean": [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.18, 0.10, 0.10, 0.03, 0.02, 0.05, 0.12, 0.20, 0.25],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
}

sum_task = {
    "VietNews": {
        "R1": {
            "mean": [0.34, 0.35, 0.37, 0.05, 0.05, 0.03, 0.41, 0.34, 0.39],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "R2": {
            "mean": [0.15, 0.14, 0.12, 0.01, 0.01, 0.01, 0.19, 0.19, 0.21],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "RL": {
            "mean": [0.23, 0.23, 0.24, 0.04, 0.05, 0.02, 0.26, 0.23, 0.26],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "SC": {
            "mean": [-0.06, -0.09, -0.10, -0.15, -0.10, -0.10, 0, -0.10, -0.10],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.09],
        },
        "BS": {
            "mean": [-0.11, -0.07, -0.24, -0.24, -0.19, -0.18, -0.03, 0.05, 0.04],
            "std": [0.18, 0.17, 0.18, 0.18, 0.04, 0.06, 0, 0.14, 0.00],
        },
        "Cv": {
            "mean": [0.10, 0.64, 0.65, 0.03, 0.07, 0.91, 0.86, 0.81, 0.83],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "De": {
            "mean": [0.10, 0.65, 0.65, 0.03, 0.07, 0.91, 0.87, 0.81, 0.83],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "Cp": {
            "mean": [39.63, 134.65, 17.92, 55.91, 55.29, 1026.61, 29.15, 128.44, 24.48],
            "std": [0.87, 3.76, 0.87, 0.65, 0.88, 3.86, 0, 2.94, 0.00],
        },
    },
    "WikiLingua": {
        "R1": {
            "mean": [0.28, 0.20, 0.37, 0.04, 0.04, 0.08, 0.46, 0.39, 0.45],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "R2": {
            "mean": [0.11, 0.07, 0.12, 0.00, 0.00, 0.02, 0.21, 0.19, 0.20],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "RL": {
            "mean": [0.19, 0.13, 0.24, 0.03, 0.04, 0.05, 0.28, 0.25, 0.27],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "SC": {
            "mean": [-0.16, -0.17, -0.17, -0.17, -0.17, -0.17, 0, -0.17, -0.17],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "BS": {
            "mean": [0.25, 0.20, 0.11, 0.09, 0.15, -0.19, 0.26, 0.28, 0.28],
            "std": [0.23, 0.11, 0.18, 0.00, 0.00, 0.05, 0, 0.11, 0.00],
        },
        "Cv": {
            "mean": [0.50, 0.38, 0.65, 0.05, 0.06, 0.78, 0.88, 0.82, 0.80],
            "std": [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.03],
        },
        "De": {
            "mean": [0.50, 0.38, 0.65, 0.05, 0.06, 0.78, 0.98, 0.82, 0.81],
            "std": [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00],
        },
        "Cp": {
            "mean": [167.42, 103.69, 20.49, 66.85, 58.32, 505.45, 19.10, 200.90, 20.40],
            "std": [7.09, 3.33, 0.95, 6.72, 3.32, 8.64, 0, 7.40, 1.59],
        },
    },
}

sent_task = {
    "VLSP 2016": {
        "AC": {
            "mean": [0.63, 0.55, 0.52, 0.46, 0.45, 0.44, 0.59, 0.64, 0.74],
            "std": [0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.0, 0.01, 0.00],
        },
        "F1": {
            "mean": [0.48, 0.52, 0.36, 0.30, 0.36, 0.27, 0.59, 0.60, 0.73],
            "std": [0.01, 0.02, 0.03, 0.01, 0.01, 0.01, 0.0, 0.01, 0.00],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.60, 0.59, 0.59, 0.55, 0.54, 0.51, 0.55, 0.0, 0.0],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.00, 0.00],
        },
        "ECE": {
            "mean": [0.09, 0.06, 0.07, 0.39, 0.20, 0.23, 0.34, 0.31, 0.41],
            "std": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.0, 0.01, 0.00],
        },
        "A@10": {
            "mean": [0.83, 0.74, 0.66, 0.70, 0.51, 0.53, 0.52, 0.54, 0.71],
            "std": [0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.05, 0.00],
        },
    },
    "UiT-VSFC": {
        "AC": {
            "mean": [0.71, 0.72, 0.73, 0.66, 0.51, 0.49, 0.69, 0.86, 0.83],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.01, 0.00],
        },
        "F1": {
            "mean": [0.45, 0.44, 0.41, 0.40, 0.33, 0.25, 0.44, 0.71, 0.70],
            "std": [0.01, 0.05, 0.01, 0.01, 0.01, 0.03, 0.0, 0.01, 0.00],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.80, 0.77, 0.71, 0.63, 0.65, 0.46, 0.61, 0.0, 0.0],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.00, 0.00],
        },
        "ECE": {
            "mean": [0.08, 0.18, 0.16, 0.11, 0.15, 0.33, 0.29, 0.53, 0.50],
            "std": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.01, 0.00],
        },
        "A@10": {
            "mean": [0.99, 0.77, 0.87, 0.89, 0.80, 0.34, 0.66, 0.86, 0.85],
            "std": [0.01, 0.02, 0.02, 0.02, 0.02, 0.03, 0.0, 0.02, 0.00],
        },
    },
}

tcl_task = {
    "UiT-VSMEC": {
        "AC": {
            "mean": [0.25, 0.30, 0.29, 0.19, 0.17, 0.09, 0.35, 0.42, 0.48],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.16, 0.11, 0.10, 0.07, 0.10, 0.09, 0.27, 0.41, 0.45],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.56, 0.51, 0.57, 0.52, 0.55, 0.51, 0.70, 0.0, 0.0],
            "std": [0.02, 0.01, 0.01, 0.01, 0.00, 0.01, 0.0, 0.00, 0.00],
        },
        "ECE": {
            "mean": [0.20, 0.26, 0.17, 0.47, 0.33, 0.91, 0.58, 0.28, 0.33],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "A@10": {
            "mean": [0.33, 0.44, 0.30, 0.43, 0.29, 0.09, 0.70, 0.30, 0.40],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
    "PhoATIS": {
        "AC": {
            "mean": [0.16, 0.01, 0.02, 0.02, 0.01, 0.02, 0.80, 0.68, 0.86],
            "std": [0.02, 0.01, 0.01, 0.00, 0.01, 0.01, 0.0, 0.02, 0.01],
        },
        "F1": {
            "mean": [0.26, 0.05, 0.04, 0.06, 0.00, 0.01, 0.55, 0.64, 0.80],
            "std": [0.03, 0.01, 0.00, 0.00, 0.00, 0.00, 0.0, 0.03, 0.02],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.79, 0.47, 0.55, 0.57, 0.56, 0.55, 0.94, 0.0, 0.0],
            "std": [0.00, 0.01, 0.01, 0.01, 0.00, 0.01, 0.0, 0.00, 0.00],
        },
        "ECE": {
            "mean": [0.79, 0.84, 0.18, 0.91, 0.69, 0.23, 0.15, 0.62, 0.80],
            "std": [0.02, 0.01, 0.01, 0.00, 0.01, 0.01, 0.0, 0.02, 0.01],
        },
        "A@10": {
            "mean": [0.08, 0.00, 0.01, 0.01, 0.02, 0.02, 0.88, 0.70, 0.91],
            "std": [0.06, 0.04, 0.02, 0.00, 0.02, 0.01, 0.0, 0.05, 0.03],
        },
    },
}

kn_task = {
    "ZaloE2E": {
        "EM": {
            "mean": [0.23, 0.18, 0.10, 0.13, 0.02, 0.05, 0.13, 0.45, 0.44],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.01, 0.01],
        },
        "F1": {
            "mean": [0.37, 0.30, 0.18, 0.21, 0.05, 0.15, 0.24, 0.61, 0.61],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.01, 0.01],
        },
    },
    "ViMMRC": {
        "AC": {
            "mean": [0.65, 0.41, 0.33, 0.39, 0.26, 0.26, 0.57, 0.90, 0.91],
            "std": [0.00, 0.00, 0.02, 0.00, 0.01, 0.01, 0.0, 0.01, 0.01],
        },
        "F1": {
            "mean": [0.53, 0.34, 0.28, 0.31, 0.20, 0.14, 0.45, 0.72, 0.73],
            "std": [0.00, 0.00, 0.02, 0.00, 0.01, 0.00, 0.0, 0.04, 0.07],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.84, 0.61, 0.61, 0.56, 0.51, 0.50, 0.53, 0.0, 0.0],
            "std": [0.00, 0.00, 0.01, 0.00, 0.01, 0.00, 0.0, 0.00, 0.00],
        },
        "ECE": {
            "mean": [0.11, 0.22, 0.19, 0.46, 0.46, 0.01, 0.35, 0.65, 0.66],
            "std": [0.00, 0.00, 0.02, 0.00, 0.01, 0.01, 0.0, 0.01, 0.07],
        },
        "A@10": {
            "mean": [0.77, 0.58, 0.33, 0.33, 0.13, 0.21, 0.58, 0.88, 0.88],
            "std": [0.00, 0.00, 0.06, 0.00, 0.03, 0.07, 0.0, 0.07, 0.04],
        },
    },
}

td_task = {
    "UiT-ViCTSD": {
        "AC": {
            "mean": [0.32, 0.27, 0.22, 0.12, 0.04, 0.11, 0.72, 0.51, 0.88],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.21, 0.26, 0.21, 0.11, 0.04, 0.11, 0.72, 0.46, 0.71],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.72, 0.56, 0.63, 0.56, 0.62, 0.54, 0.5, 0.5, 0.0, 0.0],
            "std": [0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "ECE": {
            "mean": [0.62, 0.56, 0.39, 0.66, 0.86, 0.39, 0.62, 0.01, 0.38],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "A@10": {
            "mean": [0.33, 0.12, 0.36, 0.12, 0.02, 0.13, 0.33, 0.54, 0.88],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
    "UiT-ViHSD": {
        "AC": {
            "mean": [0.14, 0.18, 0.12, 0.10, 0.01, 0.09, 0.64, 0.64, 0.78],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.12, 0.11, 0.07, 0.07, 0.00, 0.05, 0.64, 0.47, 0.56],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00],
        },
        "AR": {
            "exclude": EXCLUDE_MODELS,
            "mean": [0.64, 0.57, 0.62, 0.59, 0.54, 0.5, 0.57, 0.0, 0.0],
            "std": [0.02, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "ECE": {
            "mean": [0.61, 0.45, 0.38, 0.62, 0.79, 0.24, 0.61, 0.30, 0.44],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "A@10": {
            "mean": [0.23, 0.20, 0.19, 0.24, 0.00, 0.08, 0.23, 0.63, 0.78],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
    },
}


trans_task = {
    "PhoMT en→vi": {
        "BLEU": {
            "mean": [0.25, 0.23, 0.15, 0.20, 0.13, 0.17, 0.14, 0.31, 0.31],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
        "hLEPOR": {
            "mean": [0.58, 0.55, 0.48, 0.51, 0.41, 0.43, 0.50, 0.64, 0.65],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
    "PhoMT vi→en": {
        "BLEU": {
            "mean": [0.11, 0.10, 0.06, 0.07, 0.05, 0.07, 0.11, 0.17, 0.20],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.0, 0.00, 0.00],
        },
        "hLEPOR": {
            "mean": [0.51, 0.50, 0.46, 0.44, 0.42, 0.41, 0.46, 0.59, 0.62],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
    "OPUS100 en→vi": {
        "BLEU": {
            "mean": [0.05, 0.03, 0.02, 0.03, 0.02, 0.09, 0.06, 0.15, 0.16],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.0, 0.01, 0.01],
        },
        "hLEPOR": {
            "mean": [0.40, 0.38, 0.35, 0.36, 0.31, 0.38, 0.36, 0.49, 0.50],
            "std": [0.01, 0.01, 0.00, 0.01, 0.00, 0.01, 0.0, 0.01, 0.01],
        },
    },
    "OPUS100 vi→en": {
        "BLEU": {
            "mean": [0.06, 0.05, 0.03, 0.04, 0.03, 0.09, 0.06, 0.21, 0.23],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.0, 0.01, 0.01],
        },
        "hLEPOR": {
            "mean": [0.36, 0.38, 0.34, 0.32, 0.30, 0.33, 0.31, 0.48, 0.51],
            "std": [0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
        },
    },
}

drawing_tasks = {
    "Question-Answering": qa_task, 
    "Summarization": sum_task, 
    "Sentiment Analysis": sent_task, 
    "Text Classification": tcl_task, 
    "Knowledge": kn_task, 
    "Toxic Detection": td_task, 
    # "Language Modeling": lm_task, 
    # "Reasoning": reasoning_task,
    # "Information Retrieval": ir_task,
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

selected_metrics = ["F1", "R1", "F1", "F1", "F1", "F1", "BLEU"]
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
    # minor_ticks = np.arange(0, max_limit, max_limit/3)

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