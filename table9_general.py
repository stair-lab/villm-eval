# Import
import matplotlib.pyplot as plt
from tueplots import bundles
import tueplots
# plt.rcParams.update(bundles.iclr2024())
import numpy as np
import math
import os

# Constant
TABLE_NAME = "generaltable9_weakprompt"
ECOLOR ='orange'
BAR_COLOR = tueplots.constants.color.rgb.tue_blue
BAR_WIDTH = 0.93

# Data
MODELS = ['Our 70B', 'Our 13B', 'Our 7B', 'Llama-2 13B', 'Llama-2 7B', 'MixSUra 7x8B']
# MODELS.remove('Gemini Pro')
EXCLUDE_MODELS = ['Gemini Pro','GPT-3.5-turbo', 'GPT-4']

if not os.path.exists(TABLE_NAME):
    os.mkdir(TABLE_NAME)

qa_task = {
    "XQuAD": {
        "EM": {
            "mean": [0.21, 0.22, 0.13, 0.04, 0.06, 0.13],
            "std": [0.01, 0.01, 0.00, 0.00, 0.00, 0.0],
        },
        "F1": {
            "mean": [0.47, 0.43, 0.32, 0.28, 0.24, 0.38],
            "std": [0.01, 0.01, 0.00, 0.00, 0.00, 0.0],
        },
    },
    "MLQA": {
        "EM": {
            "mean": [0.14, 0.17, 0.10, 0.04, 0.05, 0.09],
            "std": [0.01, 0.01, 0.00, 0.00, 0.00, 0.0],
        },
        "F1": {
            "mean": [0.41, 0.40, 0.32, 0.28, 0.24, 0.36],
            "std": [0.00, 0.01, 0.00, 0.00, 0.00, 0.0],
        },
    },
}

sum_task = {
    "VietNews": {
        "R1": {
            "mean": [0.49, 0.27, 0.45, 0.45, 0.36, 0.44],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "R2": {
            "mean": [0.23, 0.12, 0.21, 0.22, 0.17, 0.22],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "RL": {
            "mean": [0.31, 0.18, 0.29, 0.29, 0.23, 0.29],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "SC": {
            "mean": [-0.08, -0.09, -0.08, -0.09, -0.09, 0.0],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "BS": {
            "mean": [0.05, 0.05, 0.03, 0.00, -0.15, 0.07],
            "std": [0.11, 0.11, 0.09, 0.14, 0.12, 0.0],
        },
        "Cv": {
            "mean": [0.89, 0.56, 0.91, 0.92, 0.69, 0.97],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "De": {
            "mean": [8.90, 5.00, 9.43, 9.49, 6.35, 35.67],
            "std": [0.03, 0.04, 0.03, 0.02, 0.03, 0.0],
        },
        "Cp": {
            "mean": [18.48, 153.55, 6.42, 8.46, 7.59, 9.43],
            "std": [0.59, 0.99, 0.05, 0.29, 0.21, 0.0],
        },
    },
    "WikiLingua": {
        "R1": {
            "mean": [0.47, 0.22, 0.42, 0.47, 0.45, 0.47],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "R2": {
            "mean": [0.20, 0.09, 0.18, 0.22, 0.20, 0.22],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "RL": {
            "mean": [0.29, 0.14, 0.27, 0.29, 0.27, 0.29],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "SC": {
            "mean": [-0.16, -0.16, -0.16, -0.16, -0.16, 0.0],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "BS": {
            "mean": [0.19, 0.20, 0.07, 0.34, 0.36, 0.19],
            "std": [0.13, 0.007, 0.12, 0.12, 0.00, 0.0],
        },
        "Cv": {
            "mean": [0.86, 0.48, 0.89, 0.92, 0.83, 0.97],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "De": {
            "mean": [6.83, 3.49, 7.58, 9.39, 7.71, 28.97],
            "std": [0.09, 0.04, 0.05, 0.05, 0.07, 0.0],
        },
        "Cp": {
            "mean": [25.30, 190.09, 7.14, 17.94, 12.39, 10.27],
            "std": [1.86, 4.92, 0.14, 2.84, 1.46, 0.0],
        },
    },
}

qa2_task = {
    "XQuAD": {
        "EM": {
            "mean": [0.08, 0.04, 0.01, 0.00, 0.00, 0.01],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "F1": {
            "mean": [0.33, 0.21, 0.11, 0.10, 0.03, 0.25],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
    },
    "MLQA": {
        "EM": {
            "mean": [0.07, 0.04, 0.01, 0.00, 0.00, 0.00],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "F1": {
            "mean": [0.31, 0.19, 0.11, 0.09, 0.03, 0.25],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
    },
}

sum2_task = {
    "VietNews": {
        "R1": {
            "mean": [0.35, 0.26, 0.41, 0.02, 0.03, 0.06],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "R2": {
            "mean": [0.16, 0.12, 0.18, 0.00, 0.01, 0.01],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "RL": {
            "mean": [0.24, 0.17, 0.27, 0.02, 0.03, 0.04],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "SC": {
            "mean": [-0.11, -0.09, -0.09, -0.09, -0.09, 0.0],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "BS": {
            "mean": [0.12, -0.08, -0.08, -0.19, -0.17, -0.13],
            "std": [0.00, 0.18, 0.13, 0.05, 0.03, 0.0],
        },
        "Cv": {
            "mean": [0.63, 0.46, 0.83, 0.01, 0.04, 0.10],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "De": {
            "mean": [5.43, 3.55, 8.13, 0.01, 0.07, 0.17],
            "std": [0.02, 0.04, 0.04, 0.00, 0.00, 0.0],
        },
        "Cp": {
            "mean": [37.78, 47.75, 8.08, 54.67, 23.86, 9.03],
            "std": [0.47, 0.65, 0.17, 0.16, 0.26, 0.0],
        },
    },
    "WikiLingua": {
        "R1": {
            "mean": [0.33, 0.14, 0.42, 0.03, 0.02, 0.03],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "R2": {
            "mean": [0.14, 0.05, 0.17, 0.00, 0.00, 0.00],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "RL": {
            "mean": [0.22, 0.09, 0.27, 0.03, 0.02, 0.03],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "SC": {
            "mean": [-0.16, -0.16, -0.16, -0.16, -0.16, 0.0],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        },
        "BS": {
            "mean": [0.24, -0.14, 0.27, -0.05, -0.04, -0.01],
            "std": [0.10, 0.12, 0.21, 0.03, 0.06, 0.0],
        },
        "Cv": {
            "mean": [0.59, 0.26, 0.84, 0.02, 0.02, 0.17],
            "std": [0.01, 0.01, 0.00, 0.00, 0.00, 0.0],
        },
        "De": {
            "mean": [4.62, 1.83, 7.15, 0.02, 0.03, 0.26],
            "std": [0.11, 0.06, 0.08, 0.00, 0.00, 0.0],
        },
        "Cp": {
            "mean": [56.56, 60.10, 8.08, 42.55, 40.31, 16.68],
            "std": [1.70, 2.16, 0.36, 0.81, 0.88, 0.0],
        },
    },
}



drawing_tasks = {
    "Question-Answering\nWeak Prompting": qa_task, 
    "Summarization\nWeak Prompting": sum_task,
    "Question-Answering\nMedium Prompting": qa2_task, 
    "Summarization\nMedium Prompting": sum2_task,  
    # "Sentiment Analysis": sent_task, 
    # "Text Classification": tcl_task, 
    # "Knowledge": kn_task, 
    # "Toxic Detection": td_task, 
    # "Language Modeling": lm_task, 
    # "Reasoning": reasoning_task,
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

selected_metrics = ["F1", "R1", "F1", "R1"]
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

    major_ticks = np.arange(0, max_limit+0.1, max_limit/3)
    minor_ticks = np.arange(0, max_limit, max_limit/3)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.grid(which='both', linestyle='--', alpha=0.8)
    plt.xlim(0, max_limit)
# plt.show()

plt.tight_layout()  # Ensures labels are not cut off
plt.subplots_adjust(hspace=.4)
tmpname = task_name.replace("\n"," ")
plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}_{tmpname}.pdf")
plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}_{tmpname}.png", dpi=500)  # Save to a file (optional)
# plt.show()