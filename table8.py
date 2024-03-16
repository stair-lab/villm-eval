# Import
import matplotlib.pyplot as plt
from tueplots import bundles
import tueplots
plt.rcParams.update(bundles.iclr2024())
import numpy as np
import math
import os

# Constant
TABLE_NAME = "table8_mathcot"
ECOLOR ='orange'
BAR_COLOR = tueplots.constants.color.rgb.tue_blue
BAR_WIDTH = 0.93

# Data
MODELS = ['Our 70B', 'Our 13B', 'Our 7B', 'Llama-2 13B', 'Llama-2 7B', 'Vietcuna 7B', 'MixSUra 7x8B', 'Gemini Pro', 'GPT-3.5-turbo', 'GPT-4']
MODELS.remove('Gemini Pro')
EXCLUDE_MODELS = ['Gemini Pro','GPT-3.5-turbo', 'GPT-4']

if not os.path.exists(TABLE_NAME):
    os.mkdir(TABLE_NAME)

reasoning_task = {
    "MATH with COT": {
        "EM": {
            "mean": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "std": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        "F1": {
            "mean": [0.12, 0.23, 0.23, 0.12, 0.10, 0.13, 0.17, 0.32, 0.32],
            "std": [0.01, 0.01, 0.01, 0.01, 0.00, 0.01, 0.01, 0.01, 0.01],
        },
        "Equivalent": {
            "mean": [0.18, 0.17, 0.09, 0.18, 0.12, 0.10, 0.33, 0.78, 0.79],
            "std": [0.02, 0.01, 0.01, 0.02, 0.02, 0.01, 0.00, 0.02, 0.02],
        },
    },
}


drawing_tasks = {
    
    # "Sentiment Analysis": sent_task, 
    # "Text Classification": tcl_task, 
    # "Knowledge": kn_task, 
    # "Toxic Detection": td_task, 
    # "Language Modeling": lm_task, 
    "Reasoning": reasoning_task,
}
for task_name, task in drawing_tasks.items():
    datasets = task.keys()
    for dataset in datasets:
        metrics = task[dataset].keys()
        for metric in metrics:
            try:
                exclude_flag = task[dataset][metric]["exclude"] if "exclude" in task[dataset][metric].keys() else []
                tmp_model = list(filter(lambda x: x not in exclude_flag, MODELS))
                mean = task[dataset][metric]["mean"][:len(tmp_model)]
                std = task[dataset][metric]["std"][:len(tmp_model)]

            
                plt.figure(figsize=(7, 10))  # Adjust figure size as needed
                # print(avg_std_F1_qa)
                # plt.figure(figsize=(6, 10))  # Adjust figure size as needed
                # Create horizontal bar plot
                y_pos = np.arange(len(tmp_model))#0069aa

                plt.barh(y_pos, list(reversed(mean)), align='center', color=BAR_COLOR, ecolor=ECOLOR, xerr=list((reversed(std))), error_kw=dict(lw=2, capsize=3, capthick=1, color='#fff'),  height=BAR_WIDTH)
                # plt.errorbar(y_pos, accuracies, xerr=std, lw=2, capsize=5, capthick=2, color='#fff')
                plt.yticks(y_pos, reversed(tmp_model), fontsize=15)
                plt.xticks(fontsize=15)
                # plt.xlabel(task_name, fontsize=15)
                plt.title(f"{dataset}\n{metric}", fontsize=15)

                # Add grid and limit y-axis to 1.0
                plt.grid(axis='x', linestyle='--', alpha=0.8)
                plt.xlim(math.floor(min(0, min(mean))), math.ceil(max(mean)))

                plt.tight_layout()  # Ensures labels are not cut off
                plt.savefig(f"{TABLE_NAME}/{TABLE_NAME}_{task_name}_{dataset}_{metric}.pdf")  # Save to a file (optional)
                plt.close()
            except Exception as e:
                print(str(e))
                print(task_name)
                print(dataset)
                print(metric)
                exit(0)
# plt.show()