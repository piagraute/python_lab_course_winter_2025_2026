import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

# --- CONFIGURATION ---
OUT_DIR = "eval/plots"
# --------------------------

def load_tensorboard_data(log_dir, label, metric_name, smooth_weight=0.8):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if "scalars" not in event_acc.Tags() or metric_name not in event_acc.Tags()["scalars"]:
        print(f"Warnung: Metrik '{metric_name}' in '{log_dir}' nicht gefunden.")
        return pd.DataFrame()
    
    events = event_acc.Scalars(metric_name)
    data = []
    
    for event in events:
        data.append({
            "Augmentation-Typ": label,
            "Step": event.step,
            "Value": event.value,
            "Wall Time": event.wall_time
        })
    
    df = pd.DataFrame(data)

    if not df.empty and smooth_weight > 0:
        df["Value"] = df["Value"].ewm(alpha=(1 - smooth_weight), adjust=False).mean()

    return df

def setup_bw_style():
    """Setzt ein sauberes, schwarz-weißes Theme für Publikationen."""
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    rc = {
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
    }
    plt.rcParams.update(rc)

def plot_metric_over_time(log_paths, labels, metric_name, out_filename, title):
    setup_bw_style()
    
    all_dataframes = []
    
    for path, label in zip(log_paths, labels):
        df = load_tensorboard_data(path, label, metric_name)
        if not df.empty:
            all_dataframes.append(df)
            
    if not all_dataframes:
        print(f"Keine Daten für '{metric_name}' gefunden. Plot wird abgebrochen.")
        return
        
    full_df = pd.concat(all_dataframes, ignore_index=True)
    
    plt.figure(figsize=(8, 5))
    
    sns.lineplot(
        data=full_df, 
        x="Step", 
        y="Value", 
        hue="Augmentation-Typ", 
        palette="tab10",
        linewidth=2
    )
    if "Accuracy" in metric_name:
        plt.ylim(0.98, 1.0) 
        
        y_ticks = np.arange(0.98, 1.001, 0.005) 
        plt.yticks(y_ticks)
        
    elif "Loss" in metric_name:
        plt.ylim(0.0, 0.2) 
      
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Epochs")
    plt.title(title, fontweight="bold", pad=15)
    
    plt.tight_layout()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Plot erfolgreich gespeichert unter: {out_path}")
    plt.close() 

def plot(log_paths_list, labels, metric_names, base_title, prefix):
    for metric in metric_names:
       
        combined_title = f"{base_title} - {metric}"
        
        out_filename = f"{prefix}_{metric.replace('/', '_')}.png"
        
        plot_metric_over_time(
            log_paths=log_paths_list,
            labels=labels,
            metric_name=metric,
            out_filename=out_filename,
            title=combined_title
        )


def plot_training():

    # define paths
    cnn_paths = [
        "eval/cnn/cnn_baselineAug_20260311_1354",
        "eval/cnn/cnn_classicAug_20260311_1358",
        "eval/cnn/cnn_fixedadvancedAug_20260311_2341",
    ]
    
    mobilenet_paths = [
        "eval/mobilenet/mobilenetv2_baselineAug_20260311_1411",
        "eval/mobilenet/mobilenetv2_classicAug_20260311_1454",
        "eval/mobilenet/mobilenetv2_fixedadvancedAug_20260311_2346"
    ]

    model_configs = [
        (cnn_paths, "Pooja Hira's CNN", "cnn"),
        (mobilenet_paths, "Google's MobileNetV2", "mobilenet")
    ]
    
    
    labels = ["Baseline Augmentation", "Classic Augementation", "Advanced Augmentation"]
    
    metrics_to_plot = ["Accuracy/Validation", "Loss/Train", "Loss/Validation"] 
    for paths_list, title, prefix in model_configs:
        plot(
            log_paths_list=paths_list,
            labels=labels,
            metric_names=metrics_to_plot,
            base_title=title,
            prefix=prefix
        )
    
def plot_fine_tune():
  
    cnn_path = "eval/cnn/cnn_fixedadvancedAug_20260311_2341"
    mobilenet_path = "eval/mobilenet/mobilenetv2_fixedadvancedAug_20260311_2346"
    mobilenet_finetune_path = "eval/finetuned mobilenet/mobilenetv2_advancedAug_20260311_1614"

    all_paths = [cnn_path, mobilenet_path, mobilenet_finetune_path]

    model_labels = [
        "Pooja Hira's CNN", 
        "Google's MobileNetV2", 
        "Finetuned MobileNetV2"
    ]

    metrics_to_plot = ["Accuracy/Validation", "Loss/Validation"] 

    for metric in metrics_to_plot:
        
        title = f"Model Comparison: {metric}"
        safe_prefix = metric.replace("/", "_").lower() 
        
        plot(
            log_paths_list=all_paths, 
            labels=model_labels,         
            metric_names=[metric],      
            base_title=title,
            prefix=safe_prefix
        )


if __name__ == "__main__":
    plot_fine_tune()