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

def plot_bar_chart(log_paths, labels, metric_name, title, display_name, filename_prefix):
    """Erstellt ein Balkendiagramm für die übergebenen Modelle und speichert es ab."""
    setup_bw_style()
    
    data = []
    for path, label in zip(log_paths, labels):
        result = load_tensorboard_data(path, label, metric_name)
        
        if result is not None:
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    data.append({"Model": label, "Value": result["Value"].iloc[-1]})
            else:
                data.append(result)
            
    df = pd.DataFrame(data)
    if df.empty:
        print(f"Keine Daten für {metric_name} gefunden. Überspringe Plot.")
        return

    plt.figure(figsize=(9, 6))
    
    ax = sns.barplot(
        data=df, 
        x="Model", 
        y="Value", 
        hue="Model", 
        palette="gray", 
        legend=False
    )
    
    plt.title(title, pad=20, fontweight="bold")
    plt.xlabel("") 
    plt.ylabel(display_name) 
    
    for i, v in enumerate(df["Value"]):
        ax.text(i, v + (df["Value"].max() * 0.02), f"{v:.4f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    plt.ylim(0, df["Value"].max() * 1.15)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    
    filepath = os.path.join(OUT_DIR, f"{filename_prefix}_{display_name.lower()}.png")
    plt.savefig(filepath, dpi=300)
    print(f"Plot gespeichert unter: {filepath}")
    
    plt.show()
    plt.close()

def main():
    # define paths
    cnn_path = "runs/cnn_advancedAug_20260313_0937"
    mobilenet_path = "runs/mobilenetv2_advancedAug_20260313_0938"
    mobilenet_finetune_path = "runs/mobilenetv2_advancedAug_20260313_0941"

    paths = [cnn_path, mobilenet_path, mobilenet_finetune_path]
    labels = ["Pooja Hira's CNN", "MobileNetV2", "Finetuned MobileNetV2"]
    
    metrics_mapping = {
        "Accuracy/Validation": "Accuracy",
        "Loss/Validation": "Loss"
    }

    for tb_metric, display_name in metrics_mapping.items():
        plot_bar_chart(
            log_paths=paths,
            labels=labels,
            metric_name=tb_metric,          
            display_name=display_name,      
            title=f"Evaluation: {display_name}",
            filename_prefix="bar_chart"
        )

if __name__ == "__main__":
    main()