import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

# --- CONFIGURATION ---
PATH_CNN = "eval/cnn"             
PATH_MOBILENET = "eval/mobilenet"

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
        # TensorBoard nutzt die Formel: smoothed = last_smoothed * weight + (1 - weight) * current
        # In Pandas entspricht das der ewm (Exponential Weighted Math) Funktion.
        # Ein smooth_weight von 0.8 bedeutet starke Glättung, 0.0 bedeutet keine Glättung.
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
        # 1. Wir schneiden den unteren Teil (z.B. die ersten schlechten Epochen) ab
        plt.ylim(0.98, 1.0) 
        
        # 2. Wir zwingen Matplotlib, feine, normale Dezimalschritte zu machen
        # np.arange(Start, Ende, Schrittweite)
        y_ticks = np.arange(0.98, 1.001, 0.005) 
        plt.yticks(y_ticks)
        
    elif "Loss" in metric_name:
        # Beim Loss willst du vielleicht eher den unteren Bereich (0 bis 0.1) genau sehen
        plt.ylim(0.0, 0.2) 
      
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Epochs")
    plt.title(title, fontweight="bold", pad=15)
    
    plt.tight_layout()
    
    # Speichern
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Plot erfolgreich gespeichert unter: {out_path}")
    plt.close() # Verhindert, dass zu viele Plots im Speicher bleiben

def plot(log_paths_list, labels, metric_names, base_title, prefix):
    # log_paths_list ist hier nun korrekt eine Liste mit 3 Pfaden!
    for metric in metric_names:
        # Kombiniere Titel und Metrik (z.B. "Pooja Hira's CNN - Loss/Train")
        combined_title = f"{base_title} - {metric}"
        
        # Erstelle einen Dateinamen, der das Modell enthält (z.B. cnn_Loss_Train.png)
        # Dadurch überschreiben sich CNN und MobileNet nicht mehr!
        out_filename = f"{prefix}_{metric.replace('/', '_')}.png"
        
        plot_metric_over_time(
            log_paths=log_paths_list,
            labels=labels,
            metric_name=metric,
            out_filename=out_filename,
            title=combined_title
        )
    
if __name__ == "__main__":
    
    # 1. Definiere die Pfade zu deinen TensorBoard Logs
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
    
    # 2. Definiere die Namen, die in der Legende stehen sollen
    
    labels = ["Baseline Augmentation", "Classic Augementation", "Advanced Augmentation"]
    
    # 3. Definiere, welche Metrik verglichen werden soll.
    # WICHTIG: Der String muss exakt so heißen wie in TensorBoard (z.B. "Loss/train" oder "accuracy")
    metrics_to_plot = ["Accuracy/Validation", "Loss/Train", "Loss/Validation"] # Hier anpassen!
    for paths_list, title, prefix in model_configs:
        plot(
            log_paths_list=paths_list,
            labels=labels,
            metric_names=metrics_to_plot,
            base_title=title,
            prefix=prefix
        )
    
    