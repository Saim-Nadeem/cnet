import os
import subprocess
import shutil
import pandas as pd
from datetime import datetime

from src import extract_features
from src import train_models


def find_tshark():
    """
    Automatically locate tshark on Windows or Linux.
    """
    # Try PATH first
    if shutil.which("tshark"):
        return "tshark"

    # Try default Windows install locations
    common_paths = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe"
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Tshark not found. Install Wireshark and ensure Tshark is installed.\n"
        "If installed, add Wireshark to PATH or reinstall with Tshark enabled."
    )


def list_interfaces(tshark_path):
    """
    Run 'tshark -D' and return list of interfaces.
    """
    try:
        result = subprocess.run(
            [tshark_path, "-D"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split("\n")
        interfaces = [line.strip() for line in lines if line.strip()]
        return interfaces
    except Exception as e:
        print("ERROR listing interfaces:", e)
        return []


def align_live(df, feature_cols):
    """
    Align live-captured features with training features.
    """
    X = df.copy()

    cat = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat)
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0

    X = X[feature_cols]
    return X


def run_test_mode():
    print("\nRunning TEST mode (Iteration-4 training)...\n")
    results = train_models.train_iteration4_model()
    print("\nTraining complete.")
    print(results)


def run_live_mode():
    print("\nRunning LIVE mode using auto-detection...\n")

    # Load trained model
    model, feature_cols = train_models.load_iteration4_model()

    # Find tshark
    tshark_path = find_tshark()
    print(f"[+] Using Tshark: {tshark_path}")

    # List interfaces
    print("\n[+] Detecting network interfaces...")
    interfaces = list_interfaces(tshark_path)

    if not interfaces:
        print("ERROR: No interfaces found.")
        return

    print("\nAvailable Interfaces:")
    for iface in interfaces:
        print("  " + iface)

    # Ask user which interface NUMBER to use
    index = input("\nEnter interface NUMBER (e.g. 1, 2): ").strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap_path = f"captures/live_{ts}.pcap"
    csv_path = f"data/live_{ts}.csv"

    duration = input("Enter capture duration in seconds: ").strip()

    print("\n[+] Starting packet capture...\n")

    cmd = [
        tshark_path,
        "-i", index,
        "-a", f"duration:{duration}",
        "-w", pcap_path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

    print(f"\n[+] Capture saved: {pcap_path}")
    print("[+] Extracting flow features...")

    extract_features.parse_pcap(pcap_path, csv_path)

    df_live = pd.read_csv(csv_path)
    X = align_live(df_live, feature_cols)

    preds = model.predict(X)
    df_live["Prediction"] = preds
    df_live["PredictionLabel"] = df_live["Prediction"].map({0: "Normal", 1: "Attack"})

    print("\n=== LIVE TRAFFIC DETECTION RESULTS ===")
    print(df_live["PredictionLabel"].value_counts())

    print("\nLast 20 rows:")
    print(df_live.tail(20))


if __name__ == "__main__":
    print("Wired Sharks Iteration-4 Pipeline")
    print("Base paper had NO ML — manual inspection only.")
    print("This pipeline trains and uses the FIRST ML model enhancement.\n")

    mode = input("Enter mode (test/live): ").strip().lower()

    if mode == "test":
        run_test_mode()
    elif mode == "live":
        run_live_mode()
    else:
        print("Invalid mode. Choose test or live.")