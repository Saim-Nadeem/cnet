import os
import argparse
import pyshark
import pandas as pd


def parse_pcap(pcap_path: str, out_path: str) -> None:
    """Parse a pcap file using PyShark and export basic features to CSV.

    Extracted features:
    - timestamp
    - src_ip, dst_ip
    - protocol
    - length
    - src_port, dst_port
    - tcp flags (if present)
    """
    print(f"[*] Reading pcap: {pcap_path}")
    cap = pyshark.FileCapture(pcap_path, keep_packets=False)

    rows = []
    for pkt in cap:
        try:
            ts = float(pkt.sniff_timestamp)
            length = int(pkt.length)

            # IP layer (may not exist for all packets)
            if hasattr(pkt, "ip"):
                src_ip = getattr(pkt.ip, "src", None)
                dst_ip = getattr(pkt.ip, "dst", None)
            else:
                src_ip = None
                dst_ip = None

            protocol = pkt.highest_layer.lower()

            src_port = None
            dst_port = None
            flags = None

            if hasattr(pkt, "tcp"):
                src_port = getattr(pkt.tcp, "srcport", None)
                dst_port = getattr(pkt.tcp, "dstport", None)
                flags = getattr(pkt.tcp, "flags", None)
            elif hasattr(pkt, "udp"):
                src_port = getattr(pkt.udp, "srcport", None)
                dst_port = getattr(pkt.udp, "dstport", None)

            rows.append(
                {
                    "timestamp": ts,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": protocol,
                    "length": length,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "flags": flags,
                }
            )
        except Exception:
            # Ignore malformed packets
            continue

    cap.close()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[*] Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, help="Input pcap file")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    parse_pcap(args.pcap, args.out)
