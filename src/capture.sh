#!/bin/bash
# Simple wrapper to capture traffic with tshark.
# Usage: ./capture.sh <interface> <duration_seconds>
IFACE=$1
DURATION=$2
OUTFILE="../captures/capture_$(date +%Y%m%d_%H%M%S).pcap"
echo "[*] Capturing on $IFACE for $DURATION seconds..."
tshark -i "$IFACE" -a duration:"$DURATION" -w "$OUTFILE"
echo "[*] Capture complete: $OUTFILE"
