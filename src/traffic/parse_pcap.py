
# A3 — Packet Reading
# Read packets sequentially from PCAP/PCAPNG files using Scapy.
# Avoid loading entire captures into memory.

from scapy.all import PcapReader


def iter_packets(path):
    with PcapReader(str(path)) as reader:
        for packet in reader:
            yield packet
