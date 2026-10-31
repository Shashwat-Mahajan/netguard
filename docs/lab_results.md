# Real-attack lab (Phase 6, part 1)

Two Ubuntu Server VMs (VMware Workstation) on an isolated host-only network
(VMnet1, 192.168.29.0/24). Attacker 192.168.29.131, victim 192.168.29.129
(nginx on 80, SSH on 22). The Windows host captures VMnet1 with Scapy/Npcap.
All attacks ran only between these two VMs, which I own.

## Method

1. Capture on the host: `sniff(iface=VMnet1)` -> `data/lab/*.pcap`.
2. Run the attack from the attacker VM inside the capture window.
3. Replay each pcap through NetGuard (ML + signature rules), Suricata with
   local rules and Suricata with ET Open (`scripts/lab/compare_lab.py`).

## Results (one run each)

| Attack | ML (XGBoost + IForest) | NetGuard rules | Suricata local | Suricata ET Open |
|---|---|---|---|---|
| nmap -sS, 1000 ports (1,051 flows) | Benign x1051, 0 threats | port_scan | alerts | no alert |
| hping3 flood :80 (7,641 flows) | Benign x7641, 0 threats | syn_flood (+4.4 s pcap time) | alerts | no alert |
| nmap -sn sweep (54 flows) | Benign | no alert | no alert | no alert |
| hping3 flood :22 (17,024 flows) | Benign x17024, 0 threats | syn_flood + port_scan | alerts | no alert |

Rules-only replay of further captures (ML not run on these):

| Capture | Result |
|---|---|
| hping3 flood :8080, closed port (4,370 flows) | syn_flood on :8080 only |
| nmap -sT, 1000 ports (1,006 flows) | port_scan only, no syn_flood |

## Findings

1. The CICIDS-trained ML model labelled every real attack flow Benign with
   confidence ~1.0 and the Isolation Forest flagged none (domain gap).
2. The original syn_flood rule only counted unanswered SYNs, so it missed a
   real flood against an open port (the victim's SYN-ACK breaks the condition).
   Fixed: count half-open flows to one destination port. Tested on floods to
   ports 80, 22 (open) and 8080 (closed).
3. Flows whose first captured packet was the server's reply were reported with
   the server as source and produced a false port_scan. Fixed with a heuristic
   (source port < 1024 and destination port >= 1024 is ignored by port_scan).
4. Suricata local rules were written for this setup, so they are not
   independent evidence. ET Open raised nothing on these plain nmap/hping3
   captures; this does not show ET Open cannot detect scans.
5. TheHive accepted the alerts (201) after its key was fixed; the earlier 400
   was never diagnosed.

## Limitations

- One attacker, one victim, one network, one run per attack; no rates.
- Rule fixes were derived from these captures; the flood variants are the only
  out-of-sample check.
- "+N s" latencies are pcap time from the attacker's first packet, not live
  detection latency. The scan's +31.6 s is not a meaningful figure.
- The port-direction heuristic would miss scanners using low source ports.
- The extra port_scan in the flood :22 capture is unexplained unless an nmap
  scan ran during that capture (unconfirmed).
- The host sweep had only 3 hosts, so no detector could be expected to fire.