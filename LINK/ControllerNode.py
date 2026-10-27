import os
import sys
import time
import select
import socket
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.stdout.reconfigure(line_buffering=True)

from LOGIC.ControllerCore import ControllerCore
from LINK.Protocol import (KIND_HELLO, KIND_STATE, KIND_END, READY_PKT, CTRL_PORT,
                           unpack_hello, unpack_state, pack_command)

MAX_PKT = 4096


def recv_wait(sock, timeout_s=1.0):
    # wait up to timeout_s for one packet; (None, None) if nothing came
    ready, _, _ = select.select([sock], [], [], timeout_s)
    if not ready:
        return None, None
    try:
        return sock.recvfrom(MAX_PKT)
    except BlockingIOError:
        return None, None


def recv_now(sock):
    # one packet that is already waiting, or (None, None); never waits
    try:
        return sock.recvfrom(MAX_PKT)
    except BlockingIOError:
        return None, None


def serve_one_run(sock):
    # ---- Startup: wait for the sim's robot constants, build the controller, answer READY ----
    while True:
        pkt, sim_addr = recv_wait(sock)
        if pkt is not None and pkt[:1] == KIND_HELLO:
            break
    cfg = unpack_hello(pkt)
    print(f"[ctrl] hello from {sim_addr[0]}, building controller ...", flush=True)
    core = ControllerCore(cfg)
    sock.sendto(READY_PKT, sim_addr)
    print("[ctrl] ready, running", flush=True)

    log = []    # per command: state seq, sim time, compute time (s), states skipped
    while True:
        pkt, addr = recv_wait(sock)
        if pkt is None:
            continue
        kind = pkt[:1]
        if kind == KIND_HELLO:              # sim resent hello while the controller was building
            sock.sendto(READY_PKT, addr)
            continue
        if kind == KIND_END:
            break
        if kind != KIND_STATE:
            continue

        # skip to the newest state already waiting; older states are stale
        skipped, ended = 0, False
        while True:
            newer, newer_addr = recv_now(sock)
            if newer is None:
                break
            if newer[:1] == KIND_STATE:
                pkt, addr, skipped = newer, newer_addr, skipped + 1
            elif newer[:1] == KIND_END:
                ended = True
                break
        if ended:
            break

        seq, t, t_wall, st = unpack_state(pkt)
        t0 = time.perf_counter()
        tau, diag, state_name = core.step(t, st)
        compute_s = time.perf_counter() - t0
        sock.sendto(pack_command(seq, t_wall, compute_s, state_name, tau, diag), addr)
        log.append((seq, t, compute_s, skipped))

    core.summary()
    return np.array(log, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Controller side of the sim link: robot state in, joint torques out.")
    parser.add_argument("--port", type=int, default=CTRL_PORT)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)
    print(f"[ctrl] listening on UDP port {args.port} (Ctrl+C to quit)", flush=True)

    run_index = 0
    try:
        while True:
            log = serve_one_run(sock)
            run_index += 1
            if len(log):
                compute_ms = 1e3 * log[:, 2]
                print(f"\n===== CONTROLLER RUN {run_index} =====")
                print(f"commands sent      : {len(log)}")
                print(f"compute time       : mean {np.mean(compute_ms):.2f}   p99 {np.percentile(compute_ms, 99):.2f}   "
                      f"max {np.max(compute_ms):.2f} ms")
                print(f"states skipped     : mean {np.mean(log[:, 3]):.2f} per command", flush=True)
                path = f"ctrl_timing_run{run_index}.csv"
                np.savetxt(path, log, delimiter=",", header="state_seq,sim_t,compute_s,skipped", comments="")
                print(f"[log] {path} saved, waiting for the next run", flush=True)
    except KeyboardInterrupt:
        print("\n[ctrl] stopped", flush=True)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
