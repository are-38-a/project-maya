"""tools/ds4/vram_sim.py - how many picks each VRAM expert policy serves, from a DS4_ROUTE_TRACE file (no GPU).

    python tools/ds4/vram_sim.py --trace FILE [--swaps-a-token N]

Each layer keeps the slots it had at the start.  `actual` is what the run's flags say.  The policies move at most N
experts into VRAM a token (a move crosses PCIe), starting from the VRAM the trace starts with.  belady has no such limit:
the bound no policy can beat.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
from collections import defaultdict
from heapq import heapify, heappop, heappush


def read_trace(path: str):
    data = pathlib.Path(path).read_bytes()
    p, seq, vram, in_vram = 0, [], defaultdict(set), 0
    while p < len(data):
        tag, a, b = struct.unpack_from("<3i", data, p)
        if tag == ord("P"):
            vram[a].update(struct.unpack_from("<%di" % b, data, p + 12))
            p += 12 + 4 * b
        else:
            k = struct.unpack_from("<i", data, p + 12)[0]
            n = b * k
            ids = struct.unpack_from("<%di" % n, data, p + 16)
            flags = data[p + 16 + 4 * n: p + 16 + 5 * n]
            for e, f in zip(ids, flags):
                if e >= 0:
                    seq.append((a, e))
                    in_vram += 1 if f else 0
            p += 16 + 5 * n
    return seq, vram, in_vram


def heat_policy(seq, vram, margin, bonus, decay, every, swaps):
    """the engine's rule: the hottest miss in when its heat passes the coldest's x margin + bonus"""
    held = {l: set(v) for l, v in vram.items()}
    heat = defaultdict(float)
    hits = tokens = budget = 0
    for i, (l, e) in enumerate(seq):
        if l == 0 and i % 5 == 0:
            tokens += 1
            budget = min(budget + swaps, 40)
            if tokens % every == 0:
                for k in heat:
                    heat[k] *= decay
        heat[(l, e)] += 1.0
        h = held.setdefault(l, set())
        if e in h:
            hits += 1
            continue
        if budget >= 1 and h:
            cold = min(h, key=lambda x: heat[(l, x)])
            if heat[(l, e)] > heat[(l, cold)] * margin + bonus:
                h.remove(cold)
                h.add(e)
                budget -= 1
    return hits


def lru_policy(seq, vram, swaps):
    held = {l: {e: 0 for e in v} for l, v in vram.items()}
    hits = tokens = 0
    budget = 0.0
    for i, (l, e) in enumerate(seq, 1):
        if l == 0 and i % 5 == 0:
            tokens += 1
            budget = min(budget + swaps, 40)
        h = held.setdefault(l, {})
        if e in h:
            hits += 1
            h[e] = i
            continue
        if budget >= 1 and h:
            del h[min(h, key=h.get)]
            h[e] = i
            budget -= 1
    return hits


def belady(seq, vram):
    hits = 0
    by = defaultdict(list)
    for l, e in seq:
        by[l].append(e)
    for l, s in by.items():
        cap = len(vram.get(l, ()))
        if cap == 0:
            continue
        nxt = [0] * len(s)
        last = {}
        for i in range(len(s) - 1, -1, -1):
            nxt[i] = last.get(s[i], 1 << 60)
            last[s[i]] = i
        held = {e: last.get(e, 1 << 60) for e in vram[l]}
        heap = [(-v, e) for e, v in held.items()]
        heapify(heap)
        for i, e in enumerate(s):
            if e in held:
                hits += 1
            elif len(held) >= cap:
                while True:
                    d, y = heappop(heap)
                    if held.get(y) == -d:
                        del held[y]
                        break
            held[e] = nxt[i]
            heappush(heap, (-nxt[i], e))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--swaps-a-token", type=float, default=2.0)
    a = ap.parse_args()
    seq, vram, in_vram = read_trace(a.trace)
    n = len(seq)
    print("%d picks, %d VRAM slots" % (n, sum(len(v) for v in vram.values())))
    print("actual            %.1f%%" % (100 * in_vram / n))
    for name, m, b, d in (("heat 1.25 d0.9", 1.25, 1.0, 0.9), ("heat 1.25 d0.6", 1.25, 1.0, 0.6), ("heat 1.25 d0.3", 1.25, 1.0, 0.3),
                          ("heat 1.1 d0.6", 1.1, 1.0, 0.6)):
        print("%-17s %.1f%%" % (name, 100 * heat_policy(seq, vram, m, b, d, 256, a.swaps_a_token) / n))
    print("%-17s %.1f%%" % ("lru", 100 * lru_policy(seq, vram, a.swaps_a_token) / n))
    print("%-17s %.1f%%" % ("belady", 100 * belady(seq, vram) / n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
