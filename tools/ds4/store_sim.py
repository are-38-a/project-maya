"""tools/ds4/store_sim.py - which RAM store policy misses least, from a DS4_ROUTE_TRACE file (no GPU).

    python tools/ds4/store_sim.py --trace FILE --pack PACK --gb 50 [--profile FILE]

Replays the CPU lane's picks (the ones VRAM does not hold) against the store's slots - a size of slots a layer's
matrices, as many a layer - filled at the start with the profile's hottest experts, and prints each policy's misses a
token and bytes read a token.  belady (the expert asked for furthest ahead gives way) is the bound no policy can beat.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import struct
from collections import OrderedDict, defaultdict
from heapq import heapify, heappop, heappush

ALIGN = 4096


def room(b: int) -> int:
    return (b + 2 * ALIGN - 1) // ALIGN * ALIGN


def read_trace(path: str):
    """the picks VRAM does not hold, in order: [(layer, expert)]; tokens; the VRAM's experts at the start."""
    data = pathlib.Path(path).read_bytes()
    p, picks, vram, tokens = 0, [], set(), 0
    while p < len(data):
        tag, a, b = struct.unpack_from("<3i", data, p)
        if tag == ord("P"):
            ids = struct.unpack_from("<%di" % b, data, p + 12)
            vram.update((a, e) for e in ids)
            p += 12 + 4 * b
        elif tag == ord("R"):
            k = struct.unpack_from("<i", data, p + 12)[0]
            n = b * k
            ids = struct.unpack_from("<%di" % n, data, p + 16)
            flags = data[p + 16 + 4 * n: p + 16 + 5 * n]
            picks.extend((a, e) for e, f in zip(ids, flags) if e >= 0 and not f)
            tokens += b if a == 0 else 0
            p += 16 + 5 * n
        else:
            raise SystemExit("%s: bad record at %d" % (path, p))
    return picks, tokens, vram


def layer_rooms(pack: pathlib.Path):
    """[layer]: the three matrices' sizes of its expert 0."""
    sizes = defaultdict(dict)
    pat = re.compile(r"^t layers\.(\d+)\.ffn\.experts\.0\.(\w+)\.weight \S+ \S+ \S+ (\d+)")
    for line in (pack / "tensors.txt").read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            sizes[int(m.group(1))][m.group(2)] = int(m.group(3))
    return {l: tuple(room(b) for _, b in sorted(v.items())) for l, v in sizes.items()}


class Sim:
    """one size of slots: capacity experts, filled with `start` (hottest first)"""

    def __init__(self, cap: int, start, policy: str, prot: float):
        self.cap, self.policy = cap, policy
        self.miss = 0
        self.a = OrderedDict()                     # lru / slru's probation: oldest first
        self.b = OrderedDict()                     # slru's protected
        self.pcap = int(cap * prot)
        self.heat = {}
        for x in reversed(list(start)[:cap]):      # (the hottest last: the newest)
            self.a[x] = True
            self.heat[x] = 1.0

    def ask(self, x) -> None:
        if self.policy == "lru":
            if x in self.a:
                self.a.move_to_end(x)
                return
            self.miss += 1
            if len(self.a) >= self.cap:
                self.a.popitem(last=False)
            self.a[x] = True
        elif self.policy == "slru":
            if x in self.b:
                self.b.move_to_end(x)
                return
            if x in self.a:                        # (asked for again: it earned protection)
                del self.a[x]
                self.b[x] = True
                if len(self.b) > self.pcap:
                    y, _ = self.b.popitem(last=False)
                    self.a[y] = True
                return
            self.miss += 1
            if len(self.a) + len(self.b) >= self.cap:
                (self.a if self.a else self.b).popitem(last=False)
            self.a[x] = True


def belady(seq, cap: int, start) -> int:
    """the misses of the policy that knows the future: seq [(key)] of one size"""
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], 1 << 60)
        last[seq[i]] = i
    first = dict(last)
    held = {}
    heap = []
    for x in list(start)[:cap]:
        held[x] = first.get(x, 1 << 60)
        heap.append((-held[x], x))
    heapify(heap)
    miss = 0
    for i, x in enumerate(seq):
        if x in held:
            held[x] = nxt[i]
            heappush(heap, (-nxt[i], x))
            continue
        miss += 1
        while len(held) >= cap:
            d, y = heappop(heap)
            if held.get(y) == -d:
                del held[y]
                break
        held[x] = nxt[i]
        heappush(heap, (-nxt[i], x))
    return miss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gb", type=float, default=50.0)
    ap.add_argument("--profile")
    a = ap.parse_args()
    pack = pathlib.Path(a.pack)
    picks, tokens, vram = read_trace(a.trace)
    rooms = layer_rooms(pack)
    n_exp = 384
    prof = pathlib.Path(a.profile) if a.profile else pack / "expert-profile.bin"
    raw = prof.read_bytes()
    counts = struct.unpack("<%dI" % (len(raw) // 4), raw)
    per_layer = len(counts) // len(rooms)
    n_exp = per_layer
    classes = sorted(set(rooms.values()))
    layer_cls = {l: classes.index(r) for l, r in rooms.items()}
    k = int(a.gb * 1e9) // sum(sum(r) for r in rooms.values())
    caps = [k * sum(1 for l in rooms if layer_cls[l] == c) for c in range(len(classes))]
    hot = sorted(((counts[l * n_exp + e], l, e) for l in rooms for e in range(n_exp) if (l, e) not in vram), reverse=True)
    start = [[(l, e) for _, l, e in hot if layer_cls[l] == c] for c in range(len(classes))]
    bytes_of = {l: sum(rooms[l]) for l in rooms}
    print("%d tokens, %d picks off VRAM (%.1f a token), %d sizes, %d slots" % (tokens, len(picks), len(picks) / max(1, tokens), len(classes), sum(caps)))
    seqs = [[x for x in picks if layer_cls[x[0]] == c] for c in range(len(classes))]
    res = []
    for name, pol, prot in (("lru", "lru", 0.0), ("slru 50", "slru", 0.5), ("slru 80", "slru", 0.8), ("slru 90", "slru", 0.9)):
        miss = mb = 0.0
        for c, seq in enumerate(seqs):
            s = Sim(caps[c], start[c], pol, prot)
            for x in seq:
                m0 = s.miss
                s.ask(x)
                if s.miss != m0:
                    mb += bytes_of[x[0]] / 1e6
            miss += s.miss
        res.append((name, miss, mb))
    b = sum(belady(seq, caps[c], start[c]) for c, seq in enumerate(seqs))
    res.append(("belady", b, 0.0))
    for name, miss, mb in res:
        print("%-8s %.2f misses a token, %.1f MB a token" % (name, miss / max(1, tokens), mb / max(1, tokens)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
