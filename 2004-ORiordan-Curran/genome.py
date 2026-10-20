"""
Labels:  0-3 inputs | 4-11 hidden | 12 output | 13-15 reserved for markers.
Node record: NODE | label | threshold | CONN | (source, weight) pairs | END, 
    with every field 4 bits wide.
    Markers: NODE=1110, CONN=1101, END=1111.

Safe decoding: the decoder scans for NODE, then reads fields at fixed widths. 
    Connections from equal or higher labels are dropped, and for a repeated label the first record wins.

Genome length: fixed at 480 bits
Feed-forward only: a node accepts connections from lower labels only.



ASSUMPTION:
 A random genome defines an output node about 63% of the time. 
 When it doesnt, we can treat the output as a constant 0.5 in the network builder, 
 so those agents just score badly.
"""

import random



N_IN, MAX_HID = 4, 8                        # no. of input nodes, cap on hidden nodes (4-11)
FIRST_HID, OUT = 4, 12                      # label of first hidden node, label of output node.
NODE, CONN, END = "1110", "1101", "1111"    
GENOME_LEN = 480                           

def _val(bits):
    """4-bit field -> value in [-4, 3.5]."""
    return (int(bits, 2) - 8) / 2

def random_genome(length=GENOME_LEN):
    return "".join(random.choice("01") for _ in range(length))

def decode(genome):
    """Return {label: (threshold, [(src_label, weight), ...])} for hidden/output nodes.
    label: node id (4-11 for hidden nodes, 12 for output. 0-3 input bits
    threshold: node's bias 
    list holds node's incoming connections, each a (source label, weight) pair.
    Connections from equal or higher labels are dropped, and for a repeated label the first record wins.
    """
    nodes = {}
    p, n = 0, len(genome)
    while p + 4 <= n:
        if genome[p:p + 4] != NODE:
            p += 1
            continue
        p += 4
        if p + 8 > n:
            break
        label = int(genome[p:p + 4], 2)
        thr = _val(genome[p + 4:p + 8])
        p += 8
        conns = []
        if genome[p:p + 4] == CONN:
            p += 4
            while p + 8 <= n and genome[p:p + 4] != END:
                src = int(genome[p:p + 4], 2)
                w = _val(genome[p + 4:p + 8])
                p += 8
                if src < label:  # feed-forward rule
                    conns.append((src, w))
            if genome[p:p + 4] == END:
                p += 4
        if FIRST_HID <= label <= OUT and label not in nodes:  # first record wins
            nodes[label] = (thr, conns)
    return nodes


def _bits(v):
    return format(int(round(v * 2 + 8)), "04b")
def encode_node(label, thr, conns):
    s = NODE + format(label, "04b") + _bits(thr) + CONN
    for src, w in conns:
        s += format(src, "04b") + _bits(w)
    return s + END

# if __name__ == "__main__":
#     g = encode_node(4, 0.5, [(0, 1.0), (1, -2.0)]) + encode_node(12, -1.0, [(4, 3.0), (2, 0.5)])
#     print(decode(g))
#     # expected: {4: (0.5, [(0, 1.0), (1, -2.0)]), 12: (-1.0, [(4, 3.0), (2, 0.5)])}
#     nodes = decode(random_genome())
#     print(len(nodes), "nodes decoded from a random genome; output defined:", OUT in nodes)