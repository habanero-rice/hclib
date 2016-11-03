#!/usr/bin/python

import os
import sys

pes_per_node = int(sys.argv[1])
threads_per_pe = 16 / pes_per_node

pes = []
for pe in range(pes_per_node):
    start_thread = pe * threads_per_pe
    end_thread = (pe + 1) * threads_per_pe
    cc = str(start_thread)
    for t in range(start_thread + 1, end_thread):
        cc = cc + ',' + str(t)
    # print(cc)
    pes.append(cc)
print(':'.join(pes))
