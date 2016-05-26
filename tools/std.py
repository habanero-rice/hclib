#!/usr/bin/python

import numpy
import sys

l = []
for line in sys.stdin:
    l.append(float(line))

print(numpy.std(l))
