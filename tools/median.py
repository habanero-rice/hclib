import os
import sys

def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]

l = []
for line in sys.stdin:
    l.append(float(line))

print str(median(l))
