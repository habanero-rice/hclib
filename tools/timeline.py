import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class Task:
    def __init__(self, start, lbl, event_id):
        self.start = start
        self.elapsed = -1
        self.lbl = lbl
        self.event_id = event_id

    def set_elapsed(self, end_time):
        assert self.elapsed == -1
        self.elapsed = end_time - self.start

    def normalize_start(self, min_time):
        self.start = self.start - min_time


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

colors_iter = 0
colors = [('r', 'Red'),
          ('y', 'Yellow'),
          ('b', 'Blue'),
          ('g', 'Green'),
          ('c', 'Cyan'),
          ('m', 'Magenta'),
          ('#FA8072', 'Salmon'),
          ('#808000', 'Olive'),
          ('#FF00FF', 'Fuchsia')]
colors_dict = {}
for color in colors:
    colors_dict[color[0]] = color[1]

labels = {}

max_timestamp = 0
min_timestamp = None
if len(sys.argv) != 2:
    print('usage: python timeline.py timeline')
    sys.exit(1)

fp = open(sys.argv[1], 'r')

total_events = 0
tasks = {}

line_no = 1
for line in fp:
    tokens = line.split(' ')
    total_events += 1

    timestamp = int(tokens[0])
    thread = int(tokens[1])
    event_type = tokens[2]
    transition = tokens[3]
    event_id = int(tokens[4])

    if event_type not in labels:
        if colors_iter >= len(colors):
            print('Ran out of colors, add some')
            sys.exit(1)
        labels[event_type] = colors[colors_iter][0]
        colors_iter += 1

    if not thread in tasks:
        tasks[thread] = []

    if transition == 'START':
        tasks[thread].append(Task(timestamp, event_type, event_id))

        if min_timestamp is None:
            min_timestamp = timestamp
        else:
            min_timestamp = min(min_timestamp, timestamp)
    elif transition == 'END':
        found = None
        for task in tasks[thread]:
            if task.event_id == event_id:
                assert found is None
                found = task
        assert not found is None
        found.set_elapsed(timestamp)
        max_timestamp = max(max_timestamp, timestamp)
    else:
        print('Unsupported transition "' + transition + '" at line ' + str(line_no))
        sys.exit(1)

    line_no = line_no + 1

fig = plt.figure(num=0, figsize=(18, 6), dpi=80)

width = 0.35       # the width of the bars: can also be len(x) sequence
color_counter = 0
ind = 0

x_labels = []

print('Elapsed time: ' + str(float(max_timestamp - min_timestamp) / 1000000.0) + ' ms')
print(str(total_events) + ' events in total')
for lbl in labels:
    print(lbl + ': ' + colors_dict[labels[lbl]])

for thread in sorted(tasks.keys()):
    x_labels.append(str(thread))

    task_no = 1
    for t in tasks[thread]:
        t.normalize_start(min_timestamp)

        if task_no % 5000 == 0:
            print(str(thread) + ' ' + str(task_no) + '/' + str(len(tasks[thread])))

        # Plot in microseconds
        plt.barh(ind, float(t.elapsed) / 1000000.0, height=width,
                 left=(float(t.start) / 1000000.0), linewidth=1,
                 color=labels[t.lbl])
        task_no = task_no + 1

    ind = ind + width

plt.ylabel('Threads')
plt.xlabel('Time (ms)')
plt.yticks(np.arange(0, len(tasks.keys()), width) + width/2.,
           x_labels)
plt.axis([ 0, float(max_timestamp-min_timestamp) / 1000000.0, 0, ind ])
plt.show()

