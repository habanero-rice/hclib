---
layout: home
---

HClib is a task-based parallel programming model that supports the finish-async,
parallel-for, and future-promise parallel programming patterns through both C
and C++ APIs. HClib explicitly
exposes hardware locality of the hardware, while allowing the programmer to fall
back on sane defaults. The HClib runtime is a lightweight, work-stealing, and
locality-aware runtime. HClib is not itself an exascale programming system, but
is intended to be the intra-node resource management and scheduling component
within an exascale programming system, integrating with inter-node communication
models such as MPI, UPC++, or OpenSHMEM.
