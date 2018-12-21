---
layout: default
title: Documentation
permalink: /documentation/
---

You can find the source code for HCLib at:
[HCLib](https://github.com/habanero-rice/hclib)

A Wiki dedicated to the Habanero project and its many subprojects (including HCLib) is located [here](https://wiki.rice.edu/confluence/display/HABANERO/Habanero+Extreme+Scale+Software+Research+Project).

While there is no "official" HCLib manual yet, there are several papers that describe and discuss different aspects of Habanero-C in genereal, and HCLib in particular:

Publications
===================================

1. [A Pluggable Framework for Composable HPC Scheduling Libraries](https://www.cs.rice.edu/~zoran/Publications_files/hiper.pdf) Max Grossman, Vivek Kumar, Nick Vrvilo, Zoran Budimlić, and Vivek Sarkar. The Seventh International Workshop on Accelerators and Hybrid Exascale Systems (AsHES), May 2017.  
   This paper describes the resource workers runtime implemented in HCLib, that allows us to have a communication layer built on top of HCLib to support a distributed HCLib on top of, for example, OpenSHMEM or MPI.
   
2. [Integrating Asynchronous Task Parallelism with OpenSHMEM](https://www.cs.rice.edu/~zoran/Publications_files/asyncshmem2016.pdf). Max Grossman, Vivek Kumar, Zoran Budimlić and Vivek Sarkar. OpenSHMEM 2016: Third workshop on OpenSHMEM and Related Technologies, August 2016, Baltimore, Maryland.  
   A paper on integration of HCLib runtime within a node and OpenSHMEM as the communication layer across a distributed machine nodes.
   
3. [HabaneroUPC++: a Compiler-free PGAS Library](https://www.cs.rice.edu/~zoran/Publications_files/habaneroupc-pgas14.pdf). Vivek Kumar, Yili Zheng, Vincent Cavé, Zoran Budimlić and Vivek Sarkar. In Proceedings of the 8th International Conference on Partitioned Global Address Space Programming Models (PGAS14), October 2014.  
   Integration of HCLib and UPC++ that uses the C++11 lambdas to to simplify the syntax of Habanero constructs within HCLib.
   
4. [Integrating Asynchronous Task Parallelism with MPI](https://www.cs.rice.edu/~zoran/Publications_files/IPDPS13.pdf). Sanjay Chatterjee, Sağnak Taşırlar, Zoran Budimlić, Vincent Cavé, Millind Chabbi, Max Grossman, Yonghong Yan and Vivek Sarkar. 27th IEEE International Parallel & Distributed Processing Symposium (IPDPS 2013), May 2013, Boston, MA.  
   The first first paper on extending the Habanero constructs beyond the intra-node parallelism, using MPI for communication.
   
5. [Comparing the Usability of Library vs. Language Approaches to Task Parallelism](https://www.cs.rice.edu/~zoran/Publications_files/PLATEAU10.pdf). Vincent Cavé, Zoran Budimlić and Vivek Sarkar. PLATEAU 2010 : Second Workshop on Evaluation and Usability of Programming Languages and Tools (held in conjunction with SPLASH), October 2010, Reno, NV.  
   Comparing the trade-offs between using a language approach (that requires a compiler) and a library approach to Habanero task parallelism.
 
6. [A Scalable Locality-aware Adaptive Work-stealing Scheduler for Multi-core Task Parallelism](https://www.cs.rice.edu/~vs3/PDF/Guo-thesis-2010.pdf). Yi Guo. Ph.D. Thesis, August 2010.  
   Everything you ever wanted to know about using work-stealing for load balancing, which is implemented in HCLib as well.
   
7. [Hierarchical Place Trees: A Portable Abstraction for Task Parallelism and Date Movement](https://www.cs.rice.edu/~vs3/PDF/hpt.pdf). Yonghong Yan, Jisheng Zhao, Yi Guo, Vivek Sarkar. Proceedings of the 22nd Workshop on Languages and Compilers for Parallel Computing (LCPC), October 2009.  
   Hierarchical Place Trees is an abstraction that HCLib uses to control _where_ the tasks should execute on a shared memory machine, in order to take advantage of memory reuse.

Tutorials
===================================

[HClib Tutorial Slides](../HClibTutoria.pdf)


