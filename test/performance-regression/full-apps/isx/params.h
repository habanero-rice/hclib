/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
    * Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef _PARAMS_H
#define _PARAMS_H

//defining mayor and minor version number of ISx code
#define MAJOR_VERSION_NUMBER 1
#define MINOR_VERSION_NUMBER 1

//OpenSHMEM compliance: shmalloc and shfree are deprecated
//to use shmem_malloc and shmem_free define OPENSHMEM_COMPLIANT
//undef for Cray, since Cray SHMEM is currently not OpenSHMEM complient
#define OPENSHMEM_COMPLIANT

// The data type used for the keys
// If you change this, you will have to change the SHMEM API calls used
typedef int KEY_TYPE;

// STRONG SCALING: Total number of keys are fixed and the number of keys per PE are reduced with increasing number of PEs
//  Invariants: Total number of keys, max key value
//  Variable:   Number of keys per PE, Bucket width

// WEAK SCALING: THe number of keys per PE is fixed and the total number of keys grow with increasing number of PEs
//  Invariants: Number of keys per PE, max key value
//  Variable:   Total Number of Keys, Bucket width 

// WEAK_ISOBUCKET: Same as WEAK except the maximum key value grows with the number of PEs to keep bucket width constant
//  Invariants: Number of keys per PE, bucket width
//  Variable:   Total number of keys, max key value

//  ***!!! WARNING !!!***
#define STRONG 1
#define WEAK 2
#define WEAK_ISOBUCKET 3
// ***!!! If you change any of the above values, you must change their corresponding
// values in the Makefile !!!***

#define ISO_BUCKET_WIDTH (8192u)

// Specifies the default maximum key value used in creation of the input
// For STRONG and WEAK scaling options, keys will be generated in 
// the range [0, DEFAULT_MAX_KEY]. WEAK_ISOBUCKET varies the MAX_KEY_VAL 
// to keep the BUCKET_WIDTH constant per PE.
#ifdef DEBUG
#define DEFAULT_MAX_KEY (32uLL)
#else
#define DEFAULT_MAX_KEY (unsigned long long)(1uLL<<28uLL)
#endif

// The number of iterations that an integer sort is performed
// (Burn in iterations are done first and are not timed)
#define NUM_ITERATIONS (1u)
#define BURN_IN (1u)

#define BARRIER_ATA


// Specifies if the all2all uses a per PE randomized target list
//#define PERMUTE

#define PRINT_MAX 64

#endif
