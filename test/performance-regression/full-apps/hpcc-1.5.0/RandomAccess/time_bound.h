
#if defined( HPCC_RA_STDALG )
#if ! defined(RA_TIME_BOUND_DISABLE)
#define RA_TIME_BOUND 1
#endif
#endif

/* time bound in seconds */
#define TIME_BOUND 60


/* _RA_SAMPLE_FACTOR determines the fraction of the total number
 * of updates used (in time_bound.c) to empirically derive an
 * upper bound for the  number of updates executed by the benchmark.
 * This upper bound must be such that the total execution time of the
 * benchmark does not exceed a specified time bound.
 * _RA_SAMPLE_FACTOR may need to be adjusted for each architecture
 * since the dafault number of updates depends on the total
 * memory size.
 */
/* 1% of total number of updates */
#define RA_SAMPLE_FACTOR 100

extern void HPCC_Power2NodesTime(HPCC_RandomAccess_tabparams_t tparams, double timeBound, u64Int *numIter);

extern void HPCC_AnyNodesTime(HPCC_RandomAccess_tabparams_t tparams, double timeBound, u64Int *numIter);

extern void HPCC_Power2NodesTimeLCG(HPCC_RandomAccess_tabparams_t tparams, double timeBound, u64Int *numIter);

extern void HPCC_AnyNodesTimeLCG(HPCC_RandomAccess_tabparams_t tparams, double timeBound, u64Int *numIter);
