/*
 This routine is called right after MPI_Init() and allows initialization
 of external software and hardware components. It can be replaced
 at the time of installation. A sample implemenation may initialize
 proprietary computational and communication libraries.
 The parameter "extdata" points to an object of size of a pointer.
 The function may choose to store a pointer to its internal data
 and it will be passed to the finalization routine HPCC_external_finalize().
 Upon success, the function should return 0.
 */
int
HPCC_external_init(int argc, char *argv[], void *extdata) {
  return 0;
}
