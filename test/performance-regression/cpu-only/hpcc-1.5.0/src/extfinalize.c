/*
 This routine is called right before MPI_Finalize() and allows finalization
 of external software and hardware components. It can be replaced
 at the time of installation. A sample implemenation may finialize
 proprietary computational and communication libraries.
 The parameter "extdata" points to an object of size of a pointer.
 "extdata" comes from HPCC_external_init().
 Upon success, the function should return 0.
 */
int
HPCC_external_finalize(int argc, char *argv[], void *extdata) {
  return 0;
}
