#include "hclib_mpi-internal.h"

#include <iostream>

static int nic_locale_id;

static void copy_func(hclib::locale_t *dst_locale, void *dst,
        hclib::locale_t *src_locale, void *src, size_t nbytes) {
    //TODO
}

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
}

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_post_initialize) {
    hclib_register_copy_func(nic_locale_id, copy_func, MUST_USE);
}

static hclib::locale_t *get_locale_for_rank(int rank, MPI_Comm comm) {
    char name_buf[256];
    sprintf(name_buf, "mpi-rank-%d", rank);

    hclib::locale_t *new_locale = (hclib::locale_t *)malloc(
            sizeof(hclib::locale_t));
    new_locale->id = -1 * rank;
    new_locale->type = nic_locale_id;
    new_locale->lbl = (char *)malloc(strlen(name_buf) + 1);
    memcpy(new_locale->lbl, name_buf, strlen(name_buf) + 1);
    new_locale->metadata = malloc(sizeof(MPI_Comm));
    *((MPI_Comm *)new_locale->metadata) = comm;
    new_locale->deques = NULL;
    return new_locale;
}

hclib::locale_t *hclib::MPI_Comm_rank(MPI_Comm comm) {
    int rank;
    CHECK_MPI(MPI_Comm_rank(comm, &rank));

    return get_locale_for_rank(rank, comm);
}

void hclib::MPI_Comm_size(MPI_Comm comm, int *size) {
    CHECK_MPI(MPI_Comm_size(comm, size));
}

hclib::locale_t *hclib::MPI_Comm_remote(MPI_Comm comm, int remote_rank) {
    return get_locale_for_rank(remote_rank, comm);
}

HCLIB_REGISTER_MODULE("mpi", mpi_pre_initialize, mpi_post_initialize)
