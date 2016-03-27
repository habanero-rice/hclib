#include "hclib_mpi-internal.h"

#include <iostream>

static int nic_locale_id;

HCLIB_MODULE_INITIALIZATION_FUNC(mpi_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
}

HCLIB_MODULE_INITIALIZATION_FUNC(mpi_post_initialize) {
    CHECK_MPI(MPI_Init(NULL, NULL));
}

HCLIB_MODULE_INITIALIZATION_FUNC(mpi_finalize) {
    MPI_Finalize();
}

static hclib::locale_t *get_locale_for_rank(int rank, MPI_Comm comm) {
    char name_buf[256];
    sprintf(name_buf, "mpi-rank-%d", rank);

    hclib::locale_t *new_locale = (hclib::locale_t *)malloc(
            sizeof(hclib::locale_t));
    /*
     * make the rank negative, and then subtract one so that even rank 0 has a
     * negative rank.
     */
    new_locale->id = -1 * rank - 1; 
    new_locale->type = nic_locale_id;
    new_locale->lbl = (char *)malloc(strlen(name_buf) + 1);
    memcpy((void *)new_locale->lbl, name_buf, strlen(name_buf) + 1);
    new_locale->metadata = malloc(sizeof(MPI_Comm));
    *((MPI_Comm *)new_locale->metadata) = comm;
    new_locale->deques = NULL;
    return new_locale;
}

hclib::locale_t *hclib::MPI_Comm_rank(MPI_Comm comm) {
    int rank;
    CHECK_MPI(::MPI_Comm_rank(comm, &rank));

    return get_locale_for_rank(rank, comm);
}

void hclib::MPI_Comm_size(MPI_Comm comm, int *size) {
    CHECK_MPI(::MPI_Comm_size(comm, size));
}

hclib::locale_t *hclib::MPI_Comm_remote(MPI_Comm comm, int remote_rank) {
    return get_locale_for_rank(remote_rank, comm);
}

int hclib::integer_rank_for_locale(locale_t *locale) {
    assert(locale->id < 0);
    return (locale->id + 1) * -1;
}

HCLIB_REGISTER_MODULE("mpi", mpi_pre_initialize, mpi_post_initialize, mpi_finalize)
