#include "hclib_mpi-internal.h"
#include "hclib-locality-graph.h"

#include <iostream>

static int nic_locale_id;
static hclib::locale_t *nic = NULL;

static int mpi_rank_to_locale_id(int mpi_rank) {
    HASSERT(mpi_rank >= 0);
    return -1 * mpi_rank - 1;
}

static int locale_id_to_mpi_rank(int locale_id) {
    HASSERT(locale_id < 0);
    return (locale_id + 1) * -1;
}

HCLIB_MODULE_INITIALIZATION_FUNC(mpi_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
}

HCLIB_MODULE_INITIALIZATION_FUNC(mpi_post_initialize) {
    int provided;
    CHECK_MPI(MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided));
    assert(provided == MPI_THREAD_FUNNELED);
    // CHECK_MPI(MPI_Init(NULL, NULL));

    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];
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
    new_locale->id = mpi_rank_to_locale_id(rank);
    new_locale->type = nic_locale_id;
    new_locale->lbl = (char *)malloc(strlen(name_buf) + 1);
    memcpy((void *)new_locale->lbl, name_buf, strlen(name_buf) + 1);
    new_locale->metadata = malloc(sizeof(MPI_Comm));
    *((MPI_Comm *)new_locale->metadata) = comm;
    new_locale->deques = NULL;
    return new_locale;
}

void hclib::MPI_Comm_rank(MPI_Comm comm, int *rank) {
    CHECK_MPI(::MPI_Comm_rank(comm, rank));
}

void hclib::MPI_Comm_size(MPI_Comm comm, int *size) {
    CHECK_MPI(::MPI_Comm_size(comm, size));
}

int hclib::integer_rank_for_locale(locale_t *locale) {
    return locale_id_to_mpi_rank(locale->id);
}

void hclib::MPI_Send(void *buf, int count, MPI_Datatype datatype, hclib::locale_t *dest,
        int tag, MPI_Comm comm) {
    hclib::finish([buf, count, datatype, dest, tag, comm] {
        hclib::async_at(nic, [buf, count, datatype, dest, tag, comm] {
            CHECK_MPI(::MPI_Send(buf, count, datatype,
                    locale_id_to_mpi_rank(dest->id), tag, comm));
        });
    });
}

void hclib::MPI_Recv(void *buf, int count, MPI_Datatype datatype, hclib::locale_t *source, int tag,
        MPI_Comm comm, MPI_Status *status) {
    hclib::finish([buf, count, datatype, source, tag, comm, status] {
        hclib::async_at(nic, [buf, count, datatype, source, tag, comm, status] {
            CHECK_MPI(::MPI_Recv(buf, count, datatype,
                    locale_id_to_mpi_rank(source->id), tag, comm, status));
        });
    });
}

hclib::future_t *hclib::MPI_Isend(void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *dest, int tag, MPI_Comm comm) {
    return hclib::async_future_at([buf, count, datatype, dest, tag, comm] {
        CHECK_MPI(::MPI_Send(buf, count, datatype,
                locale_id_to_mpi_rank(dest->id), tag, comm));
    }, nic);
}

hclib::future_t *hclib::MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *source, int tag, MPI_Comm comm) {
    return hclib::async_future_at([buf, count, datatype, source, tag, comm] {
        MPI_Status status;
        CHECK_MPI(::MPI_Recv(buf, count, datatype,
                locale_id_to_mpi_rank(source->id), tag, comm, &status));
    }, nic);
}

void hclib::MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
    CHECK_MPI(::MPI_Comm_dup(comm, newcomm));
}

void hclib::MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
    CHECK_MPI(::MPI_Comm_split(comm, color, key, newcomm));
}

void hclib::MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    hclib::finish([&sendbuf, &recvbuf, &count, &datatype, &op, &comm] {
        hclib::async_at(nic, [&sendbuf, &recvbuf, &count, &datatype, &op, &comm] {
            CHECK_MPI(::MPI_Allreduce(sendbuf, recvbuf, count, datatype, op,
                    comm));
        });
    });
}

void hclib::MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, 
        MPI_Comm comm) {
    hclib::finish([&buffer, &count, &datatype, &root, &comm] {
            hclib::async_at(nic, [&buffer, &count, &datatype, &root, &comm] {
                CHECK_MPI(::MPI_Bcast(buffer, count, datatype, root, comm));
            });
    });
}

int hclib::MPI_Barrier(MPI_Comm comm) {
    hclib::finish([&comm] {
        hclib::async_at(nic, [&comm] {
            CHECK_MPI(::MPI_Barrier(comm));
        });
    });
}

HCLIB_REGISTER_MODULE("mpi", mpi_pre_initialize, mpi_post_initialize, mpi_finalize)
