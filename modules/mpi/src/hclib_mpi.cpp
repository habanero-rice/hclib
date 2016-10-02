#include "hclib_mpi-internal.h"
#include "hclib-locality-graph.h"

#include <iostream>

#if defined(HCLIB_MEASURE_START_LATENCY) || defined(HCLIB_PROFILE)
const char *MPI_FUNC_NAMES[N_MPI_FUNCS] = {
    "MPI_Send",
    "MPI_Recv",
    "MPI_Isend",
    "MPI_Irecv",
    "MPI_Allreduce",
    "MPI_Bcast",
    "MPI_Barrier"
};
#endif

#ifdef HCLIB_MEASURE_START_LATENCY
unsigned long long mpi_latency_counters[N_MPI_FUNCS];
unsigned long long mpi_latency_times[N_MPI_FUNCS];
#endif

#ifdef HCLIB_PROFILE
unsigned long long mpi_profile_counters[N_MPI_FUNCS];
unsigned long long mpi_profile_times[N_MPI_FUNCS];
#endif

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

#ifdef HCLIB_MEASURE_START_LATENCY
    memset(mpi_latency_counters, 0x00, N_MPI_FUNCS * sizeof(unsigned long long));
    memset(mpi_latency_times, 0x00, N_MPI_FUNCS * sizeof(unsigned long long));
#endif

#ifdef HCLIB_PROFILE
    memset(mpi_profile_counters, 0x00, N_MPI_FUNCS * sizeof(unsigned long long));
    memset(mpi_profile_times, 0x00, N_MPI_FUNCS * sizeof(unsigned long long));
#endif

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

    hclib_locale_mark_special(nic, "COMM");
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
    MPI_START_LATENCY;

    hclib::finish([&] {
        hclib::async_nb_at([&] {
            MPI_END_LATENCY(MPI_Send);
            MPI_START_PROFILE;
            CHECK_MPI(::MPI_Send(buf, count, datatype,
                    locale_id_to_mpi_rank(dest->id), tag, comm));
            MPI_END_PROFILE(MPI_Send);
        }, nic);
    });
}

void hclib::MPI_Recv(void *buf, int count, MPI_Datatype datatype, hclib::locale_t *source, int tag,
        MPI_Comm comm, MPI_Status *status) {
    MPI_START_LATENCY;

    hclib::finish([&] {
        hclib::async_nb_at([&] {
            MPI_END_LATENCY(MPI_Recv);
            MPI_START_PROFILE;
            CHECK_MPI(::MPI_Recv(buf, count, datatype,
                    locale_id_to_mpi_rank(source->id), tag, comm, status));
            MPI_END_PROFILE(MPI_Recv);
        }, nic);
    });
}

hclib::future_t *hclib::MPI_Isend(void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *dest, int tag, MPI_Comm comm) {
    MPI_START_LATENCY;

    return hclib::async_nb_future_at([=] {
        MPI_END_LATENCY(MPI_Isend);
        MPI_START_PROFILE;
        CHECK_MPI(::MPI_Send(buf, count, datatype,
                locale_id_to_mpi_rank(dest->id), tag, comm));
        MPI_END_PROFILE(MPI_Isend);
    }, nic);
}

hclib::future_t *hclib::MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
        hclib::locale_t *source, int tag, MPI_Comm comm) {
    MPI_START_LATENCY;

    return hclib::async_nb_future_at([=] {
        MPI_END_LATENCY(MPI_Irecv);
        MPI_START_PROFILE;
        MPI_Status status;
        CHECK_MPI(::MPI_Recv(buf, count, datatype,
                locale_id_to_mpi_rank(source->id), tag, comm, &status));
        MPI_END_PROFILE(MPI_Irecv);
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
    MPI_START_LATENCY;

    hclib::finish([&] {
        hclib::async_nb_at([&] {
            MPI_END_LATENCY(MPI_Allreduce);
            MPI_START_PROFILE;
            CHECK_MPI(::MPI_Allreduce(sendbuf, recvbuf, count, datatype, op,
                    comm));
            MPI_END_PROFILE(MPI_Allreduce);
        }, nic);
    });
}

void hclib::MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, 
        MPI_Comm comm) {
    MPI_START_LATENCY;

    hclib::finish([&] {
            hclib::async_nb_at([&] {
                MPI_END_LATENCY(MPI_Bcast);
                MPI_START_PROFILE;
                CHECK_MPI(::MPI_Bcast(buffer, count, datatype, root, comm));
                MPI_END_PROFILE(MPI_Bcast);
            }, nic);
    });
}

int hclib::MPI_Barrier(MPI_Comm comm) {
    MPI_START_LATENCY;

    hclib::finish([&] {
        hclib::async_nb_at([&] {
            MPI_END_LATENCY(MPI_Barrier);
            MPI_START_PROFILE;
            CHECK_MPI(::MPI_Barrier(comm));
            MPI_END_PROFILE(MPI_Barrier);
        }, nic);
    });
}

void hclib::print_mpi_profiling_data() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if defined(HCLIB_PROFILE) || defined(HCLIB_MEASURE_START_LATENCY)
    printf("Rank %d MPI PROFILE INFO:\n", rank);
    for (int i = 0; i < N_MPI_FUNCS; i++) {
#ifdef HCLIB_PROFILE
        if (mpi_profile_counters[i] > 0) {
            printf("  %s: %llu calls, %llu ms\n", MPI_FUNC_NAMES[i],
                    mpi_profile_counters[i],
                    mpi_profile_times[i] / 1000000);
        }
#endif
#ifdef HCLIB_MEASURE_START_LATENCY
        if (mpi_latency_counters[i] > 0) {
            printf("  %s: %llu calls, %llu ns mean launch-to-exec latency\n",
                    MPI_FUNC_NAMES[i],
                    mpi_latency_counters[i],
                    mpi_latency_times[i] / mpi_latency_counters[i]);
        }
#endif
    }
#endif

}


HCLIB_REGISTER_MODULE("mpi", mpi_pre_initialize, mpi_post_initialize, mpi_finalize)
