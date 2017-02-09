#include "kmi.h"

int KMI_db_init(KMI_db_t *db, KMI_db_type_t type, MPI_Comm comm)
{
    int numprocs, myid;
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);
    debug_print("KMI_db_init, myid=%d\n", myid);
    *db = (KMI_db_t) kmi_malloc(sizeof(struct KMI_db));
    (*db)->db_num_local_tb = 0;
    (*db)->comm = comm;
    (*db)->myid = myid;
    (*db)->numprocs = numprocs;
    (*db)->db_index = g_db_num;
    (*db)->mpi_recvbuf = (char *) kmi_malloc(KMI_MAX_STRING_LEN * sizeof(char));
    g_db_all[g_db_num++] = *db;
    return 0;
}

int KMI_db_finalize(KMI_db_t db)
{
    /* delete all tables */
    int i;
    for (i = 0; i < db->db_num_local_tb; i++) {
        KMI_table_finalize(db->tb[i]);
    }

    kmi_free(db->mpi_recvbuf);
    kmi_free(db);
}

int KMI_db_commit(KMI_db_t db)
{
    debug_print("KMI_db_commit\n");
    MPI_Barrier(db->comm);
    return 0;
}
