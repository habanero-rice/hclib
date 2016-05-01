/** Because the query between two processes has three stages, --
 * send query, receive meta data, receive data, -- two state machines
 * are used to cope with the stages, one for send side, and the other
 * for receive side.
 *
 * To issue queries:
 *     In query.c, KMI_query_db is to send queries and receive query results.
 *     In iquery.c, KMI_iquery_db is to send queries; and KMI_test is 
 *     to receive query results.
 *
 * To answer queries:
 *     _KMI_query_db_check in query.c is to receive query, and to send 
 *     back results. Note iquery and query share the same 
 *     _KMI_query_db_check.
 */
#include "kmi.h"

/** Check the message received.
 */
inline void _KMI_query_db_check_stage0(KMI_db_t db)
{
    int flag;
    MPI_Status st;
    MPI_Test(&db->mpi_recvreq, &flag, &st);
    if (flag) {
        int count;
        MPI_Get_count(&st, MPI_CHAR, &count);
        /* use count == 1 message after all "real" messages to
         * indicate they are done */
        if (count == 1 && db->mpi_recvbuf[0] == KMI_MAGIC_CHAR) {
            db->numfinish++;
            db->stage = 4;
        } else {
            db->stage = 1;
        }
    }
}

inline int kmi_set_all_deleted(KMI_table_t tb)
{
    int rc = 0;
    int numprocs = tb->numprocs;
    int count = 0;
    int i;
    for (i = 0; i < numprocs; i++) {
        if (tb->deleted_flag[i] == 1)
            count++;
    }
    if (count == numprocs) {
#ifdef DEBUG_PRINT
        printf("[%d] all_deleted = 1, rank=%d\n", myid, rank);
#endif
        tb->all_deleted = 1;
        rc = 1;
    }
    return rc;
}

inline int kmi_set_deleted_flag(KMI_table_t tb, int rank)
{
    int myid = tb->myid;
    int numprocs = tb->numprocs;
    tb->deleted_flag[rank] = 1;
    int undel_rank_index = tb->undel_rank_index;
#ifdef DEBUG_PRINT
    printf("[%d] delete flag %d set to 1, all_deleted=%d, tb_index=%d\n", myid,
           rank, tb->all_deleted, tb->tb_index);
#endif
    int rc = kmi_set_all_deleted(tb);
    if (rc == 1)
        return 1;
    if (rank == undel_rank_index)
        while (tb->deleted_flag[undel_rank_index] == 1)
            undel_rank_index = (undel_rank_index + 1) % numprocs;
    tb->undel_rank_index = undel_rank_index;
    return 0;
}

/** Send back the query result meta-data.
 * MAYBE: piggy-back.
 */
static void _KMI_query_db_check_stage1(KMI_db_t db)
{
    int myid = db->myid;
    /* search local to get list and send it back */
    db->query = (KMI_query_t *) (db->mpi_recvbuf);
#ifdef DEBUG_PRINT
    printf("[%d]search local, str_len=%d,"
           " from_id=%d, db_index=%d, ",
           myid, db->query->str_len, db->query->from_id, db->query->db_index);
    debug_print_bin_str(db->query->str, db->query->str_len);
    printf("\n");
#endif
    int rc;
    int len = 0;
    switch (db->query->type) {
    case KMI_QUERY_PREFIX:
        rc = _KMI_query_db_prefix(db, db->query->str,
                                  db->query->str_len, db->query->etlr,
                                  &(db->list), &len);
        break;
    case KMI_QUERY_PREFIX_DELETE:
        rc = _KMI_query_db_prefix_delete(db, db->query->str,
                                         db->query->str_len, db->query->etlr,
                                         &(db->list), &len);
        break;
    case KMI_QUERY_RANDOM_PICK:
        db->stage = 5;
        return;
    case KMI_QUERY_DELETED_FLAG:
        kmi_set_deleted_flag(db->tb[db->query->tb_index], db->query->from_id);
        db->stage = 4;
        return;
    default:
        printf("db_check:query type unrecogonized\n");
        break;
    }

    /* MAYBE: compress the messages */
    /* pack metadata and send */
    int i, str_len = 0;
    for (i = 0; i < len; i++) {
        str_len += 3 * sizeof(int);     /* for str_len, count, rc_count */
        str_len += KMI_BIN_STR_LEN(db->list[i]->str_len);
    }
    db->meta_send.count = len;
    db->meta_send.len = str_len;
    db->meta_send.from_id = myid;
#ifdef DEBUG_PRINT
    printf("[%d]send meta back to %d, count:%d, len:%d\n", myid,
           db->query->from_id, len, str_len);
#endif
    MPI_Isend(&(db->meta_send), sizeof(KMI_query_meta_t), MPI_CHAR,
              db->query->from_id, KMI_QUERY_META_TAG, db->comm,
              &(db->mpi_sendreq));
    db->stage = 2;
}

/** Send back the query result data.
 */
/* MAYBE: more than one db->databuf to processing query */
static void _KMI_query_db_check_stage2(KMI_db_t db)
{
    int flag = 0;
    MPI_Test(&(db->mpi_sendreq), &flag, MPI_STATUS_IGNORE);
    if (flag) {
        int str_len = db->meta_send.len;
        int len = db->meta_send.count;
        if (str_len > 0) {
            /* pack query results and send */
            db->databuf = (char *) kmi_malloc(str_len * sizeof(char));
            char *ptr_buf = db->databuf;
            int i;
            for (i = 0; i < len; i++) {
                int *ptr_tmp = (int *) ptr_buf;
                ptr_tmp[0] = db->list[i]->str_len;
                ptr_tmp[1] = db->list[i]->count;
                ptr_tmp[2] = db->list[i]->rc_count;
                ptr_buf += 3 * sizeof(int);
                memcpy(ptr_buf, db->list[i]->str,
                       KMI_BIN_STR_LEN(db->list[i]->str_len));
                ptr_buf += KMI_BIN_STR_LEN(db->list[i]->str_len);
                kmi_free(db->list[i]->str);
            }
            MPI_Isend(db->databuf, str_len * sizeof(char), MPI_CHAR,
                      db->query->from_id, KMI_QUERY_DATA_TAG, db->comm,
                      &(db->mpi_sendreq));
            kmi_free(db->list);
            db->stage = 3;
        } else
            db->stage = 4;
    }
}

inline void _KMI_query_db_check_stage3(KMI_db_t db)
{
    int flag;
    int myid;
    MPI_Comm_rank(db->comm, &myid);
    MPI_Test(&(db->mpi_sendreq), &flag, MPI_STATUS_IGNORE);
    if (flag) {
        debug_print("[%d]send %s to %d\n", myid, db->databuf,
                    db->query->from_id);
        kmi_free(db->databuf);
        db->stage = 4;
    }
}

inline void _KMI_query_db_check_stage4(KMI_db_t db)
{
    int numprocs = db->numprocs;
    if (db->numfinish < numprocs)
        MPI_Irecv(db->mpi_recvbuf, db->mpi_recvlen, MPI_CHAR,
                  MPI_ANY_SOURCE, KMI_QUERY_DAEMON_TAG, db->comm,
                  &db->mpi_recvreq);
    db->stage = 0;
}

inline void _KMI_query_db_check_stage5(KMI_db_t db)
{
    /* send a random string back */
    KMI_table_t tb = db->tb[db->query->tb_index];
    int rlen = tb->attr.rlen;
    int str_len = 0;
    db->databuf = (char *) kmi_malloc(2 * sizeof(int) + rlen);
    char *ptr_str = db->databuf + 2 * sizeof(int);
    int rc = kmi_query_table_random_pick_local(tb, ptr_str, &str_len);
    int *ptr_int = (int *) db->databuf;
    ptr_int[0] = rc;
    ptr_int[1] = str_len;
#ifdef DEBUG_PRINT
    printf("[%d]rc=%d,str_len=%d\n", db->myid, rc, str_len);
#endif
    MPI_Isend(db->databuf, 2 * sizeof(int) + rlen, MPI_CHAR, db->query->from_id,
              KMI_QUERY_RSTR_TAG, db->comm, &(db->mpi_sendreq));
    db->stage = 3;
}

/** Check the incoming queries and outgoing answers.
 */
int _KMI_query_db_check(KMI_db_t db)
{
    switch (db->stage) {
    case 0:
        _KMI_query_db_check_stage0(db);
        break;
    case 1:
        _KMI_query_db_check_stage1(db);
        break;
    case 2:
        _KMI_query_db_check_stage2(db);
        break;
    case 3:
        _KMI_query_db_check_stage3(db);
        break;
    case 4:
        _KMI_query_db_check_stage4(db);
        break;
    case 5:
        _KMI_query_db_check_stage5(db);
        break;
    }
    return 0;
}

int KMI_query_db_start(KMI_db_t db)
{
    int myid, numprocs;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);
    if (numprocs > 1) {
        db->numfinish = 0;
        db->mpi_recvlen = KMI_MAX_STRING_LEN;
        MPI_Irecv(db->mpi_recvbuf, db->mpi_recvlen, MPI_CHAR,
                  MPI_ANY_SOURCE, KMI_QUERY_DAEMON_TAG, db->comm,
                  &db->mpi_recvreq);
        db->stage = 0;
    }
    srand((int) time(NULL) + myid);
}

int KMI_query_db_finish(KMI_db_t db)
{
    int myid, numprocs;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);

    if (numprocs > 1) {
        char buf = KMI_MAGIC_CHAR;
        MPI_Request request;
        int i;
        for (i = 0; i < numprocs; i++) {
            int dest = (myid + i) % numprocs;
            MPI_Isend(&buf, 1, MPI_CHAR, dest, KMI_QUERY_DAEMON_TAG,
                      db->comm, &request);
        }

        while (db->numfinish < numprocs) {
            _KMI_query_db_check(db);
        }
    }
}

int KMI_query_table(KMI_table_t tb, char *str, long str_len,
                    KMI_query_type_t type,
                    KMI_error_tolerance_t etlr,
                    KMI_hash_table_record_t **list, int *list_len)
{
    return 0;
}

/** Fetch strings according to pointers.
 * Before calling this fucntion, the return list will only contain the 
 * addresses of the query results. After this function is called, each
 * "str" field records in the list is filled.
 * @input/output list
 * @input list_len
 */
int _KMI_query_db_fetch(KMI_hash_table_record_t **list, int list_len)
{
    struct KMI_hash_table_record *new_list = (struct KMI_hash_table_record *)
        kmi_malloc(list_len * sizeof(struct KMI_hash_table_record));
    int i;
    for (i = 0; i < list_len; i++) {
        new_list[i] = *((*list)[i]);
        new_list[i].str =
            (char *) kmi_malloc(KMI_BIN_STR_LEN(new_list[i].str_len));
        memcpy(new_list[i].str, ((*list)[i])->str,
               KMI_BIN_STR_LEN(new_list[i].str_len));
        (*list)[i] = &new_list[i];
    }
    return 0;
}

/** Query the table and delete the matching results from table.
 */
int _KMI_query_table_prefix_delete(KMI_table_t tb, char *str, long str_len,
                                   KMI_error_tolerance_t etlr,
                                   KMI_hash_table_record_t **list,
                                   int *list_len)
{
    int rc = 0;
#ifdef KMI_CONFIG_INDEXED_ARRAY
    KMI_sorted_array_search(tb, str, str_len, 1, list, list_len);
#else
    KMI_hash_table_search(tb->htbl, str, list, list_len);
    _KMI_query_db_fetch(list, *list_len);
#endif
    if (*list_len > 0)
        rc = 1;
    return rc;
}

int _KMI_query_table_prefix(KMI_table_t tb, char *str, long str_len,
                            KMI_error_tolerance_t etlr,
                            KMI_hash_table_record_t **list, int *list_len)
{
    /* query the local hash table */
    int rc = 0;
#ifdef KMI_CONFIG_INDEXED_ARRAY
    KMI_sorted_array_search(tb, str, str_len, 0, list, list_len);
#else
    KMI_hash_table_search(tb->htbl, str, list, list_len);
    _KMI_query_db_fetch(list, *list_len);
#endif
    if (*list_len > 0)
        rc = 1;
    return rc;
}

inline int _KMI_query_db_prefix_delete(KMI_db_t db, char *str, long str_len,
                                       KMI_error_tolerance_t etlr,
                                       KMI_hash_table_record_t **list,
                                       int *list_len)
{
    int i, rc = 0;
    KMI_table_t traverse_table;

    /* search string in all tables */
    for (i = 0; i < db->db_num_local_tb; i++) {
        /* gather results from all tables */
        traverse_table = db->tb[i];
        _KMI_query_table_prefix_delete(traverse_table, str, str_len, etlr, list,
                                       list_len);
    }
    if (*list_len > 0)
        rc = 1;
    return rc;
}

inline int _KMI_query_db_prefix(KMI_db_t db, char *str, long str_len,
                                KMI_error_tolerance_t etlr,
                                KMI_hash_table_record_t **list, int *list_len)
{
    /* query the hash table */
    int i, rc = 0;
    KMI_table_t traverse_table;

    /* search string in all tables */
    for (i = 0; i < db->db_num_local_tb; i++) {
        /* gather results from all tables */
        traverse_table = db->tb[i];
        _KMI_query_table_prefix(traverse_table, str, str_len, etlr, list,
                                list_len);
    }
    if (*list_len > 0)
        rc = 1;
    return rc;
}

int KMI_query_delete_list(KMI_hash_table_record_t *list, int list_len)
{
    if (list_len > 0) {
        int i;
        for (i = 0; i < list_len; i++)
            kmi_free(list[i]->str);
        kmi_free(list);
    }
    return 0;
}

#ifdef KMI_CONFIG_SORTING
int kmi_query_db(KMI_db_t db, char *str, int str_len,
                 KMI_query_type_t type, KMI_error_tolerance_t etlr,
                 KMI_hash_table_record_t **list, int *list_len,
                 int list_len_flag)
{
    int numprocs, myid;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);
    if (numprocs > 1)
        _KMI_query_db_check(db);

    int rc = 0;
    int i;
    int str_bin_len;
    char *str_bin;
    KMI_str_ltr2bin(str, str_len, &str_bin, &str_bin_len);

    if (list_len_flag)
        *list_len = 0;
    for (i = 0; i < db->db_num_local_tb; i++) {
        /* search all tables */
        int low, high;
        KMI_keyspace_splitter(db->tb[i], str_bin, str_len, numprocs, &low,
                              &high);
        int hash_key;
        for (hash_key = low; hash_key <= high; hash_key++) {
#ifdef DEBUG_PRINT
            if (myid == 0) {
                debug_print_bin_str(str_bin, str_len);
                printf(" str_len:%d, hash_key:%d, myid=%d\n", str_len, hash_key,
                       myid);
            }
#endif
            if (hash_key == myid) {
                switch (type) {
                case KMI_QUERY_PREFIX:
                    rc = _KMI_query_table_prefix(db->tb[i], str_bin, str_len,
                                                 etlr, list, list_len);
                    break;
                case KMI_QUERY_PREFIX_DELETE:
                    rc = _KMI_query_table_prefix_delete(db->tb[i], str_bin,
                                                        str_len, etlr, list,
                                                        list_len);
                    break;
                default:
                    printf("query type unrecogonized\n");
                    return -101;
                }
            } else {
                /* send the query to the owner process */
                KMI_query_t query;
                if (str_len > KMI_MAX_QUERY_LEN * 4 || str_len < 0) {
                    printf("[error] query length %d out of limit\n", str_len);
                    exit(1);
                }
                memcpy(query.str, str_bin, KMI_BIN_STR_LEN(str_len));
                query.str_len = str_len;
                query.from_id = myid;
                query.db_index = db->db_index;
                query.type = type;
                query.etlr = etlr;
                int flag = 0;
                MPI_Status status;
                MPI_Request sendreq;
                MPI_Isend(&query, sizeof(KMI_query_t), MPI_CHAR, hash_key,
                          KMI_QUERY_DAEMON_TAG, db->comm, &sendreq);
                MPI_Test(&sendreq, &flag, &status);
                while (!flag) {
                    MPI_Test(&sendreq, &flag, &status);
                    _KMI_query_db_check(db);
                }

                /* recv query result meta-data */
                KMI_query_meta_t meta_recv;
                flag = 0;
                MPI_Request recvreq;
                MPI_Irecv(&meta_recv, sizeof(KMI_query_meta_t), MPI_CHAR,
                          hash_key, KMI_QUERY_META_TAG, db->comm, &recvreq);
                MPI_Test(&recvreq, &flag, &status);
                while (!flag) {
                    MPI_Test(&recvreq, &flag, &status);
                    _KMI_query_db_check(db);
                }
                debug_print
                    ("[%d]meta_recv, count:%d, len:%d, from_id:%d, query:%s\n",
                     myid, meta_recv.count, meta_recv.len, meta_recv.from_id,
                     str);
#ifdef DEBUG_PRINT
                printf("hash_key:%d, list_len = %d ", hash_key, *list_len);
                debug_print_bin_str(str_bin, str_len);
                printf(" meta len:%d\n", meta_recv.len);
#endif

                /* recv query result data */
                if (meta_recv.len > 0) {
                    flag = 0;
                    char *databuf =
                        (char *) kmi_malloc(meta_recv.len * sizeof(char));
                    MPI_Irecv(databuf, meta_recv.len * sizeof(char), MPI_CHAR,
                              hash_key, KMI_QUERY_DATA_TAG, db->comm, &recvreq);
                    MPI_Test(&recvreq, &flag, &status);
                    while (!flag) {
                        MPI_Test(&recvreq, &flag, &status);
                        _KMI_query_db_check(db);
                    }
                    debug_print("[%d]data received: %s\n", myid, databuf);

                    /* form a record list from databuf */
                    if (meta_recv.count > 0) {
                        struct KMI_hash_table_record *new_list =
                            (struct KMI_hash_table_record *)
                            kmi_malloc(meta_recv.count *
                                       sizeof(struct KMI_hash_table_record));
                        int i;
                        char *ptr_buf = databuf;
                        for (i = 0; i < meta_recv.count; i++) {
                            int *ptr_tmp = (int *) ptr_buf;
                            new_list[i].str_len = ptr_tmp[0];
                            new_list[i].count = ptr_tmp[1];
                            new_list[i].rc_count = ptr_tmp[2];
                            ptr_buf += 3 * sizeof(int);
                            new_list[i].str = (char *)
                                kmi_malloc(KMI_BIN_STR_LEN
                                           (new_list[i].str_len));
                            memcpy(new_list[i].str, ptr_buf,
                                   KMI_BIN_STR_LEN(new_list[i].str_len));
                            ptr_buf += KMI_BIN_STR_LEN(new_list[i].str_len);
                        }

                        int old_len = *list_len;
                        *list_len += meta_recv.count;
                        if (old_len > 0) {
                            /* already has a list */
                            *list = (KMI_hash_table_record_t *)
                                kmi_realloc(*list,
                                            (*list_len) *
                                            sizeof(KMI_hash_table_record_t));
                            for (i = 0; i < meta_recv.count; i++)
                                (*list)[old_len + i] = &new_list[i];
                        } else {
                            *list = (KMI_hash_table_record_t *)
                                kmi_malloc(meta_recv.count *
                                           sizeof(KMI_hash_table_record_t));
                            for (i = 0; i < meta_recv.count; i++)
                                (*list)[i] = &new_list[i];

                        }
                        rc = 1;
                    }

                    kmi_free(databuf);
                }
            }
        }
    }

    kmi_free(str_bin);
    return rc;
}

int kmi_query_db_bin(KMI_db_t db, char *str_bin, int str_len,
                 KMI_query_type_t type, KMI_error_tolerance_t etlr,
                 KMI_hash_table_record_t **list, int *list_len,
                 int list_len_flag)
{
    int numprocs, myid;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);
    if (numprocs > 1)
        _KMI_query_db_check(db);

    int rc = 0;
    int i;
    int str_bin_len = KMI_BIN_STR_LEN(str_len);

    if (list_len_flag)
        *list_len = 0;
    for (i = 0; i < db->db_num_local_tb; i++) {
        /* search all tables */
        int low, high;
        KMI_keyspace_splitter(db->tb[i], str_bin, str_len, numprocs, &low,
                              &high);
        int hash_key;
        for (hash_key = low; hash_key <= high; hash_key++) {
#ifdef DEBUG_PRINT
            if (myid == 0) {
                debug_print_bin_str(str_bin, str_len);
                printf(" str_len:%d, hash_key:%d, myid=%d\n", str_len, hash_key,
                       myid);
            }
#endif
            if (hash_key == myid) {
                switch (type) {
                case KMI_QUERY_PREFIX:
                    rc = _KMI_query_table_prefix(db->tb[i], str_bin, str_len,
                                                 etlr, list, list_len);
                    break;
                case KMI_QUERY_PREFIX_DELETE:
                    rc = _KMI_query_table_prefix_delete(db->tb[i], str_bin,
                                                        str_len, etlr, list,
                                                        list_len);
                    break;
                default:
                    printf("query type unrecogonized\n");
                    return -101;
                }
            } else {
                /* send the query to the owner process */
                KMI_query_t query;
                if (str_len > KMI_MAX_QUERY_LEN * 4 || str_len < 0) {
                    printf("[error] query length %d out of limit\n", str_len);
                    exit(1);
                }
                memcpy(query.str, str_bin, KMI_BIN_STR_LEN(str_len));
                query.str_len = str_len;
                query.from_id = myid;
                query.db_index = db->db_index;
                query.type = type;
                query.etlr = etlr;
                int flag = 0;
                MPI_Status status;
                MPI_Request sendreq;
                MPI_Isend(&query, sizeof(KMI_query_t), MPI_CHAR, hash_key,
                          KMI_QUERY_DAEMON_TAG, db->comm, &sendreq);
                MPI_Test(&sendreq, &flag, &status);
                while (!flag) {
                    MPI_Test(&sendreq, &flag, &status);
                    _KMI_query_db_check(db);
                }

                /* recv query result meta-data */
                KMI_query_meta_t meta_recv;
                flag = 0;
                MPI_Request recvreq;
                MPI_Irecv(&meta_recv, sizeof(KMI_query_meta_t), MPI_CHAR,
                          hash_key, KMI_QUERY_META_TAG, db->comm, &recvreq);
                MPI_Test(&recvreq, &flag, &status);
                while (!flag) {
                    MPI_Test(&recvreq, &flag, &status);
                    _KMI_query_db_check(db);
                }
#ifdef DEBUG_PRINT
                printf("hash_key:%d, list_len = %d ", hash_key, *list_len);
                debug_print_bin_str(str_bin, str_len);
                printf(" meta len:%d\n", meta_recv.len);
#endif

                /* recv query result data */
                if (meta_recv.len > 0) {
                    flag = 0;
                    char *databuf =
                        (char *) kmi_malloc(meta_recv.len * sizeof(char));
                    MPI_Irecv(databuf, meta_recv.len * sizeof(char), MPI_CHAR,
                              hash_key, KMI_QUERY_DATA_TAG, db->comm, &recvreq);
                    MPI_Test(&recvreq, &flag, &status);
                    while (!flag) {
                        MPI_Test(&recvreq, &flag, &status);
                        _KMI_query_db_check(db);
                    }
                    debug_print("[%d]data received: %s\n", myid, databuf);

                    /* form a record list from databuf */
                    if (meta_recv.count > 0) {
                        struct KMI_hash_table_record *new_list =
                            (struct KMI_hash_table_record *)
                            kmi_malloc(meta_recv.count *
                                       sizeof(struct KMI_hash_table_record));
                        int i;
                        char *ptr_buf = databuf;
                        for (i = 0; i < meta_recv.count; i++) {
                            int *ptr_tmp = (int *) ptr_buf;
                            new_list[i].str_len = ptr_tmp[0];
                            new_list[i].count = ptr_tmp[1];
                            new_list[i].rc_count = ptr_tmp[2];
                            ptr_buf += 3 * sizeof(int);
                            new_list[i].str = (char *)
                                kmi_malloc(KMI_BIN_STR_LEN
                                           (new_list[i].str_len));
                            memcpy(new_list[i].str, ptr_buf,
                                   KMI_BIN_STR_LEN(new_list[i].str_len));
                            ptr_buf += KMI_BIN_STR_LEN(new_list[i].str_len);
                        }

                        int old_len = *list_len;
                        *list_len += meta_recv.count;
                        if (old_len > 0) {
                            /* already has a list */
                            *list = (KMI_hash_table_record_t *)
                                kmi_realloc(*list,
                                            (*list_len) *
                                            sizeof(KMI_hash_table_record_t));
                            for (i = 0; i < meta_recv.count; i++)
                                (*list)[old_len + i] = &new_list[i];
                        } else {
                            *list = (KMI_hash_table_record_t *)
                                kmi_malloc(meta_recv.count *
                                           sizeof(KMI_hash_table_record_t));
                            for (i = 0; i < meta_recv.count; i++)
                                (*list)[i] = &new_list[i];

                        }
                        rc = 1;
                    }

                    kmi_free(databuf);
                }
            }
        }
    }

    return rc;
}

int KMI_query_db(KMI_db_t db, char *str, int str_len,
                 KMI_query_type_t type,
                 KMI_error_tolerance_t etlr,
                 KMI_hash_table_record_t **list, int *list_len)
{
    kmi_query_db(db, str, str_len, type, etlr, list, list_len, 1);
}

int KMI_query_db_bin(KMI_db_t db, char *str_bin, int str_len,
                 KMI_query_type_t type,
                 KMI_error_tolerance_t etlr,
                 KMI_hash_table_record_t **list, int *list_len)
{
    kmi_query_db_bin(db, str_bin, str_len, type, etlr, list, list_len, 1);
}

long long KMI_get_local_total_count(KMI_table_t tb)
{
    return tb->s_array_count;
}

long long KMI_get_local_count(KMI_table_t tb)
{
    return tb->s_array_undel;
}

int KMI_check_all_deleted(KMI_table_t tb)
{
    return tb->all_deleted;
}

int kmi_set_local_count(KMI_table_t tb, long long count)
{
    tb->s_array_undel = count;
    return 0;
}

int kmi_query_table_random_pick_remote(KMI_table_t tb, int rank, char *str,
                                       int *str_len)
{
    KMI_query_t query;
    query.str_len = 0;
    query.from_id = tb->myid;
    query.db_index = tb->db_index;
    query.tb_index = tb->tb_index;
    query.type = KMI_QUERY_RANDOM_PICK;
    int flag = 0;
    MPI_Status status;
    MPI_Request sendreq;
    MPI_Isend(&query, sizeof(KMI_query_t), MPI_CHAR, rank,
              KMI_QUERY_DAEMON_TAG, tb->comm, &sendreq);
    MPI_Test(&sendreq, &flag, &status);
    while (!flag) {
        MPI_Test(&sendreq, &flag, &status);
        _KMI_query_db_check(tb->db);
    }

    /* recv a random string */
    flag = 0;
    MPI_Request recvreq;
    int rlen = tb->attr.rlen;
    char *databuf = (char *) kmi_malloc(2 * sizeof(int) + rlen);
    MPI_Irecv(databuf, 2 * sizeof(int) + rlen, MPI_CHAR, rank,
              KMI_QUERY_RSTR_TAG, tb->comm, &recvreq);
    MPI_Test(&recvreq, &flag, &status);
    while (!flag) {
        MPI_Test(&recvreq, &flag, &status);
        _KMI_query_db_check(tb->db);
    }
    int *ptr_int = (int *) databuf;
    char *ptr_char = databuf + 2 * sizeof(int);
    if (ptr_int[0]) {
        memcpy(str, ptr_char, ptr_int[1]);
        *str_len = ptr_int[1];
        kmi_free(databuf);
        return 1;
    }

    return 0;
}

/** Pick a random string in local table. Return 1 if available, 0 if none.
 */
int kmi_query_table_random_pick_local(KMI_table_t tb, char *str, int *str_len)
{
    int myid;
    MPI_Comm_rank(tb->comm, &myid);
    int i;
    int rc = 0;
    long long s_array_count = tb->s_array_count;
    long long s_array_undel = tb->s_array_undel;
    long long s_array_undel_idx = tb->s_array_undel_idx;
    int rslen = tb->raw_string_len;
    if (s_array_undel == 0)
        return 0;

#ifdef DEBUG_PRINT
    printf
        ("[%d] There is %lld records in sorted array, s_array_count=%lld, undel=%lld\n",
         myid, tb->s_array_len, s_array_count, s_array_undel);
#endif
    int r = rand() % s_array_count;
    KMI_raw_string_t *ptr;
    for (i = 0; i < 5; i++) {
        ptr = (void *) tb->s_array + r * rslen;
        if (!(ptr->deleted)) {
            rc = 1;
            break;
        }
        r = rand() % s_array_count;
    }
    if (rc == 0) {
        /* choose one from the default position */
        ptr = (void *) tb->s_array + s_array_undel_idx * rslen;
#ifdef DEBUG_PRINT
        printf("[%d] Pick from default position: %lld, undel:%lld\n", myid,
               s_array_undel_idx, s_array_undel);
#endif
        assert(!(ptr->deleted));
    }
    *str_len = ptr->str_len;
#ifdef DEBUG_PRINT
    printf("[%d] index %d is not deleted, str_len:%d\n", myid, r, ptr->str_len);
#endif
    KMI_str_bin2ltr2(ptr->str, ptr->str_len, str);
    return rc;
}


int KMI_query_table_random_pick_local(KMI_table_t tb, char *str, int *str_len)
{
    return kmi_query_table_random_pick_local(tb, str, str_len);
}

#ifdef DEBUG_PRINT
int max_count = 1000;
int debug_count = 0;
#endif
/* A best effort random string picker. Try five times to get a random string. 
 * If all fails, use the leftmost of all strings.
 * @input tb
 * @output str
 * @output str_len: should be the same as read length
 * @return: return 1 if there is successful pick, 0 if none (i.e. all strings are deleted)
 */
int KMI_query_table_random_pick(KMI_table_t tb, char *str, int *str_len)
{
    int rc = 0;
    int numprocs, myid;
    MPI_Comm_size(tb->comm, &numprocs);
    MPI_Comm_rank(tb->comm, &myid);
    if (tb->all_deleted) {
        return 0;
    }

    while (!(tb->all_deleted) && rc == 0) {
        int r = rand() % numprocs;
        if (tb->deleted_flag[r])
            r = tb->undel_rank_index;
#ifdef DEBUG_PRINT
        if (debug_count++ < max_count) {
            printf("[%d]pick process id: %d, flags: ", myid, r);
            int i = 0;
            for (i = 0; i < numprocs; i++) {
                printf("%d ", tb->deleted_flag[i]);
            }
            printf("\n");
        }
#endif

        if (r == myid) {
            rc = kmi_query_table_random_pick_local(tb, str, str_len);
        } else {
            rc = kmi_query_table_random_pick_remote(tb, r, str, str_len);
        }

        if (rc == 0)
            kmi_set_all_deleted(tb);
    }

    /* rc = kmi_query_table_random_pick_local(tb, str, str_len); */

    return rc;
}

/* Return a group of query results.
 * e.g. str=AGTCCT, min_len=3 is equal to searching
 * a group of {CCT*, TCCT*, GTCCT*, AGTCCT}
 * and return the result in grouped list.
 * @input db
 * @input str
 * @input str_len
 * @input min_len
 * @input type
 * @input etlr
 * @output list
 * @output list_len
 */
int KMI_query_db_group(KMI_db_t db, char *str, int str_len, int min_len,
                       KMI_query_type_t type, KMI_error_tolerance_t etlr,
                       KMI_hash_table_record_t **list, int *list_len)
{
    if (min_len > str_len) {
        printf("[error]KMI_query_db_group(): min_len > str_len\n");
        exit(0);
    }
    int rc = 0;
    int i;
    *list_len = 0;
    for (i = str_len; i >= min_len; i--) {
        char *ptr_str = str + (str_len - i);
        int rc_tmp =
            kmi_query_db(db, ptr_str, i, type, etlr, list, list_len, 0);
        if (rc_tmp > 0)
            rc = 1;
    }

    return rc;
}

/* Return a group of query results.
 * e.g. str=AGTCCT, min_len=3 is equal to searching
 * a group of {CCT*, TCCT*, GTCCT*, AGTCCT}
 * and return the result in grouped list.
 * @input db
 * @input str
 * @input str_len
 * @input min_len
 * @input type
 * @input etlr
 * @output list
 * @output list_len
 * @output offsets
 */
int KMI_query_db_group_with_offsets(KMI_db_t db, char *str, int str_len, int min_len,
                                    KMI_query_type_t type, KMI_error_tolerance_t etlr,
                                    KMI_hash_table_record_t **list, int *list_len, int **offsets)
{
    if (min_len > str_len) {
        printf("[error]KMI_query_db_group(): min_len > str_len\n");
        exit(0);
    }
    int rc = 0;
    int i, j, old_len = 0;
    *list_len = 0;
    *offsets = 0;
    for (i = str_len;  i >= min_len; i--) {
        char *ptr_str = str + (str_len - i);
        int rc_tmp =
            kmi_query_db(db, ptr_str, i, type, etlr, list, list_len, 0);
        if (rc_tmp > 0) {
            rc = 1;
            *offsets = kmi_realloc(*offsets, (*list_len) * sizeof(int));
            for (j = old_len; j < (*list_len); j++)
                (*offsets)[j] = (str_len - i);
            old_len = (*list_len);
        }
    }

    return rc;
}


#else
int KMI_query_db(KMI_db_t db, char *str, int str_len,
                 KMI_query_type_t type,
                 KMI_error_tolerance_t etlr,
                 KMI_hash_table_record_t **list, int *list_len)
{
    int numprocs, myid;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);
    if (numprocs > 1)
        _KMI_query_db_check(db);

    int rc = 0;
    int i;
#ifdef KMI_CONFIG_BINARY_STRING
    int str_bin_len;
    char *str_bin;
    KMI_str_ltr2bin(str, str_len, &str_bin, &str_bin_len);
#endif
    int hash_key = KMI_keyspace_partition(KMI_STRING, str_len, numprocs);

    *list_len = 0;
    for (i = 0; i < db->db_num_local_tb; i++) {
        /* search all tables */
        if (hash_key == myid) {
            switch (type) {
            case KMI_QUERY_PREFIX:
                rc = _KMI_query_table_prefix(db->tb[i], KMI_STRING, str_len,
                                             etlr, list, list_len);
                break;
            default:
                printf("query type unrecogonized\n");
                return -101;
            }
        } else {
            /* send the query to the owner process */
            KMI_query_t query;
            memcpy(query.str, KMI_STRING, KMI_BIN_STR_LEN(str_len));
            query.str_len = str_len;
            query.from_id = myid;
            query.db_index = db->db_index;
            query.type = type;
            query.etlr = etlr;
            int flag = 0;
            MPI_Status status;
            MPI_Send(&query, sizeof(KMI_query_t), MPI_CHAR, hash_key,
                     KMI_QUERY_DAEMON_TAG, db->comm);
            debug_print("[%d] send query %s to %d\n", myid, query.str,
                        hash_key);

            /* recv query result meta-data */
            KMI_query_meta_t meta_recv;
            flag = 0;
            MPI_Request recvreq;
            MPI_Irecv(&meta_recv, sizeof(KMI_query_meta_t), MPI_CHAR,
                      hash_key, KMI_QUERY_META_TAG, db->comm, &recvreq);
            MPI_Test(&recvreq, &flag, &status);
            while (!flag) {
                MPI_Test(&recvreq, &flag, &status);
                _KMI_query_db_check(db);
            }
            debug_print
                ("[%d]meta_recv, count:%d, len:%d, from_id:%d, query:%s\n",
                 myid, meta_recv.count, meta_recv.len, meta_recv.from_id, str);

            /* recv query result data */
            if (meta_recv.len > 0) {
                flag = 0;
                char *databuf =
                    (char *) kmi_malloc(meta_recv.len * sizeof(char));
                MPI_Irecv(databuf, meta_recv.len * sizeof(char), MPI_CHAR,
                          hash_key, KMI_QUERY_DATA_TAG, db->comm, &recvreq);
                MPI_Test(&recvreq, &flag, &status);
                while (!flag) {
                    MPI_Test(&recvreq, &flag, &status);
                    _KMI_query_db_check(db);
                }
                debug_print("[%d]data received: %s\n", myid, databuf);

                /* form a record list from databuf */
                if (meta_recv.count > 0) {
                    *list = (KMI_hash_table_record_t *)
                        kmi_malloc(meta_recv.count *
                                   sizeof(KMI_hash_table_record_t));
                    struct KMI_hash_table_record *new_list =
                        (struct KMI_hash_table_record *)
                        kmi_malloc(meta_recv.count *
                                   sizeof(struct KMI_hash_table_record));
                    int i;
                    char *ptr_buf = databuf;
                    for (i = 0; i < meta_recv.count; i++) {
                        int *ptr_tmp = (int *) ptr_buf;
                        new_list[i].str_len = ptr_tmp[0];
                        new_list[i].count = ptr_tmp[1];
                        new_list[i].rc_count = ptr_tmp[2];
                        ptr_buf += 3 * sizeof(int);
                        new_list[i].str = (char *)
                            kmi_malloc(KMI_BIN_STR_LEN(new_list[i].str_len));
                        memcpy(new_list[i].str, ptr_buf,
                               KMI_BIN_STR_LEN(new_list[i].str_len));
                        ptr_buf += KMI_BIN_STR_LEN(new_list[i].str_len);
                        (*list)[i] = &new_list[i];
                    }

                    rc = 1;
                    *list_len = meta_recv.count;
                }

                kmi_free(databuf);
            }
        }
    }

#ifdef KMI_CONFIG_BINARY_STRING
    kmi_free(str_bin);
#endif
    return rc;
}
#endif
