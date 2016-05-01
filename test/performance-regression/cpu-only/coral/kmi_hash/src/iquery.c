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

int _KMI_query_fn(void *extra_state, MPI_Status * status)
{
    status->MPI_SOURCE = MPI_UNDEFINED;
    status->MPI_TAG = MPI_UNDEFINED;
    MPI_Status_set_cancelled(status, 0);
    MPI_Status_set_elements(status, MPI_CHAR, 0);
    return 0;
}

int _KMI_free_fn(void *extra_state)
{
    return 0;
}

int _KMI_cancel_fn(void *extra_state, int complete)
{
    return 0;
}

inline int _KMI_iquery_db_helper(KMI_db_t db, char *str, long str_len,
                                 KMI_query_type_t type,
                                 KMI_error_tolerance_t etlr,
                                 KMI_request_t *request,
                                 KMI_hash_table_record_t **list, int *list_len,
                                 int i, int hash_key)
{
    int numprocs, myid;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);
    int rc = 0;
    return rc;
}

int KMI_iquery_db(KMI_db_t db, char *str, long str_len,
                  KMI_query_type_t type,
                  KMI_error_tolerance_t etlr,
                  KMI_request_t *request,
                  KMI_hash_table_record_t **list, int *list_len)
{
    int numprocs, myid;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);
    int rc = 0;
    int i;
    int hash_key;
#ifdef KMI_CONFIG_BINARY_STRING
    int str_bin_len;
    char *str_bin;
    KMI_str_ltr2bin(str, str_len, &str_bin, &str_bin_len);
#endif

    MPI_Grequest_start(_KMI_query_fn, _KMI_free_fn, _KMI_cancel_fn, NULL,
                       &(request->mpi_greq));

    *list_len = 0;
    /* FIXME: for now only one table is supported for iquery.
     * If iquery for multiple tables is needed, request will still be one,
     * but instead of one mpi_greq/sendreq/recvreq, multiple requests are 
     * needed, one for each table.
     *
     * On the KMI_test side, instead of test one request, request corresponding
     * to each table should be checked use MPI_Testany.
     */
    for (i = 0; i < db->db_num_local_tb; i++) {
#ifdef KMI_CONFIG_SORTING
        /* splitter vector use 1-to-n mapping */
        int low, high;
        KMI_keyspace_splitter(db->tb[i], KMI_STRING, str_len, numprocs, &low,
                              &high);
        request->req_count = 0;
        request->unfinished_count = 0;
        request->low = low;
        request->high = high;
#ifdef DEBUG_PRINT
        printf("[%d]low:%d, high:%d\n", myid, low, high);
#endif
        if (high - low + 1 > KMI_MAX_NUM_SEND_REQ) {
            printf
                ("Splitter vector failed to limit query to less than %d processor\n",
                 KMI_MAX_NUM_SEND_REQ);
            printf("You can increase KMI_MAX_NUM_SEND_REQ in kmi.h\n");
            printf
                ("or find a better distributed sorting algorithm to solve this problem\n");
            exit(0);
        }
        for (hash_key = low; hash_key <= high; hash_key++) {
            request->hash_key = hash_key;
            request->myid = myid;
            request->db = db;
            request->list = list;
            request->list_len = list_len;
            if (hash_key == myid) {
                switch (type) {
                case KMI_QUERY_PREFIX:
                    _KMI_query_table_prefix(db->tb[i], KMI_STRING, str_len,
                                            etlr, list, list_len);
                    if (*list_len > 0)
                        rc = 1;
                    *(request->list_len) = *list_len;
                    break;
                default:
                    printf("query type unrecogonized\n");
                    return -101;
                }
            } else {
                /* send the query to the owner process */
                memcpy(request->query.str, KMI_STRING,
                       KMI_BIN_STR_LEN(str_len));
                request->query.str_len = str_len;
                request->query.from_id = myid;
                request->query.db_index = db->db_index;
                request->query.type = type;
                request->query.etlr = etlr;
                request->stage = 0;
#ifdef DEBUG_PRINT
                printf("[%d]send req to %d\n", myid, hash_key);
#endif
                MPI_Isend(&(request->query), sizeof(KMI_query_t), MPI_CHAR,
                          hash_key, KMI_QUERY_DAEMON_TAG, db->comm,
                          &(request->mpi_sendreq[request->req_count]));
                (request->req_count)++;
                (request->unfinished_count)++;
            }
        }
#else
        /* KMI_keyspace_partition use 1-to-1 mapping */
        hash_key = KMI_keyspace_partition(KMI_STRING, str_len, numprocs);
        request->hash_key = hash_key;
        request->myid = myid;
        request->db = db;
        request->list = list;
        request->list_len = list_len;
        if (hash_key == myid) {
            switch (type) {
            case KMI_QUERY_PREFIX:
                rc = _KMI_query_table_prefix(db->tb[i], KMI_STRING, str_len,
                                             etlr, list, list_len);
                *(request->list_len) = *list_len;
                break;
            default:
                printf("query type unrecogonized\n");
                return -101;
            }
        } else {
            /* send the query to the owner process */
            memcpy(request->query.str, KMI_STRING, KMI_BIN_STR_LEN(str_len));
            request->query.str_len = str_len;
            request->query.from_id = myid;
            request->query.db_index = db->db_index;
            request->query.type = type;
            request->query.etlr = etlr;
            request->stage = 0;
            MPI_Isend(&(request->query), sizeof(KMI_query_t), MPI_CHAR,
                      hash_key, KMI_QUERY_DAEMON_TAG, db->comm,
                      &(request->mpi_sendreq));
        }
#endif
    }

#ifdef KMI_CONFIG_BINARY_STRING
    kmi_free(str_bin);
#endif
    return rc;
}

inline void _KMI_test_complete(KMI_request_t *request, int *flag,
                               KMI_status_t *status)
{
    *flag = 1;
    MPI_Status_set_elements(status, MPI_CHAR, *(request->list_len));
    MPI_Grequest_complete(request->mpi_greq);
}

/** Test if the query is sent, then recv the meta-data response.
 */
inline void _KMI_test_stage0(KMI_request_t *request)
{
    int send_flag = 0;
#ifdef KMI_CONFIG_SORTING
    int index;
    MPI_Testany(request->req_count, request->mpi_sendreq, &index, &send_flag,
                MPI_STATUS_IGNORE);
    request->index = index;
    if (send_flag) {
        request->stage = 1;
        int src = request->index + request->low;
        if (request->myid <= request->high
            && request->myid >= request->low
            && src - request->low >= request->myid - request->low) {
            /* deal with the case when myid is in query range [low,high] */
            src = src + 1;
        }
#ifdef DEBUG_PRINT
        printf("[%d]src:%d, meta, index:%d,low:%d,high:%d\n", request->myid,
               src, request->index, request->low, request->high);
#endif
        MPI_Irecv(&(request->meta_recv), sizeof(KMI_query_meta_t),
                  MPI_CHAR, src, KMI_QUERY_META_TAG,
                  request->db->comm, &(request->mpi_recvreq));
    }
#else
    MPI_Test(&(request->mpi_sendreq), &send_flag, MPI_STATUS_IGNORE);
    if (send_flag) {
        request->stage = 1;
        MPI_Irecv(&(request->meta_recv), sizeof(KMI_query_meta_t),
                  MPI_CHAR, request->hash_key, KMI_QUERY_META_TAG,
                  request->db->comm, &(request->mpi_recvreq));
    }
#endif
}

/** Test if the meta-data is received, then recv the data.
 */
inline void _KMI_test_stage1(KMI_request_t *request)
{
    /* recv query result meta-data */
    int recv_flag = 0;
    MPI_Test(&(request->mpi_recvreq), &recv_flag, &(request->recv_status));
    if (recv_flag) {
        /* recv query result data */
        if (request->meta_recv.len > 0) {
            request->databuf =
                (char *) kmi_malloc(request->meta_recv.len * sizeof(char));
#ifdef KMI_CONFIG_SORTING
            int src = request->index + request->low;
            if (request->myid <= request->high
                && request->myid >= request->low
                && src - request->low >= request->myid - request->low) {
                /* deal with the case when myid is in query range [low,high] */
                src = src + 1;
            }
#ifdef DEBUG_PRINT
            printf("[%d]src:%d, data\n", request->myid, src);
#endif
            MPI_Irecv(request->databuf,
                      request->meta_recv.len * sizeof(char),
                      MPI_CHAR, src,
                      KMI_QUERY_DATA_TAG, request->db->comm,
                      &(request->mpi_recvreq));
#else
            MPI_Irecv(request->databuf,
                      request->meta_recv.len * sizeof(char),
                      MPI_CHAR, request->hash_key,
                      KMI_QUERY_DATA_TAG, request->db->comm,
                      &(request->mpi_recvreq));
#endif
            request->stage = 2;
        } else {
            request->stage = 3;
        }
    }
}

/** Test if the data is received, the form the list.
 */
inline void _KMI_test_stage2(KMI_request_t *request)
{
    int recv_flag = 0;
    MPI_Test(&(request->mpi_recvreq), &recv_flag, &(request->recv_status));
    if (recv_flag) {
        /* form a record list from databuf */
        if (request->meta_recv.count > 0) {
            int old_len = *(request->list_len);
            *(request->list_len) += request->meta_recv.count;
            if (old_len > 0) {
                *(request->list) = (KMI_hash_table_record_t *)
                    kmi_realloc(*(request->list), *(request->list_len) *
                                sizeof(KMI_hash_table_record_t));
            } else {
                *(request->list) = (KMI_hash_table_record_t *)
                    kmi_malloc(request->meta_recv.count *
                               sizeof(KMI_hash_table_record_t));
            }
            struct KMI_hash_table_record *new_list =
                (struct KMI_hash_table_record *)
                kmi_malloc(request->meta_recv.count *
                           sizeof(struct KMI_hash_table_record));
            int i;
            char *ptr_buf = request->databuf;
            for (i = 0; i < request->meta_recv.count; i++) {
                int *ptr_tmp = (int *) ptr_buf;
                new_list[i].str_len = ptr_tmp[0];
                new_list[i].count = ptr_tmp[1];
                new_list[i].rc_count = ptr_tmp[2];
                ptr_buf += 3 * sizeof(int);
                new_list[i].str =
                    (char *) kmi_malloc(new_list[i].str_len * sizeof(char));
                memcpy(new_list[i].str, ptr_buf,
                       KMI_BIN_STR_LEN(new_list[i].str_len));
                ptr_buf += KMI_BIN_STR_LEN(new_list[i].str_len);
                (*(request->list))[old_len + i] = &new_list[i];
            }
        }

        kmi_free(request->databuf);
        request->stage = 3;
    }
}

int KMI_test(KMI_request_t *request, int *flag, KMI_status_t *status)
{
    _KMI_query_db_check(request->db);
    if (request->hash_key == request->myid) {
        _KMI_test_complete(request, flag, status);
    } else {
        if (request->stage == 0)
            _KMI_test_stage0(request);
        if (request->stage == 1)
            _KMI_test_stage1(request);
        if (request->stage == 2)
            _KMI_test_stage2(request);
        if (request->stage == 3) {
#ifdef KMI_CONFIG_SORTING
            request->unfinished_count--;
            if (request->unfinished_count == 0)
                _KMI_test_complete(request, flag, status);
            else
                request->stage = 0;
#else
            _KMI_test_complete(request, flag, status);
#endif
        }
    }
    return 0;
}

int KMI_get_count(KMI_status_t *status, KMI_Datatype datatype, int *count)
{
    MPI_Get_count(status, datatype, count);
    return 0;
}

int KMI_testany(int count, KMI_request_t reqs[], int *indx,
                int *flag, KMI_status_t *status)
{
    return 0;
}
