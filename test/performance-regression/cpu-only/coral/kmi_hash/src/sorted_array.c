#include "kmi.h"

#define DH_KEY_START

#if defined(KMI_CONFIG_SORTING) && defined(KMI_CONFIG_INDEXED_ARRAY)
int kmi_check_undel(KMI_table_t table)
{
    int myid = table->myid;
    int numprocs = table->numprocs;
    if (table->s_array_undel == 0) {
        table->deleted_flag[myid] = 1;
        KMI_query_t query;
        query.str_len = 0;
        query.type = KMI_QUERY_DELETED_FLAG;
        query.db_index = table->db_index;
        query.tb_index = table->tb_index;
        int i;
        for (i = 0; i < numprocs; i++) {
            if (i != myid) {
#ifdef DEBUG_PRINT
                printf("[%d] kmi_check_undel %d\n", myid, i);
#endif
                query.from_id = myid;
                int flag = 0;
                MPI_Status status;
                MPI_Request sendreq;
                MPI_Isend(&query, sizeof(KMI_query_t), MPI_CHAR, i,
                          KMI_QUERY_DAEMON_TAG, table->comm, &sendreq);
                MPI_Test(&sendreq, &flag, &status);
                while (!flag) {
                    MPI_Test(&sendreq, &flag, &status);
                    _KMI_query_db_check(table->db);
                }
            }
        }
        return 1;
    }
    return 0;
}

int KMI_sorted_array_init(KMI_table_t table, char *old_array,
                          long long array_len)
{
    int myid = table->myid;
    int numprocs = table->numprocs;
    long long i;

    table->s_array_len = array_len;
    table->s_array_count = array_len / table->raw_string_len;
    table->s_array_undel = table->s_array_count;
    table->s_array_undel_idx = 0;
    table->all_deleted = 0;
    table->deleted_flag = (int *) kmi_malloc(numprocs * sizeof(int));
    memset(table->deleted_flag, 0, numprocs * sizeof(int));
    table->undel_rank_index = myid;
    table->s_array = kmi_realloc(old_array, array_len);
    old_array = NULL;
    kmi_check_undel(table);

#ifdef KMI_CONFIG_INDEXED_ARRAY_HASH
    /** Build index for local sorted array.  
     * The index is a direct-address hashtable, used for speeding up search.
     * Since all local strings may share first several letters, the index is 
     * built for the first KMI_ARRAY_INDEX_LEN different letters for all 
     * local strings. These strings are in the range 
     * (splitter_vector[myid - 1], splitter_vector[myid]].
     */
    int rlen = table->attr.rlen;
    const int rslen = table->raw_string_len;
    int dh_keylen = MIN(rlen, KMI_ARRAY_INDEX_LEN);
    assert(KMI_ARRAY_INDEX_LEN > 0 && KMI_ARRAY_INDEX_LEN <= 32 && rlen > 0);
    long long left, right;
    char *ptr;
    long long dh_len;
#ifdef DH_KEY_START
    int dh_key_start;
    char *str_left, *str_right;
    char *ptr_left, *ptr_right;
    if (myid == 0) {
        ptr_left = table->min_string;
    } else {
        ptr_left = table->splitter_vector + (myid - 1) * KMI_BIN_STR_LEN(rlen);
    }
    KMI_str_bin2ltr(ptr_left, rlen, &str_left);
    if (myid == numprocs - 1) {
        ptr_right = table->max_string;
    } else {
        ptr_right = table->splitter_vector + myid * KMI_BIN_STR_LEN(rlen);
    }
    KMI_str_bin2ltr(ptr_right, rlen, &str_right);
    for (i = 0; i < rlen; i++)
        if (str_left[i] != str_right[i]) {
            dh_key_start = i;
            break;
        }
    if (dh_key_start + dh_keylen > rlen)
        dh_key_start = rlen - dh_keylen;
    kmi_free(str_left);
    kmi_free(str_right);
    dh_keylen += dh_key_start;
#ifdef DEBUG_PRINT
    printf("[%d]dh_key_start = %d, dh_keylen=%d\n", myid, dh_key_start,
           dh_keylen);
#endif
#endif

    if (myid == 0) {
        left = 0;
    } else {
        ptr = table->splitter_vector + (myid - 1) * KMI_BIN_STR_LEN(rlen);
        left = KMI_char2longlong(ptr, dh_keylen);
    }
    if (myid == numprocs - 1) {
        right = 0;
        for (i = 0; i < dh_keylen * 2; i++)
            right |= 1UL << i;
    } else {
        ptr = table->splitter_vector + myid * KMI_BIN_STR_LEN(rlen);
        right = KMI_char2longlong(ptr, dh_keylen);
    }
    dh_len = right - left + 1;
#ifdef DH_KEY_START
    table->dh_key_start = dh_key_start;
#endif
    table->dh_keylen = dh_keylen;
    table->dh_len = dh_len;
    table->dh_left = left;
    if (dh_len == 0)
        return 0;

    table->dh =
        (KMI_dh_record_t *) kmi_malloc(dh_len * sizeof(KMI_dh_record_t));
    memset(table->dh, 0xff, dh_len * sizeof(KMI_dh_record_t));
    long long i_end = array_len / rslen;
    KMI_raw_string_t *ptr_raw;
    for (i = 0; i < i_end;) {
        ptr_raw = (void *) table->s_array + i * rslen;
        long long tl, tr;
        tl = KMI_char2longlong(ptr_raw->str, dh_keylen);
        assert(tl - left < dh_len);
        table->dh[tl - left].left = i;
        i++;
        if (i < i_end) {
            ptr_raw = (void *) table->s_array + i * rslen;
            tr = KMI_char2longlong(ptr_raw->str, dh_keylen);
        }
        while (tr == tl && i < i_end) {
            i++;
            ptr_raw = (void *) table->s_array + i * rslen;
            tr = KMI_char2longlong(ptr_raw->str, dh_keylen);
        }
        table->dh[tl - left].right = i - 1;
    }
#endif
    return 0;
}

int KMI_sorted_array_destroy(KMI_table_t table)
{
    kmi_free(table->s_array);
    kmi_free(table->deleted_flag);
#ifdef KMI_CONFIG_INDEXED_ARRAY_HASH
    if (table->dh_len > 0)
        kmi_free(table->dh);
#endif
    return 0;
}

/** Search a string prefix in a sorted array.
 * Return a matching range in the array res_index.
 * @input table
 * @input str
 * @input str_len
 * @input del: whether to delete after query
 * @output res_len: length of result array
 * @output res_index: the result array
 * @output: return 1 if found, 0 if not
 */
int KMI_sorted_array_search_prefix(KMI_table_t table, char *str, long str_len,
                                   int del, int *res_len, long long **res_index)
{
    const int rslen = table->raw_string_len;
    long long left = 0;
    long long right = table->s_array_len / rslen;
    long long s_array_undel = table->s_array_undel;
    *res_len = 0;
    if (right == 0 || s_array_undel == 0) {
        *res_index = NULL;
        return 0;
    }
#ifdef KMI_CONFIG_INDEXED_ARRAY_HASH
    /* speed up search using hash table */
    if (str_len >= table->dh_keylen) {
        if (table->dh_len == 0) {
            *res_index = NULL;
            return 0;
        }
        long dh_index = KMI_char2longlong(str, table->dh_keylen);
        dh_index -= table->dh_left;
        left = table->dh[dh_index].left;
        right = table->dh[dh_index].right;
        if (left == -1) {
            *res_index = NULL;
            return 0;
        }
    }
#endif

    /* binary search */
    long long middle, middle_index;
    long long s_array_undel_idx = table->s_array_undel_idx;
    int rc_cmp;
    KMI_raw_string_t *ptr;
    KMI_raw_string_t *ptr_idx;
    while (left <= right) {
        middle = (left + right) / 2;
        middle_index = middle * rslen;
        ptr = (void *) table->s_array + middle_index;
        rc_cmp = KMI_prefix_cmp(str, ptr->str, str_len);
        if (rc_cmp == 0) {
            if (ptr->deleted == 0) {
                *res_len += 1;
                if (*res_len == 1)
                    *res_index = (long long *) kmi_malloc(sizeof(long long));
                else
                    *res_index =
                        kmi_realloc(*res_index, (*res_len) * sizeof(long long));
                (*res_index)[*res_len - 1] = middle;
                if (del == 1) {
                    ptr->deleted = 1;
                    table->s_array_undel--;
                    if (middle == s_array_undel_idx) {
#ifdef DEBUG_PRINT
                        printf("delete in position s_array_undel_idx=%lld\n",
                               s_array_undel_idx);
#endif
                        s_array_undel_idx++;
                        ptr_idx =
                            (void *) table->s_array + s_array_undel_idx * rslen;
                        while (ptr_idx->deleted) {
                            s_array_undel_idx++;
                            ptr_idx =
                                (void *) table->s_array +
                                s_array_undel_idx * rslen;
                        }
                    }
                }
            }
            long long tmp = middle - 1;
            while (tmp >= 0) {
                ptr = (void *) table->s_array + tmp * rslen;
                rc_cmp = KMI_prefix_cmp(ptr->str, str, str_len);
                if (rc_cmp == 0) {
                    if (ptr->deleted == 0) {
                        *res_len += 1;
                        if (*res_len == 1)
                            *res_index =
                                (long long *) kmi_malloc(sizeof(long long));
                        else
                            *res_index =
                                kmi_realloc(*res_index,
                                            (*res_len) * sizeof(long long));
                        (*res_index)[*res_len - 1] = tmp;
                        if (del == 1) {
                            ptr->deleted = 1;
                            table->s_array_undel--;
                            if (tmp == s_array_undel_idx) {
                                s_array_undel_idx++;
                                ptr_idx =
                                    (void *) table->s_array +
                                    s_array_undel_idx * rslen;
                                while (ptr_idx->deleted) {
                                    s_array_undel_idx++;
                                    ptr_idx =
                                        (void *) table->s_array +
                                        s_array_undel_idx * rslen;
                                }
                            }
                        }
                    }
                    tmp--;
                } else
                    break;
            }
            tmp = middle + 1;
            while (tmp < right) {
                ptr = (void *) table->s_array + tmp * rslen;
                rc_cmp = KMI_prefix_cmp(ptr->str, str, str_len);
                if (rc_cmp == 0) {
                    if (ptr->deleted == 0) {
                        *res_len += 1;
                        if (*res_len == 1)
                            *res_index =
                                (long long *) kmi_malloc(sizeof(long long));
                        else
                            *res_index =
                                kmi_realloc(*res_index,
                                            (*res_len) * sizeof(long long));
                        (*res_index)[*res_len - 1] = tmp;
                        if (del == 1) {
                            ptr->deleted = 1;
                            table->s_array_undel--;
                            if (tmp == s_array_undel_idx) {
                                s_array_undel_idx++;
                                ptr_idx =
                                    (void *) table->s_array +
                                    s_array_undel_idx * rslen;
                                while (ptr_idx->deleted) {
                                    s_array_undel_idx++;
                                    ptr_idx =
                                        (void *) table->s_array +
                                        s_array_undel_idx * rslen;
                                }
                            }
                        }
                    }
                    tmp++;
                } else
                    break;
            }
            if (del == 1) {
                table->s_array_undel_idx = s_array_undel_idx;
                kmi_check_undel(table);
            }
            return 1;
        } else if (rc_cmp < 0) {
            right = middle - 1;
        } else {
            left = middle + 1;
        }
    }

    *res_index = NULL;
    return 0;
}

/* MAYBE: unify the KMI_hash_table_record_t and KMI_raw_string_t.
 * KMI_sorted_array_search and KMI_hash_table_search should use the
 * same record format.
 */
int KMI_sorted_array_search(KMI_table_t table, char *str, long str_len, int del,
                            KMI_hash_table_record_t **list, int *list_len)
{
    const int rslen = table->raw_string_len;
    int i;
    int res_len;
    long long *res_index;
    int rc = KMI_sorted_array_search_prefix(table, str, str_len, del, &res_len,
                                            &res_index);
    KMI_raw_string_t *ptr;

    if (rc == 1 && res_len > 0) {
        struct KMI_hash_table_record *new_list =
            (struct KMI_hash_table_record *)
            kmi_malloc(res_len * sizeof(struct KMI_hash_table_record));
        if (*list_len == 0) {
            *list = (KMI_hash_table_record_t *)
                kmi_malloc(res_len * sizeof(struct KMI_hash_table_record));
        } else {
            *list = kmi_realloc((*list),
                                (*list_len +
                                 res_len) * sizeof(KMI_hash_table_record_t));
        }
        for (i = 0; i < res_len; i++) {
            ptr = (void *) table->s_array + res_index[i] * rslen;
            new_list[i].str =
                (char *) kmi_malloc(KMI_BIN_STR_LEN(ptr->str_len));
            memcpy(new_list[i].str, ptr->str, KMI_BIN_STR_LEN(ptr->str_len));
            new_list[i].str_len = ptr->str_len;
            new_list[i].count = ptr->count;
            new_list[i].rc_count = ptr->rc_count;
            (*list)[(*list_len) + i] = &new_list[i];
        }
        *list_len += res_len;
    }
    if (res_len > 0)
        kmi_free(res_index);

    return 0;
}

int KMI_sorted_array_rebalance(KMI_table_t table)
{
    return 0;
}
#endif
