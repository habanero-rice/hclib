/** Hash table for string storage and retrieve.
 * Note the difference between partition.c and hash_table.c. The former
 * is used to partition the data between processes, and only the hash
 * function is used. The latter use a hash table to store the actual 
 * string data on each process.
 */

#include "kmi.h"

/** Hash function
 * @input str
 * @output: index of the key
 */
int _KMI_hash_table_hashfunc(KMI_hash_table_t htbl, char *str)
{
    int index = 0;
    unsigned long str_id;
#ifdef KMI_CONFIG_BINARY_STRING
    /* mapping (key_len * 2) bits to 2^KMI_HASH_INDEX_SIZE slots 
     * MAYBE: a good hash function
     * requirement: all reads with the same key should be linked together */
    if (htbl->key_len <= 32) {
        str_id = (unsigned long) (*str - '0');
    } else {
        /* for key_len > 32, use the first 32 letters, i.e., 8 bytes */
        memcpy(&str_id, str, sizeof(unsigned long));
    }
#else
    char *str_bin;
    int str_bin_len;
    KMI_str_ltr2bin(str, htbl->key_len, &str_bin, &str_bin_len);
    /* mapping (key_len * 2) bits to 2^KMI_HASH_INDEX_SIZE slots 
     * MAYBE: a good hash function
     * requirement: all reads with the same key should be linked together */
    if (htbl->key_len <= 32) {
        str_id = (unsigned long) (*str_bin - '0');
    } else {
        /* for key_len > 32, use the first 32 letters, i.e., 8 bytes */
        memcpy(&str_id, str_bin, sizeof(unsigned long));
    }
    kmi_free(str_bin);
#endif
    index = str_id % (1 << KMI_HASH_INDEX_SIZE);
    return index;
}

/** Initialize a hash table record for a rc string.
 * @input/output record
 * @input str
 * @input str_len
 */
int _KMI_hash_table_record_init_rc(KMI_hash_table_record_t *record,
                                   char *str, int str_len)
{
    debug_print("_KMI_hash_table_record_init, %s, str_len=%d\n", str, str_len);
    (*record) = (KMI_hash_table_record_t)
        kmi_malloc(sizeof(struct KMI_hash_table_record));
    (*record)->str =
        (char *) kmi_malloc(KMI_BIN_STR_LEN(str_len) * sizeof(char));
    memcpy((*record)->str, str, KMI_BIN_STR_LEN(str_len));
    (*record)->str_len = str_len;
    (*record)->count = 0;
    (*record)->rc_count = 1;
    (*record)->next = NULL;
}

/** Initialize a hash table record.
 * @input/output record
 * @input str
 * @input str_len
 */
int _KMI_hash_table_record_init(KMI_hash_table_record_t *record, char *str,
                                int str_len)
{
    debug_print("_KMI_hash_table_record_init, %s, str_len=%d\n", str, str_len);
    (*record) = (KMI_hash_table_record_t)
        kmi_malloc(sizeof(struct KMI_hash_table_record));
    (*record)->str =
        (char *) kmi_malloc(KMI_BIN_STR_LEN(str_len) * sizeof(char));
    memcpy((*record)->str, str, KMI_BIN_STR_LEN(str_len));
    (*record)->str_len = str_len;
    (*record)->count = 1;
    (*record)->rc_count = 0;
    (*record)->next = NULL;
}

/** Delete a hash table record.
 */
int KMI_hash_table_record_delete(KMI_hash_table_record_t record)
{
    debug_print("KMI_hash_table_record_delete\n");
    kmi_free(record->str);
    kmi_free(record);
}

/** Initialize a hash table.
 * @input key_len: the length of the search key, counted in char before 
 * converted to compact form.
 * @input read_len: the length of read
 * @input/output htbl: the hash table.
 */
int KMI_hash_table_init(KMI_hash_table_t *htbl, int key_len, int read_len)
{
    debug_print("KMI_hash_table_init, key_len=%d, read_len=%d\n", key_len,
                read_len);
    *htbl = (KMI_hash_table_t) kmi_malloc(sizeof(struct KMI_hash_table));
    /* MAYBE: deside the best hash table size based on key_len and read_len */
    int size = 1 << KMI_HASH_INDEX_SIZE;
    debug_print("KMI_hash_table_init, size = %d,"
                " sizeof(KMI_hash_table_record_t) = %lu\n", size,
                sizeof(KMI_hash_table_record_t));
    /* in this case, the converted key is less than 64 bits */
    (*htbl)->key_len = key_len;
    (*htbl)->read_len = read_len;
    (*htbl)->size = size;
    (*htbl)->list = (KMI_hash_table_record_t *)
        kmi_malloc(size * sizeof(KMI_hash_table_record_t));
    memset((*htbl)->list, 0, size * sizeof(KMI_hash_table_record_t));
    return 0;
}

/** Find an item in the hash table.
 * Note the difference between 'insert' and 'search'. The former insert a
 * record of record->str_len, the latter search a key of length htbl->key_len.
 * @input htbl
 * @input str
 * @output : return 1 if found, 0 if not and slot is empty, or 2 if not and 
 * slot is not empty.
 * @output index: the index of record 
 * @output rcd_found: the record found
 */
int _KMI_hash_table_search_record(KMI_hash_table_t htbl, char *str,
                                  int str_len, int *index,
                                  KMI_hash_table_record_t *rcd_found)
{
    *index = _KMI_hash_table_hashfunc(htbl, str);
    *rcd_found = (htbl->list)[*index];
    if (*rcd_found != NULL) {
        int found = 0;
        while (!found && (*rcd_found) != NULL) {
            /* Note searching use key_len, comparing the strings in the chained
             * linked-list use read_len */
            /* check both length and string */
            if (str_len == (*rcd_found)->str_len) {
#ifdef KMI_CONFIG_BINARY_STRING
                int rc = KMI_prefix_cmp(str, (*rcd_found)->str, str_len);
#else
                int rc = strncmp(str, (*rcd_found)->str, str_len);
#endif
                if (rc == 0)
                    return 1;   /* string found */
            }
            *rcd_found = (*rcd_found)->next;
        }
        return 2;               /* slot not empty, but no searched key */
    } else
        return 0;               /* slot empty */
}

int Test_KMI_hash_table_print_index(KMI_hash_table_t htbl, int index)
{
    debug_print("\tTest_KMI_hash_table_print_index at index %d:\n", index);

    KMI_hash_table_record_t rcd = htbl->list[index];
    while (rcd != NULL) {
        debug_print("\t%s(%d,%d)\n", rcd->str, rcd->count, rcd->rc_count);
        rcd = rcd->next;
    }
    debug_print("\n");
    return 0;
}

/** For test use. Seach the hash table.
 */
int Test_KMI_hash_table_search(KMI_hash_table_t htbl, char *str,
                               KMI_hash_table_record_t **list, int *len)
{
    debug_print("Test_KMI_hash_table_search, %.*s, key_len: %d\n",
                htbl->key_len, str, htbl->key_len);
    char *str_bin;
    int str_bin_len;
    KMI_str_ltr2bin(str, htbl->key_len, &str_bin, &str_bin_len);
    int rc = KMI_hash_table_search(htbl, str_bin, list, len);
    kmi_free(str_bin);
    return rc;
}

int KMI_hash_table_search(KMI_hash_table_t htbl, char *str,
                          KMI_hash_table_record_t **list, int *list_len)
{
    int res = 0;
    int index = _KMI_hash_table_hashfunc(htbl, str);
    KMI_hash_table_record_t rcd = (htbl->list)[index];
    while (rcd != NULL) {
#ifdef KMI_CONFIG_BINARY_STRING
        int rc = KMI_prefix_cmp(str, rcd->str, htbl->key_len);
#else
        int rc = strncmp(str, rcd->str, htbl->key_len);
#endif
        if (rc == 0) {
            if (*list_len == 0) {
                *list = (KMI_hash_table_record_t *)
                    kmi_malloc(sizeof(KMI_hash_table_record_t));
            } else {
                *list = kmi_realloc((*list), (*list_len + 1)
                                    * sizeof(KMI_hash_table_record_t));
            }
            (*list)[*list_len] = rcd;
            res = 1;
            (*list_len)++;
        }
        rcd = rcd->next;
    }
    return res;
}

/** For test use.
 */
int Test_KMI_hash_table_search_index(KMI_hash_table_t htbl, char *str,
                                     int *index)
{
    int res = 0;
    debug_print("Test_KMI_hash_table_search_index, %.*s\n", htbl->key_len, str);
#ifdef KMI_CONFIG_BINARY_STRING
    char *str_bin;
    int str_bin_len;
    KMI_str_ltr2bin(str, htbl->read_len, &str_bin, &str_bin_len);
    *index = _KMI_hash_table_hashfunc(htbl, str_bin);
    KMI_hash_table_record_t rcd = (htbl->list)[*index];
    while (rcd != NULL) {
        int rc = KMI_prefix_cmp(str_bin, rcd->str, htbl->key_len);
        if (rc == 0)
            res = 1;
        rcd = rcd->next;
    }
    kmi_free(str_bin);
#else
    *index = _KMI_hash_table_hashfunc(htbl, str);
    KMI_hash_table_record_t rcd = (htbl->list)[*index];
    while (rcd != NULL) {
        int rc = strncmp(str, rcd->str, htbl->key_len);
        if (rc == 0)
            res = 1;
        rcd = rcd->next;
    }
#endif
    return res;
}

int KMI_hash_table_insert(KMI_hash_table_t htbl, char *str,
                          int str_len, KMI_string_type_t type)
{
    int index;
    KMI_hash_table_record_t rcd_found;

    /* Inserting the original string */
    int rc = _KMI_hash_table_search_record(htbl, str, str_len, &index,
                                           &rcd_found);
    if (rc == 1) {
        /* add one count */
        switch (type) {
        case KMI_STRING_ORIGINAL:
            rcd_found->count++;
            break;
        case KMI_STRING_RC:
            rcd_found->rc_count++;
            break;
        default:
            printf("[error] KMI_string_type_t not recognized\n");
            return -111;
        }
    } else {
        /* insert a new record */
        KMI_hash_table_record_t rcd_new;
        switch (type) {
        case KMI_STRING_ORIGINAL:
            _KMI_hash_table_record_init(&rcd_new, str, str_len);
            break;
        case KMI_STRING_RC:
            _KMI_hash_table_record_init_rc(&rcd_new, str, str_len);
            break;
        default:
            printf("[error] KMI_string_type_t not recognized\n");
            return -111;
        }
        rcd_new->next = htbl->list[index];
        htbl->list[index] = rcd_new;
        if (rc == 2)
            debug_print("KMI_hash_table_insert %s, insert to slot %d\n",
                        str, index);
        else
            debug_print
                ("KMI_hash_table_insert %s, insert to empty slot %d\n",
                 str, index);
    }

    return rc;
}

/** For test use. Insert a read into the hash table.
 * Hash table is the internal data structure of KMI. It is not supposed 
 * to be used directly by users. This function is used in test_kmi_hash.c
 * for testing the correctness of hash table.
 * The read is stored in a KMI_hash_table_record, and the first key_len 
 * letters are used as the hash table key for search.
 */
int Test_KMI_hash_table_insert(KMI_hash_table_t htbl, char *str)
{
    char *str_bin;
    int str_bin_len;
    KMI_str_ltr2bin(str, htbl->read_len, &str_bin, &str_bin_len);
    int rc = KMI_hash_table_insert(htbl, str_bin, htbl->read_len,
                                   KMI_STRING_ORIGINAL);
    kmi_free(str_bin);
    return rc;
}

/** Delete an item in the hash table.
 * @input htbl
 * @input str
 * @output : return 1 if item in the hash table, or 0 if not
 */
int KMI_hash_table_delete(KMI_hash_table_t htbl, char *str)
{
    return 0;
}

/** Destroy the hash table.
 * @input/output htbl
 */
int KMI_hash_table_destroy(KMI_hash_table_t htbl)
{
    debug_print("KMI_hash_table_destroy\n");
    int i, len = htbl->size;
    KMI_hash_table_record_t rcd, rcd_next;
    for (i = 0; i < len; i++) {
        rcd = htbl->list[i];
        while (rcd != NULL) {
            rcd_next = rcd->next;
            kmi_free(rcd->str);
            kmi_free(rcd);
            rcd = rcd_next;
        }
    }
    kmi_free(htbl->list);
    kmi_free(htbl);
    return 0;
}
