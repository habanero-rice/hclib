/** Key space partition.
 * Note the difference between partition.c and hash_table.c. The former
 * is used to partition the data between processes, and only the hash
 * function is used. The latter use a hash table to store the actual 
 * string data on each process.
 */
#include "kmi.h"

#ifdef KMI_CONFIG_SORTING
/** Splitter based on sort.
 * The strings in the databased in partitioned by the splitter vector
 * of length (p - 1), where p is the number of all the processes.
 * @output low
 * @output high
 */
int KMI_keyspace_splitter(KMI_table_t tb, char *str, long str_len, int numprocs,
                          int *low, int *high)
{
#ifdef DEBUG_PRINT
    debug_print_bin_str(str, str_len);
    printf(" str_len:%ld\n", str_len);
#endif
    if (numprocs == 1) {
        *low = 0;
        *high = 0;
        return 0;
    }
    assert(str_len >= 4);
    /* partition function based on splitter vector.
     * binary search a place where splitter[i-1] < str <= splitter [i]. */
    int rlen_bin = KMI_BIN_STR_LEN(tb->attr.rlen);
    int str_len_bin = KMI_BIN_STR_LEN(str_len);
    int left = 0;
    int right = numprocs - 2;
    int middle;
    int rc_cmp;
    char *ptr;
    while (left <= right) {
        middle = (left + right) / 2;
        ptr = tb->splitter_vector + middle * rlen_bin;
        rc_cmp = memcmp(str, ptr, str_len_bin);
        if (rc_cmp == 0) {
            /* special case when contiguous areas has the same prefix */
            *low = middle;
            *high = middle;
            int tmp = middle - 1;
            while (tmp >= 0) {
                ptr = tb->splitter_vector + tmp * rlen_bin;
                rc_cmp = memcmp(str, ptr, str_len_bin);
                if (rc_cmp == 0) {
                    *low = tmp;
                    tmp--;
                } else
                    break;
            }
            tmp = middle + 1;
            while (tmp <= numprocs - 2) {
                ptr = tb->splitter_vector + tmp * rlen_bin;
                rc_cmp = memcmp(str, ptr, str_len_bin);
                if (rc_cmp == 0) {
                    *high = tmp;
                    tmp++;
                } else
                    break;
            }
            return 0;
        } else if (rc_cmp < 0) {
            right = middle - 1;
        } else {
            left = middle + 1;
        }
    }
    if (left > middle) {
        *low = left;
        *high = left;
    } else {
        *low = middle;
        *high = middle;
    }
    return 0;
}
#endif

/** Hash function for key space partition.
 * The string length must >= 4, and if the string length > 32, only the first
 * 32 letters are used for this hash function.
 * @output: return node ID
 */
#ifdef KMI_CONFIG_BINARY_STRING
int KMI_keyspace_partition(char *str, long str_len, int numprocs)
{
    assert(str_len >= 4);
    int hash_key = -1;
    int i;

    /* compute the hash key */
    long len = KMI_BIN_STR_LEN(str_len);
    if (len > 8)
        len = 8;
    unsigned long magic_num = 0;
    /* FIXME: unsigned long is 64 bits, not 256 bits */
    for (i = 0; i < len; i++) {
        magic_num |= ((unsigned long) str[i]) << (i * 8);
    }

    hash_key = magic_num % numprocs;
    debug_print("magic_num=%lu, hash_key=%d, for str: %s\n", magic_num,
                hash_key, str);
    return hash_key;
}
#else
int KMI_keyspace_partition(char *str, long str_len, int numprocs)
{
    assert(str_len >= 4);
    int hash_key = -1;

    /* convert string to binary form */
    char *str_xor_bin;
    long str_bin_len = KMI_BIN_STR_LEN(str_len);
    str_xor_bin = (char *) kmi_malloc(str_bin_len);
    long i, j;
    for (i = 0; i < str_len; i++) {
        j = i / 4;
        int shift = (i % 4) * 2;
        if (shift == 0) {
            str_xor_bin[j] = (char) KMI_str_xor_nt2int(str[i]);
        } else {
            str_xor_bin[j] |= (((char) KMI_str_xor_nt2int(str[i])) << shift);
        }
    }

    /* compute the hash key */
    unsigned long magic_num = 0;
    int len, remainder;
    if (str_len > 32) {
        len = 8;
        remainder = 0;
    } else {
        len = str_len / 4;
        remainder = str_len % 4;
    }
    for (i = 0; i < len; i++) {
        magic_num |= ((unsigned long) str_xor_bin[i]) << (i * 8);
    }
    if (remainder != 0) {
        magic_num |= ((unsigned long) str_xor_bin[i]) << (i * 8);
    }
    hash_key = magic_num % numprocs;
    debug_print("magic_num=%lu, hash_key=%d, for str: %s\n", magic_num,
                hash_key, str);
    kmi_free(str_xor_bin);
    return hash_key;
}
#endif                          /* end KMI_CONFIG_BINARY_STRING */


int KMI_keyspace_partition2(char *str, long str_len, int numprocs,
                            char **str_bin)
{
    assert(str_len >= 4);
    int hash_key = -1;

    /* convert string to binary form */
    char *str_xor_bin;
    long str_bin_len = KMI_BIN_STR_LEN(str_len);
    *str_bin = (char *) kmi_malloc(str_bin_len);
    str_xor_bin = (char *) kmi_malloc(str_bin_len);
    long i, j;
    for (i = 0; i < str_len; i++) {
        j = i / 4;
        int shift = (i % 4) * 2;
        if (shift == 0) {
            (*str_bin)[j] = (char) KMI_str_nt2int(str[i]);
            str_xor_bin[j] = (char) KMI_str_xor_nt2int(str[i]);
        } else {
            (*str_bin)[j] |= (((char) KMI_str_nt2int(str[i])) << shift);
            str_xor_bin[j] |= (((char) KMI_str_xor_nt2int(str[i])) << shift);
        }
    }

    /* compute the hash key */
    unsigned long magic_num = 0;
    int len, remainder;
    if (str_len > 32) {
        len = 8;
        remainder = 0;
    } else {
        len = str_len / 4;
        remainder = str_len % 4;
    }
    for (i = 0; i < len; i++) {
        magic_num |= ((long) str_xor_bin[i]) << (i * 8);
    }
    if (remainder != 0) {
        magic_num |= ((long) str_xor_bin[i]) << (i * 8);
    }
    hash_key = magic_num % numprocs;
    debug_print("magic_num=%lu, hash_key=%d, for str: %s\n", magic_num,
                hash_key, str);
    kmi_free(str_xor_bin);
    return hash_key;
}
