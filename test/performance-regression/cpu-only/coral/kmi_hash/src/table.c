#include "kmi.h"
#include <limits.h>

int KMI_table_init_fixed_length(KMI_table_t table)
{
#if 0
    /* MAYBE: if it is de Bruijn graph */
    if (table->attr.qlen == table->attr.rlen - 1) {
    }
#endif
}

int KMI_table_init_variable_length(KMI_table_t table)
{
}

#ifdef KMI_CONFIG_SORTING
int KMI_compare_raw_string(const void *a, const void *b)
{
    KMI_raw_string_t *pa = (KMI_raw_string_t *) a;
    KMI_raw_string_t *pb = (KMI_raw_string_t *) b;
    int rc;
    if (pa->str_len == pb->str_len) {
        rc = memcmp(pa->str, pb->str, KMI_BIN_STR_LEN(pa->str_len));
    } else if (pa->str_len < pb->str_len) {
        rc = memcmp(pa->str, pb->str, KMI_BIN_STR_LEN(pa->str_len));
        if (rc != 0)
            return rc;
        else
            rc = -1;
    } else {
        rc = memcmp(pa->str, pb->str, KMI_BIN_STR_LEN(pb->str_len));
        if (rc != 0)
            return rc;
        else
            rc = 1;
    }
    return rc;
}

int KMI_compare_raw_string_min(const void *a, const void *b)
{
    KMI_raw_string_t *pa = (KMI_raw_string_t *) a;
    KMI_raw_string_t *pb = (KMI_raw_string_t *) b;
    int len = MIN(pa->str_len, pb->str_len);
    int rc = memcmp(pa->str, pb->str, KMI_BIN_STR_LEN(len));
    return rc;
}

/** Do a distributed histogram sort of the data, partition the data, then 
 * redistribute the data.
 */
int KMI_table_commit_fixed_length(KMI_table_t table)
{
#ifdef KMI_PROFILING_ALLTOALL
    double time_commit = 0;
    double time_localsort = 0;
    double time_sampling = 0;
    double time_alltoall = 0;
    double time_insert = 0;
    double t1, t2, t3, t4, t5;
    t1 = MPI_Wtime();
#endif
    int myid = table->myid;
    int numprocs = table->numprocs;
    long long i, j, j_last;
    int rc = 0;
    /** Because the string is stored in compressed binary form, the
     * following different type of lengths are used.
     *
     * splitter_vector, sampling_pivots:  
     *   original length "rlen", actual length "rlen_bin".
     * raw_string_buffer[i]: 
     *   struct length "rslen", string length "rlen", 
     *   string actual length "rlen_bin" 
     */
    const int rslen = table->raw_string_len;
    int rlen = table->attr.rlen;
    int rlen_bin = KMI_BIN_STR_LEN(rlen);

    /* local sort of commit buffer */
    qsort(table->raw_string_buffer, table->raw_string_count,
          rslen, KMI_compare_raw_string);

#ifdef KMI_PROFILING_ALLTOALL
    t2 = MPI_Wtime();
#endif
    if (table->raw_string_count == 0) {
        printf
            ("Process %d should at least insert one string to make KMI_table_commit work\n",
             myid);
        exit(0);
    }
#ifdef KMI_CONFIG_SAMPLING_ITERATION
    double ratio = numprocs;
    double global_ratio;
    long long num_pivots = 4 * (numprocs - 1);
    long long max_num_pivots =
        MIN(16 * (numprocs - 1), table->raw_string_count);
    if (max_num_pivots < 2)
        max_num_pivots = 2;
    long long global_max_num_pivots;
    MPI_Allreduce(&max_num_pivots, &global_max_num_pivots, 1, MPI_LONG_LONG,
                  MPI_MAX, table->comm);
    max_num_pivots = global_max_num_pivots;

    /* get max and min string of ALL strings */
    KMI_raw_string_t *ptr1 = (void *) table->raw_string_buffer;
    KMI_raw_string_t *ptr2;
    char *ptr_char1;
    char *ptr_char2;
    table->mpi_recvbuf = (char *) kmi_malloc(numprocs * rlen_bin);
    MPI_Allgather(ptr1->str, rlen_bin, MPI_CHAR, table->mpi_recvbuf, rlen_bin,
                  MPI_CHAR, table->comm);
    for (i = 1, j = 0; i < numprocs; i++) {
        ptr_char1 = table->mpi_recvbuf + j * rlen_bin;
        ptr_char2 = table->mpi_recvbuf + i * rlen_bin;
        int rc_cmp = KMI_prefix_cmp(ptr_char1, ptr_char2, rlen);
        if (rc_cmp > 0)
            j = i;
    }
    ptr_char1 = table->mpi_recvbuf + j * rlen_bin;
    memcpy(table->min_string, ptr_char1, rlen_bin);
    ptr1 = (void *) table->raw_string_buffer
        + (size_t)(table->raw_string_count - 1) * rslen;
    MPI_Allgather(ptr1->str, rlen_bin, MPI_CHAR, table->mpi_recvbuf, rlen_bin,
                  MPI_CHAR, table->comm);
    for (i = 1, j = 0; i < numprocs; i++) {
        ptr_char1 = table->mpi_recvbuf + j * rlen_bin;
        ptr_char2 = table->mpi_recvbuf + i * rlen_bin;
        int rc_cmp = KMI_prefix_cmp(ptr_char1, ptr_char2, rlen);
        if (rc_cmp < 0)
            j = i;
    }
    ptr_char1 = table->mpi_recvbuf + j * rlen_bin;
    memcpy(table->max_string, ptr_char1, rlen_bin);
    kmi_free(table->mpi_recvbuf);

    /** Sampling pivots, determine splitter vector.
     */
    /* initialize splitter vector */
    char *str_diff = (char *) kmi_malloc(rlen_bin);
    char *str_diff_step = (char *) kmi_malloc(rlen_bin);
    KMI_binary_sub(table->max_string, table->min_string, rlen, str_diff);
    KMI_binary_div(str_diff, rlen, numprocs, str_diff_step);
    ptr_char1 = table->splitter_vector;
    KMI_binary_add(table->min_string, str_diff_step, rlen, ptr_char1);
    for (i = 1; i < numprocs; i++) {
        ptr_char1 = table->splitter_vector + (i - 1) * rlen_bin;
        ptr_char2 = table->splitter_vector + i * rlen_bin;
        KMI_binary_add(ptr_char1, str_diff_step, rlen, ptr_char2);
    }
#ifdef DEBUG_PRINT
    if (myid == 0) {
        printf("min: ");
        debug_print_bin_str(table->min_string, rlen);
        printf(", max: ");
        debug_print_bin_str(table->max_string, rlen);
        printf(", diff: ");
        debug_print_bin_str(str_diff, rlen);
        printf(", diff step: ");
        debug_print_bin_str(str_diff_step, rlen);
        printf("\n");
        printf("init splitter vector: ");
        int print_len = 5;
        for (i = 0; i < print_len && i < numprocs - 1; i++) {
            ptr_char1 = table->splitter_vector + i * rlen_bin;
            debug_print_bin_str(ptr_char1, rlen);
            printf("\t");
        }
        printf("...");
        if (numprocs > print_len) {
            for (i = numprocs - print_len; i < numprocs - 1; i++) {
                ptr_char1 = table->splitter_vector + i * rlen_bin;
                debug_print_bin_str(ptr_char1, rlen);
                printf("\t");
            }
        }
        printf("\n");
    }
#endif
    int sampling_num_total;
    int sampling_num_actual;    /* because of round up, it may not equal to sampling_num_total */
    int sampling_num_average;
    int sampling_num_uitv;      /* number of unfilled intervals */
    int sampling_num_fitv;      /* number of filled intervals */
    long long step = numprocs;
    if (step < max_num_pivots / 16)
        step = max_num_pivots;
    while (ratio > KMI_RATIO_MAX && num_pivots < max_num_pivots) {
        sampling_num_total = num_pivots - (numprocs - 1);
        sampling_num_fitv = 0;
        sampling_num_actual = 0;
        /* calculate number of intervals that needs to be filled */
        int last_flag = 0;
        for (i = 0; i < numprocs; i++) {
            if (last_flag == 1 && table->splitter_flag[i] == 1) {       /* the (numprocs - 1) flag is always 1 */
                sampling_num_fitv++;
            }
            last_flag = table->splitter_flag[i];
        }
        sampling_num_uitv = numprocs - sampling_num_fitv;
        if (sampling_num_uitv == 0)
            break;

        /* fill intervals with pivots */
        sampling_num_average = sampling_num_total / sampling_num_uitv;
        assert(sampling_num_average > 0);
        char *fill_left, *fill_right, *fill_pivots;
        fill_left = table->min_string;
        int sampling_dist = 0;
        int sampling_num_fpvt;  /* number of fill pivots in current interval */
        char *sampling_pivots = (char *) kmi_malloc(num_pivots * rlen_bin);
        int sampling_index = 0; /* index for sampling_pivots[] */
        if (table->splitter_flag[0] == 0) {
            last_flag = 0;
            sampling_dist++;
        } else {
            last_flag = 1;
            sampling_dist = 0;
            fill_right = table->splitter_vector;
            fill_pivots = sampling_pivots;
            memcpy(fill_pivots, fill_right, rlen_bin);
            sampling_index++;
            fill_left = table->splitter_vector;
        }
        for (i = 1; i < numprocs; i++) {
            /* 00: dist++;
             * 01: dist++; fill; add splitter; change last_flag
             * 10: dist = 1; change last_flag
             * 11: dist = 0; add splitter */
            if (last_flag == 0 && table->splitter_flag[i] == 0) {
                sampling_dist++;
            } else if (last_flag == 0 && table->splitter_flag[i] == 1) {
                if (i == numprocs - 1) {
                    fill_right = table->max_string;
                } else {
                    fill_right = table->splitter_vector + rlen_bin * i;
                }
                sampling_dist++;
                sampling_num_fpvt = sampling_num_average * sampling_dist;
                assert(sampling_num_fpvt > 0);
                KMI_binary_sub(fill_right, fill_left, rlen, str_diff);
                KMI_binary_div(str_diff, rlen, sampling_num_fpvt,
                               str_diff_step);
#ifdef DEBUG_PRINT
                if (myid == 0) {
                    printf("i = %lld, fpvt: %d, average:%d, dist:%d, left: ",
                           i, sampling_num_fpvt, sampling_num_average,
                           sampling_dist);
                    debug_print_bin_str(fill_left, rlen);
                    printf(" right: ");
                    debug_print_bin_str(fill_right, rlen);
                    printf(" diff: ");
                    debug_print_bin_str(str_diff, rlen);
                    printf(" diff step: ");
                    debug_print_bin_str(str_diff_step, rlen);
                    printf("\n");
                }
#endif

                KMI_binary_add(fill_left, str_diff_step, rlen, str_diff);
                fill_pivots = sampling_pivots + rlen_bin * sampling_index;
                memcpy(fill_pivots, str_diff, rlen_bin);
                sampling_index++;
                for (j = 1; j < sampling_num_fpvt; j++) {
                    fill_left =
                        sampling_pivots + rlen_bin * (sampling_index - 1);
                    KMI_binary_add(fill_left, str_diff_step, rlen, str_diff);
                    fill_pivots = sampling_pivots + rlen_bin * sampling_index;
                    memcpy(fill_pivots, str_diff, rlen_bin);
                    sampling_index++;
                }
                fill_pivots = sampling_pivots + rlen_bin * sampling_index;
                memcpy(fill_pivots, fill_right, rlen_bin);
                sampling_index++;

                fill_left = table->splitter_vector + rlen_bin * i;
                last_flag = 1;
            } else if (last_flag == 1 && table->splitter_flag[i] == 0) {
                sampling_dist = 1;
                last_flag = 0;
            } else {
                if (i == numprocs - 1) {
                    fill_right = table->max_string;
                } else {
                    fill_right = table->splitter_vector + rlen_bin * i;
                }
                fill_pivots = sampling_pivots + rlen_bin * sampling_index;
                memcpy(fill_pivots, fill_right, rlen_bin);
                sampling_index++;

                sampling_dist = 0;
                fill_left = table->splitter_vector + rlen_bin * i;
            }
        }
        assert(sampling_index <= num_pivots);
#ifdef DEBUG_PRINT
        if (myid == 0) {
            printf("#sampling pivots:%d, sampling_pivots: ", sampling_index);
            int print_len = 10;
            for (i = 0; i < print_len && i < sampling_index; i++) {
                ptr_char1 = sampling_pivots + rlen_bin * i;
                debug_print_bin_str(ptr_char1, rlen);
                printf(" ");
            }
            printf("...");
            if (sampling_index > print_len) {
                for (i = sampling_index - print_len; i < sampling_index; i++) {
                    ptr_char1 = sampling_pivots + rlen_bin * i;
                    debug_print_bin_str(ptr_char1, rlen);
                    printf(" ");
                }
            }
            printf("\n");
        }
#endif

        /** Compute local & global histogram.
         */
        sampling_num_actual = sampling_index;
        table->local_histo =
            (long long *) kmi_malloc(sampling_num_actual * sizeof(long long));
        table->global_histo =
            (long long *) kmi_malloc(sampling_num_actual * sizeof(long long));
        j_last = 0;
        /* compute local histogram */
        for (i = 0, j = 0;
             i < sampling_num_actual && j < table->raw_string_count;) {
            ptr_char1 = sampling_pivots + i * rlen_bin;
            ptr1 = (void *) table->raw_string_buffer + j * rslen;
            int rc_cmp = KMI_prefix_cmp(ptr_char1, ptr1->str, rlen);
            if (rc_cmp < 0) {
                table->local_histo[i] = j - j_last;
                i++;
                j_last = j;
            } else {
                j++;
            }
        }
        if (i < sampling_num_actual)
            table->local_histo[i++] = j - j_last;
        while (i < sampling_num_actual)
            table->local_histo[i++] = 0;
        if (j < table->raw_string_count)
            table->local_histo[sampling_num_actual - 1] +=
                table->raw_string_count - j;
        /* compute global histogram */
        MPI_Allreduce(table->local_histo, table->global_histo,
                      sampling_num_actual, MPI_LONG_LONG, MPI_SUM, table->comm);
        /* compute new splitter_vector */
        for (i = 1; i < sampling_num_actual; i++)
            table->global_histo[i] += table->global_histo[i - 1];
        double average =
            (double) table->global_histo[sampling_num_actual - 1] / numprocs;
        double sum = average;
        j = 0;
        double max_ratio = 1;
        for (i = 0; i < sampling_num_actual; i++) {
            if (table->global_histo[i] > sum) {
                double current_ratio;
                current_ratio = (table->global_histo[i] - sum) / average + 1;
                ptr_char1 = sampling_pivots + i * rlen_bin;
                if (table->splitter_flag[j] == 0) {
                    /* record the splitter */
                    char *ptr_splitter = table->splitter_vector + j * rlen_bin;
                    memcpy(ptr_splitter, ptr_char1, rlen_bin);
                    if (current_ratio < KMI_RATIO_MAX) {
#ifdef DEBUG_PRINT
                        if (myid == 0) {
                            printf("record splitter[%lld], i=%lld: ", j, i);
                            debug_print_bin_str(ptr_splitter, rlen);
                            printf("\n");
                        }
#endif
                        table->splitter_flag[j] = 1;
                    }
                }
                j++;
                if (current_ratio > max_ratio)
                    max_ratio = current_ratio;
                sum += average;
            }
        }
        ratio = max_ratio;

        kmi_free(sampling_pivots);
        kmi_free(table->local_histo);
        kmi_free(table->global_histo);
        MPI_Allreduce(&ratio, &global_ratio, 1, MPI_DOUBLE, MPI_MAX,
                      table->comm);
        if (global_ratio <= KMI_RATIO_MAX && num_pivots < max_num_pivots) {
            assert(j == numprocs - 1);
            rc = 1;
            break;
        }
        num_pivots += step;
#ifdef DEBUG_PRINT
        if (myid == 0)
            printf
                ("num_pivots: %lld, step: %lld, max_num_pivots:%lld, ratio:%f\n",
                 num_pivots, step, max_num_pivots, global_ratio);
#endif
    }
#ifdef DEBUG_PRINT
    if (myid == 0) {
        printf("splitter:\n");
        int print_len = 10;
        for (i = 0; i < print_len && i < numprocs; i++) {
            ptr_char1 = table->splitter_vector + rlen_bin * i;
            debug_print_bin_str(ptr_char1, rlen);
            printf(" ");
        }
        printf("...");
        if (numprocs > print_len) {
            for (i = numprocs - print_len; i < numprocs; i++) {
                ptr_char1 = table->splitter_vector + rlen_bin * i;
                debug_print_bin_str(ptr_char1, rlen);
                printf(" ");
            }
        }
        printf("\n");
    }
#endif
    kmi_free(str_diff);
    kmi_free(str_diff_step);

#else
    double ratio = numprocs;    /* skew ratio */
    double global_ratio;
    long long num_pivots =
        MIN(numprocs, (table->raw_string_count) / 16 / numprocs);
    if (num_pivots < 1)
        num_pivots = 1;
    if (num_pivots > 256)
        num_pivots = 256;
    long long global_num_pivots;
    MPI_Allreduce(&num_pivots, &global_num_pivots, 1, MPI_LONG_LONG, MPI_MAX,
                  table->comm);
    num_pivots = global_num_pivots;
    long long max_num_pivots =
        MIN(numprocs * 16, table->raw_string_count / numprocs);
    if (max_num_pivots < 2)
        max_num_pivots = 2;
    if (max_num_pivots > 4096)
        max_num_pivots = 4096;
    long long global_max_num_pivots;
    MPI_Allreduce(&max_num_pivots, &global_max_num_pivots, 1, MPI_LONG_LONG,
                  MPI_MAX, table->comm);
    max_num_pivots = global_max_num_pivots;
    while (ratio > KMI_RATIO_MAX && num_pivots < max_num_pivots) {
#ifdef DEBUG_PRINT
        if (myid == 0) {
            printf("[%d]ratio:%f,num_pivots:%lld,max:%lld\n", myid, ratio,
                   num_pivots, max_num_pivots);
        }
#endif
        /* sampling #num_pivots pivots and broadcast the pivots */
        long long num_pivots_all = num_pivots * numprocs;
        table->mpi_sendbuf = (char *) kmi_malloc(num_pivots * rslen);
        table->mpi_recvbuf = (char *) kmi_malloc(num_pivots_all * rslen);
#ifdef KMI_CONFIG_SORTING_RANDOM
        long long num_block = table->raw_string_count / (num_pivots + 1);
        srand((int) time(NULL));
#endif
        for (i = 0; i < num_pivots; i++) {
#ifdef KMI_CONFIG_SORTING_RANDOM
            /* add randomness */
            int r = rand() % (2 * num_block) - num_block;
            j = (i + 1) * num_block + r;
            assert(j >= 0 && j <= table->raw_string_count);
#else
            j = (i + 1) * (table->raw_string_count / (num_pivots + 1));
#endif
            char *dest = table->mpi_sendbuf + i * rslen;
            char *src = (void *) (table->raw_string_buffer)
                + j * rslen;
            memcpy(dest, src, rslen);
        }
        MPI_Allgather(table->mpi_sendbuf, num_pivots * rslen, MPI_CHAR,
                      table->mpi_recvbuf, num_pivots * rslen, MPI_CHAR,
                      table->comm);

        /* sort p*k pivots, compute a local histogram h */
        qsort(table->mpi_recvbuf, num_pivots_all,
              rslen, KMI_compare_raw_string);
        /* for i in mpi_recvbuf, find its position p_i in raw_string_buffer
         * record p_{i+1} - p_{i} in local_histo */
        table->local_histo =
            (long long *) kmi_malloc(num_pivots_all * sizeof(long long));
        table->global_histo =
            (long long *) kmi_malloc(num_pivots_all * sizeof(long long));
        j_last = 0;
        for (i = 0, j = 0; i < num_pivots_all && j < table->raw_string_count;) {
            /* compare raw_string_buffer[j] and mpi_recvbuf[i] */
            KMI_raw_string_t *ptr1 = (void *) table->mpi_recvbuf + i * rslen;
            KMI_raw_string_t *ptr2 =
                (void *) table->raw_string_buffer + j * rslen;
            int rc_cmp = KMI_compare_raw_string(ptr1, ptr2);
            if (rc_cmp < 0) {
                table->local_histo[i] = j - j_last;
                i++;
                j_last = j;
#ifdef DEBUG_PRINT
                if (j < 0)
                    printf("[%d](%lld,%lld)", myid, i, j);
#endif
            } else {
                j++;
            }
        }
#ifdef DEBUG_PRINT
        if (myid == 0)
            printf
                ("[%d]j=%lld,j_last:%lld,num_pivots_all:%lld,raw_string_count:%d,i:%lld\n",
                 myid, j, j_last, num_pivots_all, table->raw_string_count, i);
#endif
        if (i < num_pivots_all)
            table->local_histo[i++] = j - j_last;
        while (i < num_pivots_all)
            table->local_histo[i++] = 0;
        if (j < table->raw_string_count)
            table->local_histo[num_pivots_all - 1] +=
                table->raw_string_count - j;

        /* reduce histogram of all processes to H */
        MPI_Allreduce(table->local_histo, table->global_histo, num_pivots_all,
                      MPI_LONG_LONG, MPI_SUM, table->comm);

        /* compute (p-1) pivots out of (p*k) based on H */
        for (i = 1; i < num_pivots_all; i++)
            table->global_histo[i] += table->global_histo[i - 1];
        double average =
            (double) table->global_histo[num_pivots_all - 1] / numprocs;
#ifdef DEBUG_PRINT
        if (myid == 0) {
            printf("average:%f, num_pivots_all:%lld\n", average,
                   num_pivots_all);
            int print_len = 10;
            for (i = 0; i < num_pivots_all && i < print_len; i++)
                printf("%lld ", table->global_histo[i]);
            printf("...");
            if (num_pivots_all - print_len > 0) {
                for (i = num_pivots_all - print_len; i < num_pivots_all; i++)
                    printf("%lld ", table->global_histo[i]);
            }
            printf("\n");
        }
#endif
        double sum = average;
        j = 0;
        double max_ratio = 1;
        for (i = 0; i < num_pivots_all; i++) {
            if (table->global_histo[i] > sum) {
                double current_ratio;
                KMI_raw_string_t *ptr;
                current_ratio = (table->global_histo[i] - sum) / average + 1;
                ptr = (void *) table->mpi_recvbuf + i * rslen;
#if 0
                /* MAYBE: need to fix case 0 */
                if (table->global_histo[i] - sum <
                    sum - table->global_histo[i - 1]) {
                    current_ratio =
                        (table->global_histo[i] - sum) / average + 1;
                    ptr = (void *) table->mpi_recvbuf + i * rslen;
                } else {
                    current_ratio =
                        (sum - table->global_histo[i - 1]) / average + 1;
                    ptr = (void *) table->mpi_recvbuf + (i - 1) * rslen;
                }
#endif
                /* record the splitter */
                char *ptr_splitter = table->splitter_vector + j * rlen_bin;
                j++;
#ifdef DEBUG_PRINT
                if (myid == 0)
                    printf("i:%lld, j:%lld, memcpy (%p <- %p), %d\n", i, j,
                           ptr_splitter, ptr->str, rlen_bin);
#endif
                memcpy(ptr_splitter, ptr->str, rlen_bin);
                if (current_ratio > max_ratio)
                    max_ratio = current_ratio;
                sum += average;
            }
        }
        ratio = max_ratio;
        MPI_Allreduce(&ratio, &global_ratio, 1, MPI_DOUBLE, MPI_MAX,
                      table->comm);
        kmi_free(table->local_histo);
        kmi_free(table->global_histo);
        kmi_free(table->mpi_sendbuf);
        kmi_free(table->mpi_recvbuf);
        if (global_ratio <= KMI_RATIO_MAX && num_pivots < max_num_pivots) {
            assert(j == numprocs - 1);
            rc = 1;
            break;
        } else {
            num_pivots *= 2;
        }

    }

#endif

#ifdef KMI_PROFILING_ALLTOALL
    if (myid == 0)
        printf
            ("Histogram sort sampling finish, num_pivots = %lld, screw ratio=%f\n",
             num_pivots, global_ratio);
    MPI_Barrier(table->comm);
    t3 = MPI_Wtime();
#endif
    
    
    
    long long *i_sendcounts, *i_recvcounts, *i_sdispls, *i_rdispls;
	i_sendcounts = (long long*)malloc (numprocs * sizeof(long long));
	i_recvcounts = (long long*)malloc (numprocs * sizeof(long long));
	i_sdispls = (long long*)malloc (numprocs * sizeof(long long));
	i_rdispls = (long long*)malloc (numprocs * sizeof(long long));
    
	int tmp1;
	for (tmp1 = 0; tmp1 < numprocs; tmp1++)
	{
		i_sendcounts[tmp1] = 0;
		i_recvcounts[tmp1] = 0;
		i_sdispls[tmp1] = 0;
		i_rdispls[tmp1] = 0;
	}
    
    
    /* re-distribute the data */
    /* MAYBE: compress the duplicates */
    j_last = 0;
    for (i = 0, j = 0; i < numprocs - 1 && j < table->raw_string_count;) {
        /* Prepare data for MPI_Alltoall.
         * Compare raw_string_buffer[j] and splitter_vector[i].
         * find (i,j) where raw_string_count[j] < splitter_vector[i]
         * <= raw_string_count[j+1], record in mpi_sendcounts. */
        char *ptr1 = table->splitter_vector + i * rlen_bin;
        KMI_raw_string_t *ptr2 = (void *) table->raw_string_buffer + j * rslen;
        int rc_cmp = memcmp(ptr1, ptr2->str, rlen_bin);
        if (rc_cmp < 0) {
                //table->mpi_sendcounts[i] = rslen * (j - j_last);
            i_sendcounts[i] = rslen * (j - j_last);
            i++;
            j_last = j;
        } else {
            j++;
        }
    }
    if (i < numprocs) {
            //table->mpi_sendcounts[i] = (table->raw_string_count - j_last) * rslen;
        i_sendcounts[i] = (table->raw_string_count - j_last) * rslen;
        i++;
    }
    while (i < numprocs - 1) {
            //table->mpi_sendcounts[i] = 0;
        i_sendcounts[i] = 0;
        i++;
    }
    
    table->mpi_sdispls[0] = 0;
    i_sdispls[0] = 0;
    for (i = 1; i < numprocs; i++) {
            //table->mpi_sdispls[i] =
            //  table->mpi_sdispls[i - 1] + table->mpi_sendcounts[i - 1];
        
        i_sdispls[i] = i_sdispls[i - 1] + i_sendcounts[i - 1];
        
    }
    
    
        //    MPI_Alltoall(table->mpi_sendcounts, 1, MPI_INT,
        //         table->mpi_recvcounts, 1, MPI_INT, table->comm);
    
    MPI_Alltoall(i_sendcounts, 1, MPI_LONG_LONG,
                 i_recvcounts, 1, MPI_LONG_LONG, table->comm);
    
        //table->mpi_rdispls[0] = 0;
    i_rdispls[0] = 0;
    
        //long long recvsum = table->mpi_recvcounts[0];
    long long recvsum = i_recvcounts[0];
    long long i_recvsum = i_recvcounts[0];
    
    
    for (i = 1; i < numprocs; i++) {
            //table->mpi_rdispls[i] = table->mpi_rdispls[i - 1] +
            //table->mpi_recvcounts[i - 1];
            //recvsum += table->mpi_recvcounts[i];
        
        i_rdispls[i] = i_rdispls[i - 1] + i_recvcounts[i - 1];
        i_recvsum += i_recvcounts[i];
    }
    
    recvsum = i_recvsum;
    
#ifdef DEBUG_PRINT
    if (recvsum > 1000000000 || recvsum < 0) {
        printf("[%d] recvsum= %lld\n", myid, recvsum);
        printf("\t[%d]splitter:", myid);
        for (i = 0; i < numprocs - 1; i++) {
            printf(" (%lld)", i);
            char *ptr_splitter = table->splitter_vector + i * rlen_bin;
            debug_print_bin_str(ptr_splitter, table->attr.rlen);
        }
        printf("\n");
        printf("[%d]recvcounts: ", myid);
        for (i = 0; i < numprocs; i++) {
            printf("(%lld)%d ", i, table->mpi_recvcounts[i]);
        }
        printf("\n");
    }
#endif
    
    
    /*
    printf ("\n");
    printf (" Rank %d - Orig_RecvCount: %lld New_RecvCount: %lld \n", \
            myid, recvsum, i_recvsum );
    
    printf ("RecvCounts: ID %d", myid);
    for (i = 0; i < numprocs; i++) {
        printf(" (%lld)%d (%lld)%lld ", \
               i, table->mpi_recvcounts[i], i, i_recvcounts[i]);
    }
    printf ("\n");
    
    printf ("RecvDisplacement: ID %d", myid);
    for (i = 0; i < numprocs; i++) {
        printf(" (%lld)%d (%lld)%lld ", \
               i, table->mpi_rdispls[i], i, i_rdispls[i]);
    }
    printf ("\n");
    
    printf ("Send Counts: ID %d", myid);
    for (i = 0; i < numprocs; i++) {
        printf(" (%lld)%d (%lld)%lld ", \
               i, table->mpi_sendcounts[i], i, i_sendcounts[i]);
    }
    printf ("\n");
    
    printf ("Send Displacement: ID %d", myid);
    for (i = 0; i < numprocs; i++) {
        printf(" (%lld)%d (%lld)%lld ", \
               i, table->mpi_sdispls[i], i, i_sdispls[i]);
    }
    printf ("\n");
    
    */
    
    
    table->mpi_recvbuf = (char *) kmi_malloc(recvsum * sizeof(char));
    
    
    /* Note: each process sending data of less than MAX_INT,
     * but the total sum of mpi_recvcounts[i] can be larger 
     * than MAX_INT. So "recvsum" is of type long long. */
/*
    MPI_Alltoallv(table->raw_string_buffer, table->mpi_sendcounts,
                  table->mpi_sdispls, MPI_CHAR, table->mpi_recvbuf,
                  table->mpi_recvcounts, table->mpi_rdispls, MPI_CHAR,
                  table->comm);
  
 */
    //  ---------------------------------------
    //  Replacing ALLtoALLV with P2P messaging
    //  ---------------------------------------
    
        //    printf (" Rank %d done with ALLTOALLV \n", myid);
    
    MPI_Barrier (table->comm);
    
    
        //char* i_recvbuf = (char *) kmi_malloc(i_recvsum * sizeof(char));
    
    
    int rank_itr;
	long long  rank_recv_itr, cur_recv_itr,tot_recv_itr;
	long long i_INT_MAX = INT_MAX;
    
    
    // Calculate the number of batch receives needed from each rank
    // A single receive will be of max size INT_MAX
    tot_recv_itr = 0;
	for (rank_itr = 0; rank_itr < numprocs; rank_itr++)
	{
        // Local copy and not over the network
        if (rank_itr == myid)
			continue;
        
		rank_recv_itr = i_recvcounts[rank_itr] / i_INT_MAX;
		if (i_recvcounts[rank_itr] % i_INT_MAX)
			rank_recv_itr++;
		
		tot_recv_itr += rank_recv_itr;
        
            //printf (" Rank %d <-- Rank %d : %lld Iterations %lld\n", myid, rank_itr, i_recvcounts[rank_itr], rank_recv_itr);
	}
    
    // Allocate memory for the recvs
	MPI_Request* request;
	MPI_Status* req_status;
    
    request = (MPI_Request*)malloc( tot_recv_itr * sizeof(MPI_Request));
	req_status = (MPI_Status*)malloc( tot_recv_itr * sizeof(MPI_Status));
    
    long long size_recv, recv_left;
	int cur_request = 0;
    
    // Pre-post the receives
    for (rank_itr = 0; rank_itr < numprocs; rank_itr++)
    {
        //Local copy and not over the network
        if (rank_itr == myid)
        	continue;
        
		rank_recv_itr = i_recvcounts[rank_itr] / i_INT_MAX;
        if (i_recvcounts[rank_itr] % i_INT_MAX)
            rank_recv_itr++;
		
		size_recv = 0;
		recv_left = i_recvcounts[rank_itr];
        
		for (cur_recv_itr = 0; cur_recv_itr < (rank_recv_itr-1); cur_recv_itr++)
		{
			MPI_Irecv (&table->mpi_recvbuf[i_rdispls[rank_itr] + size_recv], INT_MAX, MPI_CHAR, rank_itr, cur_recv_itr, table->comm, &request[cur_request]);
			cur_request++;
			size_recv += INT_MAX;
			recv_left -= INT_MAX;
		}
        
		MPI_Irecv (&table->mpi_recvbuf[i_rdispls[rank_itr] + size_recv], recv_left, MPI_CHAR, rank_itr, rank_recv_itr -1, table->comm, &request[cur_request]);
        
        cur_request++;
		
	}
    
    MPI_Barrier (table->comm);
    
    //printf (" Sending the Data - %d \n", myid);
    
    // Post all the Sends now
	long long rank_send_itr,cur_send_itr, cur_disp, size_left, size_sent;
	
	for (rank_itr = 0; rank_itr < numprocs; rank_itr++)
	{
        // Local copy and not over the network
        if (rank_itr == myid)
            continue;
        
		size_left = i_sendcounts[rank_itr];
		size_sent = 0;
        
		rank_send_itr = i_sendcounts[rank_itr] / i_INT_MAX;
        if (i_sendcounts[rank_itr] % i_INT_MAX)
            rank_send_itr++;
        
            //printf (" Rank %d --> Rank %d : %lld Iterations %lld\n", myid, rank_itr, i_sendcounts[rank_itr], rank_send_itr);
        
		for (cur_send_itr = 0; cur_send_itr < (rank_send_itr -1); cur_send_itr++)
		{
                //printf(" Rank %d Itr %lld Displ %lld\n", myid, cur_send_itr, i_sdispls[rank_itr] + size_sent );
            
			MPI_Send ((void*)table->raw_string_buffer + i_sdispls[rank_itr] + size_sent, INT_MAX, MPI_CHAR, rank_itr, cur_send_itr, table->comm);
			size_left -= INT_MAX;
			size_sent += INT_MAX;
		}
        
        // Send the Last Chunk
            //printf(" Rank %d Itr %lld Displ %lld\n", myid, rank_send_itr -1, i_sdispls[rank_itr] + size_sent );
        
		MPI_Send ((void*)table->raw_string_buffer + i_sdispls[rank_itr] + size_sent, size_left, MPI_CHAR, rank_itr, rank_send_itr -1 , table->comm);
	}
	

    // Wait for all recvs to finish
	MPI_Waitall (tot_recv_itr, request, req_status);
    
    
    
        //printf (" Rank %d Starting LC %lld %lld %lld %lld \n", \
            myid,i_rdispls[myid], i_sdispls[myid], i_recvcounts[myid], i_sendcounts[myid] );
    
    memcpy (&table->mpi_recvbuf[i_rdispls[myid]], (void*)table->raw_string_buffer + i_sdispls[myid],i_recvcounts[myid]);
        // &table->raw_string_buffer[i_sdispls[myid]], i_recvcounts[myid]);
    

        //    printf(" Rank %d - Done with local copy \n", myid);
    
    MPI_Barrier(table->comm);
    
    kmi_free(table->raw_string_buffer);
    
    if (i_sendcounts)
    {
        free (i_sendcounts);
        i_sendcounts = 0;
    }
    
    if (i_recvcounts)
    {
        free (i_recvcounts);
        i_recvcounts = 0;
    }
    
    if (i_sdispls)
    {
        free (i_sdispls);
        i_sdispls = 0;
    }
    if (i_rdispls)
    {
        free (i_rdispls);
        i_rdispls = 0;
    }
    
    if (request)
    {
        free (request);
        request = 0;
    }
    if (req_status)
    {
        free (req_status);
        req_status = 0;
    }
    
    
#ifdef KMI_PROFILING_ALLTOALL
    MPI_Barrier(table->comm);
    t4 = MPI_Wtime();
#endif

#ifdef DEBUG_PRINT
    if (myid == 1) {
        printf("\ton process %d, recvsum %lld: ", myid, recvsum);
        int print_len = MIN(recvsum, 5 * rslen);
        for (i = 0; i < print_len && i < recvsum; i += rslen) {
            KMI_raw_string_t *ptr_tmp = (void *) table->mpi_recvbuf + i;
            debug_print_bin_str(ptr_tmp->str, ptr_tmp->str_len);
            printf("(%d,%d,%d) ", ptr_tmp->count, ptr_tmp->rc_count,
                   ptr_tmp->str_len);
        }
        printf("...");
        if (recvsum > print_len) {
            for (i = recvsum - print_len; i < recvsum; i += rslen) {
                KMI_raw_string_t *ptr_tmp = (void *) table->mpi_recvbuf + i;
                printf("(%d,%d,%d) ", ptr_tmp->count, ptr_tmp->rc_count,
                       ptr_tmp->str_len);
                debug_print_bin_str(ptr_tmp->str, ptr_tmp->str_len);
            }
        }
        printf("\n");
        printf("\tsplitter:");
        for (i = 0; i < numprocs - 1; i++) {
            printf(" (%lld)", i);
            char *ptr_splitter = table->splitter_vector + i * rlen_bin;
            debug_print_bin_str(ptr_splitter, table->attr.rlen);
        }
        printf("\n");
    }
#endif
#ifdef KMI_CONFIG_INDEXED_ARRAY
    qsort(table->mpi_recvbuf, recvsum / rslen, rslen, KMI_compare_raw_string);
#ifdef DEBUG_PRINT
    if (0) {
        printf("\t==> after qsort, on process %d, recvsum %lld: ", myid,
               recvsum);
        int print_len = MIN(recvsum, 5 * rslen);
        for (i = 0; i < recvsum; i += rslen) {
            KMI_raw_string_t *ptr_tmp = (void *) table->mpi_recvbuf + i;
            debug_print_bin_str(ptr_tmp->str, ptr_tmp->str_len);
            printf("(%d,%d,%d) ", ptr_tmp->count, ptr_tmp->rc_count,
                   ptr_tmp->str_len);
        }
        printf("\n");
    }
#endif
    /* organize them into records */
    i = 0;
    j = rslen;
    KMI_raw_string_t *ptr_i = (void *) table->mpi_recvbuf + i;
    KMI_raw_string_t *ptr_j = (void *) table->mpi_recvbuf + j;
    int merge_flag = 0;
    for (; j < recvsum;) {
        int len = MAX(ptr_i->str_len, ptr_j->str_len);
        int rc = KMI_prefix_cmp(ptr_i->str, ptr_j->str, len);
        while (rc == 0 && j < recvsum) {
            ptr_i->count += ptr_j->count;
            ptr_i->rc_count += ptr_j->rc_count;
            j += rslen;
            ptr_j = (void *) table->mpi_recvbuf + j;
            len = MAX(ptr_i->str_len, ptr_j->str_len);
            rc = KMI_prefix_cmp(ptr_i->str, ptr_j->str, len);
            merge_flag = 1;
        }
        i += rslen;
        ptr_i = (void *) table->mpi_recvbuf + i;
        if (j > i && j < recvsum) {
            memcpy(ptr_i, ptr_j, rslen);
            merge_flag = 0;
        }
        j += rslen;
        ptr_j = (void *) table->mpi_recvbuf + j;
    }
    long long merged_len = i;
    if (merge_flag == 0)
        merged_len = i + rslen;
    if (recvsum == 0)
        merged_len = 0;         /* if no string on this process */
    KMI_sorted_array_init(table, table->mpi_recvbuf, merged_len);
#ifdef DEBUG_PRINT
    /* print the strings after re-distribution */
    if (myid == 0) {
        printf("\ton process %d, merged_len %lld, merged count: %lld, ", myid,
               merged_len, merged_len / rslen);
        int print_len = MIN(merged_len, 5 * rslen);
        for (i = 0; i < print_len && i < merged_len; i += rslen) {
            KMI_raw_string_t *ptr_tmp = (void *) table->mpi_recvbuf + i;
            debug_print_bin_str(ptr_tmp->str, ptr_tmp->str_len);
            printf("(%d,%d,%d,i=%lld) ", ptr_tmp->count, ptr_tmp->rc_count,
                   ptr_tmp->str_len, i);
        }
        printf("...");
        if (merged_len > print_len) {
            for (i = merged_len - print_len; i < merged_len; i += rslen) {
                KMI_raw_string_t *ptr_tmp = (void *) table->mpi_recvbuf + i;
                printf("(%d,%d,%d) ", ptr_tmp->count, ptr_tmp->rc_count,
                       ptr_tmp->str_len);
                debug_print_bin_str(ptr_tmp->str, ptr_tmp->str_len);
            }
        }
        printf("\n");
        print_len = 5;
        printf("\tsplitter:");
        for (i = 0; i < print_len && i < numprocs - 1; i++) {
            printf(" (%lld)", i);
            char *ptr_splitter = table->splitter_vector + i * rlen_bin;
            debug_print_bin_str(ptr_splitter, table->attr.rlen);
        }
        printf("...");
        if (numprocs - 1 > print_len) {
            for (i = numprocs - 1 - print_len; i < numprocs - 1; i++) {
                printf(" (%lld)", i);
                char *ptr_splitter = table->splitter_vector + i * rlen_bin;
                debug_print_bin_str(ptr_splitter, table->attr.rlen);
            }
        }
        printf("\n");
    }
#endif
#else
    /* insert to local hash table */
    for (i = 0; i < recvsum; i += rslen) {
        KMI_raw_string_t *ptr_tmp = (void *) table->mpi_recvbuf + i;
        KMI_string_type_t type;
        if (ptr_tmp->count == 1)
            type = KMI_STRING_ORIGINAL;
        else
            type = KMI_STRING_RC;
        _KMI_table_add_string(table, ptr_tmp->str, ptr_tmp->str_len, type);
    }
#endif
#ifdef KMI_PROFILING_ALLTOALL
    t5 = MPI_Wtime();
    time_localsort = t2 - t1;
    time_sampling = t3 - t2;
    time_alltoall = t4 - t3;
    time_insert = t5 - t4;
    time_commit = t5 - t1;
    long long local_num_string = recvsum / rslen;
    long long alltoall_buf_size = recvsum + (long long)table->raw_string_count * rslen;
    long long global_num_string;
    MPI_Allreduce(&local_num_string, &global_num_string, 1, MPI_LONG_LONG,
                  MPI_SUM, table->comm);
    double string_size =
        (double) global_num_string * rslen / 1024 / 1024 / 1024;
    double buf_size = (double) alltoall_buf_size / 1024 / 1024 / 1024;
    if (myid == 0)
        printf
            ("Profiling KMI_commit, time in seconds:\nOUTPUT:\t#str\t#str_g\tsize_g(GB)\tbuf_size(GB)\tlocalsort\tsampling\talltoallv\tinsert/sort\ttotal_commit\nOUTPUT:\t%lld\t%lld\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
             local_num_string, global_num_string, string_size, buf_size,
             time_localsort, time_sampling, time_alltoall, time_insert,
             time_commit);
#endif

    return rc;

}
#else
int KMI_table_commit_fixed_length(KMI_table_t table)
{
    int i;
    int myid = table->myid;
    int numprocs = table->numprocs;
    /* set sdispls and alloc send buffer, copy send data to send buffer */
    /* MAYBE: reduce redundance in the send data */
    int record_len = sizeof(KMI_string_type_t) + sizeof(int)
        + KMI_BIN_STR_LEN(table->attr.rlen);
    int sendsum = 0;
    for (i = 0; i < numprocs; i++) {
        table->mpi_sendcounts[i] = table->sendcounts[i] * record_len;
        sendsum += table->mpi_sendcounts[i];
    }
    table->mpi_sdispls[0] = 0;
    for (i = 1; i < numprocs; i++) {
        table->mpi_sdispls[i] =
            table->mpi_sdispls[i - 1] + table->mpi_sendcounts[i - 1];
    }
    table->mpi_sendbuf = (char *) kmi_malloc(sendsum * sizeof(char));
    for (i = 0; i < numprocs; i++) {
        memcpy(table->mpi_sendbuf + (table->mpi_sdispls[i]),
               table->sendbuf[i], table->mpi_sendcounts[i] * sizeof(char));
    }
    /* set recvcounts and rdispls */
    MPI_Alltoall(table->mpi_sendcounts, 1, MPI_INT,
                 table->mpi_recvcounts, 1, MPI_INT, table->comm);
    table->mpi_rdispls[0] = 0;
    long long recvsum = table->mpi_recvcounts[0];
    for (i = 1; i < numprocs; i++) {
        table->mpi_rdispls[i] = table->mpi_rdispls[i - 1] +
            table->mpi_recvcounts[i - 1];
        recvsum += table->mpi_recvcounts[i];
    }
    table->mpi_recvbuf = (char *) kmi_malloc(recvsum * sizeof(char));
    for (i = 0; i < numprocs; i++)
        if (table->sendcounts[i] != 0)
            kmi_free(table->sendbuf[i]);
    /* Note: each process sending data of less than MAX_INT,
     * but the total sum of mpi_recvcounts[i] can be larger 
     * than MAX_INT. So "recvsum" is of type long long. */
    MPI_Alltoallv(table->mpi_sendbuf, table->mpi_sendcounts,
                  table->mpi_sdispls, MPI_CHAR, table->mpi_recvbuf,
                  table->mpi_recvcounts, table->mpi_rdispls, MPI_CHAR,
                  table->comm);
    kmi_free(table->mpi_sendbuf);
    debug_print("[%d]recvbuf: %s\n", myid, table->mpi_recvbuf);

    /* insert the received string into local hash_table */
    char *ptr = table->mpi_recvbuf;
    for (i = 0; i < recvsum; i += record_len) {
        char *ptr_char = ptr;
        KMI_string_type_t *ptr_type = (KMI_string_type_t *) ptr_char;
        ptr_char += sizeof(KMI_string_type_t);
        int *ptr_int = (int *) ptr_char;
        ptr_char += sizeof(int);
        _KMI_table_add_string(table, ptr_char, *ptr_int, *ptr_type);
        ptr += record_len;
    }
    kmi_free(table->mpi_recvbuf);

    return 0;
}
#endif

/** Initialize the KMI table.
 * Because the string is internally stored in binary form. So it should
 * be make clear here that all size are the original size. But the string
 * is stored as binary.
 */
int KMI_table_init(KMI_db_t db, KMI_table_t *table,
                   KMI_table_attribute_t table_attr)
{
    int myid, numprocs, i;
    MPI_Comm_size(db->comm, &numprocs);
    MPI_Comm_rank(db->comm, &myid);
    debug_print("KMI_table_init, db = %p, myid = %d\n", db, myid);
    *table = (KMI_table_t) kmi_malloc(sizeof(struct KMI_table));
    (*table)->table_len = 0;
    (*table)->comm = db->comm;
    (*table)->db_index = db->db_index;
    (*table)->tb_index = db->db_num_local_tb;
    (*table)->db = db;
    (*table)->attr = table_attr;
    (*table)->myid = myid;
    (*table)->numprocs = numprocs;
#ifdef KMI_CONFIG_SORTING
    (*table)->raw_string_count = 0;
    (*table)->raw_string_buffer_allocated_count = 0;
    (*table)->raw_string_len = sizeof(KMI_raw_string_t)
        + KMI_BIN_STR_LEN(table_attr.rlen) - 1; /* flexible array member */
    int len_splitter = numprocs * KMI_BIN_STR_LEN(table_attr.rlen);     /* only (numprocs - 1) is used */
    (*table)->splitter_vector = (char *) kmi_malloc(len_splitter);
    memset((*table)->splitter_vector, 0, len_splitter);
#ifdef KMI_CONFIG_SAMPLING_ITERATION
    (*table)->splitter_flag = (int *) kmi_malloc(numprocs * sizeof(int));
    memset((*table)->splitter_flag, 0, numprocs * sizeof(int));
    (*table)->splitter_flag[numprocs - 1] = 1;
    (*table)->min_string =
        (char *) kmi_malloc(KMI_BIN_STR_LEN(table_attr.rlen));
    (*table)->max_string =
        (char *) kmi_malloc(KMI_BIN_STR_LEN(table_attr.rlen));
#endif
#else
    KMI_hash_table_init(&((*table)->htbl), table_attr.qlen, table_attr.rlen);
#endif
    db->tb[db->db_num_local_tb++] = (*table);
    debug_print("[%d]Add table to database, db->db_num_local_tb=%d\n",
                myid, db->db_num_local_tb);

    if (numprocs > 1) {
        (*table)->sendcounts = (int *) kmi_malloc(numprocs * sizeof(int));
        (*table)->sendbuf = (char **) kmi_malloc(numprocs * sizeof(char *));
        (*table)->sbuf_end = (char **) kmi_malloc(numprocs * sizeof(char *));
        (*table)->mpi_sendcounts = (int *) kmi_malloc(numprocs * sizeof(int));
        (*table)->mpi_sdispls = (int *) kmi_malloc(numprocs * sizeof(int));
        (*table)->mpi_recvcounts = (int *) kmi_malloc(numprocs * sizeof(int));
        (*table)->mpi_rdispls = (int *) kmi_malloc(numprocs * sizeof(int));
        for (i = 0; i < numprocs; i++) {
            (*table)->sendcounts[i] = 0;
            (*table)->mpi_sendcounts[i] = 0;
            (*table)->mpi_sdispls[i] = 0;
            (*table)->mpi_recvcounts[i] = 0;
            (*table)->mpi_rdispls[i] = 0;
        }
    }

    if (numprocs > 1)
        switch (table_attr.type) {
        case KMI_TABLE_TYPE_FIXED_LENGTH:
            KMI_table_init_fixed_length(*table);
            break;
        case KMI_TABLE_TYPE_VARIABLE_LENGTH:
            KMI_table_init_variable_length(*table);
            break;
        default:
            printf("This should not happen at KMI_table_init\n");
            break;
        }

    return 0;
}

int KMI_table_finalize(KMI_table_t tb)
{
    kmi_free(tb->sendcounts);
    kmi_free(tb->sendbuf);
    kmi_free(tb->sbuf_end);
    kmi_free(tb->mpi_sendcounts);
    kmi_free(tb->mpi_sdispls);
    kmi_free(tb->mpi_recvcounts);
    kmi_free(tb->mpi_rdispls);
#ifdef KMI_CONFIG_SORTING
    kmi_free(tb->splitter_vector);
#ifdef KMI_CONFIG_SAMPLING_ITERATION
    kmi_free(tb->splitter_flag);
    kmi_free(tb->min_string);
    kmi_free(tb->max_string);
#endif
#ifdef KMI_CONFIG_INDEXED_ARRAY
    KMI_sorted_array_destroy(tb);
#endif
#else
    KMI_hash_table_destroy(tb->htbl);
#endif
    kmi_free(tb);
}

/** Send all other processes a zero message to finish table operation
 */
int KMI_table_commit(KMI_table_t table)
{
    int myid = table->myid;
    int numprocs = table->numprocs;

    if (numprocs > 1)
        switch (table->attr.type) {
        case KMI_TABLE_TYPE_FIXED_LENGTH:
            KMI_table_commit_fixed_length(table);
            break;
        case KMI_TABLE_TYPE_VARIABLE_LENGTH:
            break;
        default:
            printf("This should not happen at KMI_table_commit\n");
            break;
        }

    return 0;
}

#ifdef KMI_CONFIG_SORTING
/** In KMI_CONFIG_SORTING configuration, strings added are all buffered 
 * until commit. Then strings will be sorted and re-distributed to 
 * different processors.
 */
int KMI_table_add_string(KMI_table_t table, char *str, long str_len)
{
    long long count = table->raw_string_count;
    const int rslen = table->raw_string_len;
    /* batch allocate buffer */
    if (count + 2 > table->raw_string_buffer_allocated_count)
    {
        if (table->raw_string_buffer_allocated_count == 0) {
            table->raw_string_buffer_allocated_count = KMI_BUF_RECORD_NUM_X2;
            table->raw_string_buffer = (KMI_raw_string_t *)kmi_malloc((size_t)table->raw_string_buffer_allocated_count * rslen);
        }
        else {
            table->raw_string_buffer_allocated_count = (int)(table->raw_string_buffer_allocated_count * KMI_RAW_BUFFER_ALLOCATION_GROWTH);
            table->raw_string_buffer = (KMI_raw_string_t *)kmi_realloc(table->raw_string_buffer, (size_t)table->raw_string_buffer_allocated_count * rslen);
        }
    }

    /* add original string to local buffer */
    char *str_bin;
    int str_bin_len;
    KMI_raw_string_t *ptr;
    ptr = (void *) (table->raw_string_buffer) + (size_t)count * rslen;
    ptr->str_len = str_len;
    ptr->count = 1;
    ptr->rc_count = 0;
    ptr->deleted = 0;
    KMI_str_ltr2bin(str, str_len, &str_bin, &str_bin_len);
    memcpy(ptr->str, str_bin, str_bin_len);
    kmi_free(str_bin);
    ptr = (void *) ptr + rslen;

    /* add rc string to local buffer */
    char *str_rc;
    char *str_rc_bin;
    int str_rc_bin_len;
    ptr->str_len = str_len;
    ptr->count = 0;
    ptr->rc_count = 1;
    ptr->deleted = 0;
    KMI_str_rc(str, str_len, &str_rc);
    KMI_str_ltr2bin(str_rc, str_len, &str_rc_bin, &str_rc_bin_len);
    memcpy(ptr->str, str_rc_bin, str_rc_bin_len);
    kmi_free(str_rc);
    kmi_free(str_rc_bin);
    table->raw_string_count += 2;

    return 0;
}

#else

int _KMI_table_add_string(KMI_table_t table, char *str, long str_len,
                          KMI_string_type_t type)
{
    int myid = table->myid;
    int numprocs = table->numprocs;

    int hash_key = KMI_keyspace_partition(str, str_len, numprocs);
    if (hash_key == myid) {
        debug_print
            ("[%d]_KMI_table_add_string local, table %p, db %d: %s, hash_key=%d\n",
             myid, table, table->db_index, str, hash_key);
        KMI_hash_table_insert(table->htbl, str, str_len, type);
    } else {
        /* Gather strings to be sent to other processes. Its type
         * is also send along with the string. One record is like
         * | KMI_string_type_t | string |.
         * The buffer is batch allocated, each time allocate 
         * KMI_BUF_RECORD_NUM records.
         * The length of each record can be different, just use the table->attr.rlen,
         * which is the max length of all reads. */
        int record_len = sizeof(KMI_string_type_t) + sizeof(int)
            + KMI_BIN_STR_LEN(table->attr.rlen);
        if ((table->sendcounts[hash_key] % KMI_BUF_RECORD_NUM) == 0) {
            if (table->sendcounts[hash_key] == 0) {
                table->sendbuf[hash_key] = (char *)
                    kmi_malloc(KMI_BUF_RECORD_NUM * record_len * sizeof(char));
                table->sbuf_end[hash_key] = table->sendbuf[hash_key];
            } else {
                long long  bufsize = (table->sendcounts[hash_key] + KMI_BUF_RECORD_NUM)
                    * record_len * sizeof(char);
                table->sendbuf[hash_key] =
                    kmi_realloc(table->sendbuf[hash_key], bufsize);
                table->sbuf_end[hash_key] = table->sendbuf[hash_key] +
                    (table->sendcounts[hash_key]) * record_len * sizeof(char);
            }
        }
        debug_print
            ("[%d]_KMI_table_add_string, memcpy %s to sendbuf[%d]\n", myid,
             str, hash_key);
        char *ptr_char = table->sbuf_end[hash_key];
        KMI_string_type_t *ptr_type = (KMI_string_type_t *) ptr_char;
        *ptr_type = type;
        ptr_char += sizeof(KMI_string_type_t);
        int *ptr_int = (int *) ptr_char;
        *ptr_int = str_len;
        ptr_char += sizeof(int);
        memcpy(ptr_char, str, KMI_BIN_STR_LEN(str_len));
        table->sbuf_end[hash_key] += record_len;
        table->sendcounts[hash_key]++;
    }

    return 0;
}

int KMI_table_add_string(KMI_table_t table, char *str, long str_len)
{
    int myid = table->myid;
    int numprocs = table->numprocs;

    /* Insert the original string */
#ifdef KMI_CONFIG_BINARY_STRING
    char *str_bin;
    int str_bin_len;
    KMI_str_ltr2bin(str, str_len, &str_bin, &str_bin_len);
    _KMI_table_add_string(table, str_bin, str_len, KMI_STRING_ORIGINAL);
    kmi_free(str_bin);
#else
    _KMI_table_add_string(table, str, str_len, KMI_STRING_ORIGINAL);
#endif

    /* Inserting the rc string */
    char *str_rc;
    KMI_str_rc(str, str_len, &str_rc);
    debug_print("[%d]KMI_table_add_string (rc) %s, str_len=%d\n",
                myid, str_rc, str_len);
#ifdef KMI_CONFIG_BINARY_STRING
    char *str_rc_bin;
    int str_rc_bin_len;
    KMI_str_ltr2bin(str_rc, str_len, &str_rc_bin, &str_rc_bin_len);
    _KMI_table_add_string(table, str_rc_bin, str_len, KMI_STRING_RC);
    kmi_free(str_rc);
    kmi_free(str_rc_bin);
#else
    _KMI_table_add_string(table, str_rc, str_len, KMI_STRING_RC);
    kmi_free(str_rc);
#endif

    return 0;
}
#endif
