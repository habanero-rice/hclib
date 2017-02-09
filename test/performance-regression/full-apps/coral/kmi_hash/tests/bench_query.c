#include "kmi.h"

#define STRING_LEN              100
#define QUERY_LEN               20
#define DEFAULT_NUM_STRING      2048
#define DEFAULT_NUM_QUERIES     2048

int numstrings;
int numqueries;
int str_len;
int query_len;

void print_usage()
{
    int myid, numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0)
        printf("Benchmark of the partition time.\n"
               "Usage:\n"
               "  mpiexec -n <num processes> bench_query [options]\n"
               "Options:\n"
               "  h|? : this message\n"
               "  n   : the number of strings each processor generates (Default %d)\n"
               "  m   : the number of quries each processor issues(Default %d)\n"
               "  l   : read length (Default %d)\n"
               "  q   : query length (Default %d)\n"
               "Example:\n"
               "  mpiexec -n 4 bench_query -n 2048\n"
               "  mpiexec -n 4 bench_query -n 2048 -m 2048\n",
               DEFAULT_NUM_STRING, DEFAULT_NUM_QUERIES, STRING_LEN, QUERY_LEN);
}

void get_options(int argc, char **argv)
{
    int c, err = 0;
    extern char *optarg;
    while ((c = getopt(argc, argv, "?hn:l:q:m:")) != -1)
        switch (c) {
        case 'h':
        case '?':
            print_usage();
            exit(EXIT_SUCCESS);
            break;
        case 'n':
            numstrings = atoi(optarg);
            break;
        case 'm':
            numqueries = atoi(optarg);
            break;
        case 'l':
            str_len = atoi(optarg);
            break;
        case 'q':
            query_len = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Unrecognized option\n");
            exit(EXIT_FAILURE);
            break;
        }
}

inline void generate_random_string(char *str, int len)
{
    int r;
    unsigned int x;
    int i;
    for (i = 0; i < len; i++) {
        r = rand();
        x = 0x00018000 & r;
        x = x >> 15;
        if (x == 0)
            str[i] = 'A';
        else if (x == 1)
            str[i] = 'C';
        else if (x == 2)
            str[i] = 'G';
        else
            str[i] = 'T';
    }
}

void generate_random_queries(char * query, char * buf, int num)
{
	int i;
    for (i = 0; i < num; i++) {
        generate_random_string(buf, query_len);
        KMI_str2binchar(buf, query_len, query);
        query += KMI_BIN_STR_LEN(query_len);
	}
}

// Returns hit count
long long query_loop(char *query, KMI_db_t db, int myid)
{
    KMI_error_tolerance_t error_tlr = {.sub = 0,.ins = 0,.del = 0 };
    KMI_hash_table_record_t *list;
	int i;
	int len;
    long long hit_count = 0;
    for (i = 0; i < numqueries; i++) {
#ifdef DEBUG_PRINT
        if (myid == 0) {
            if (i % 10000 == 0) {
                debug_t2 = MPI_Wtime();
                printf("query count: %d, time: %f\n", i, debug_t2 - debug_t1);
                debug_t1 = debug_t2;
            }
        }
#endif
        len = 0;
        KMI_query_db_bin(db, query, query_len, KMI_QUERY_PREFIX, error_tlr, &list,
                     &len);

        /* user code */

        if (len > 0) {
			//KMI_print_query_list(list, len, 1, myid, query);
            hit_count++;
            KMI_query_delete_list(list, len);
        }
       
        query += KMI_BIN_STR_LEN(query_len);
    }
    
    return hit_count;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    str_len = STRING_LEN;
    query_len = QUERY_LEN;
    numstrings = DEFAULT_NUM_STRING;
    numqueries = DEFAULT_NUM_QUERIES;
    if (argc > 1)
        get_options(argc, argv);
    KMI_init();

    /* initialize the database and add strings */
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, numprocs;
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);
    KMI_db_t db;
    KMI_db_init(&db, KMI_DB_TYPE_ALPHABET_NT, comm);
    KMI_table_t tb;
    KMI_table_attribute_t tb_attr = {
        .type = KMI_TABLE_TYPE_FIXED_LENGTH,
        .rlen = str_len,        /* use the max length of all reads */
        .qlen = query_len
    };
    KMI_table_init(db, &tb, tb_attr);

    double t0 = MPI_Wtime();
    /* parallel add strings to the table */
    char *str;
    str = (char *) malloc(str_len * sizeof(char));
    srand((int) time(NULL) + myid);
    int i;
    for (i = 0; i < numstrings; i++) {
        generate_random_string(str, str_len);
        KMI_table_add_string(tb, str, str_len);
    }
    free(str);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    KMI_table_commit(tb);
    KMI_db_commit(db);
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    char *query;
    query = (char *) malloc(KMI_BIN_STR_LEN(query_len) * sizeof(char) * numqueries);
    char *buf;
    buf = (char *)malloc(query_len * sizeof(char));
    generate_random_queries(query, buf, numqueries);

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

#ifdef DEBUG_PRINT
    double debug_t1, debug_t2;
    debug_t1 = t2;
#endif
    KMI_query_db_start(db);
    
    long long hit_count = query_loop(query, db, myid);
    
    KMI_query_db_finish(db);
    double t4 = MPI_Wtime();

    free(query);

    double time_add = t1 - t0;
    double time_commit = t2 - t1;
    double time_generate_query = t3 - t2;
    double time_query = t4 - t3;

    long long total_hit_count;
    MPI_Reduce(&hit_count, &total_hit_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, comm);

    if (myid == 0) {
        printf("Hit count: %lld\n"
               "Adding time cost (s): %f\n"
               "Commit time cost (s): %f\n"
               "Generate query time cost (s): %f\n"
               "Query time cost (s): %f\n"
               "Query throughput (global #queries/s): %.1f\n",
               total_hit_count, time_add, time_commit, time_generate_query, time_query,
               (double) numqueries * numprocs / time_query);
    }

    KMI_finalize();
    MPI_Finalize();
    return 0;
}
