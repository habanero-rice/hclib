#ifndef _KMI_
#define _KMI_
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <pthread.h>
#include "mpi.h"


#define KMI_NAME                "K-mer Matching Interface"
#define KMI_VERSION             "0.01"
#define CHAR_SIZE               8
#define KMI_MAX_NUM_TABLE       4096
#define KMI_MAX_NUM_DB          1024
#define KMI_MAX_NUM_SEND_REQ    1024
#define KMI_HASH_INDEX_SIZE     12
#define KMI_BUF_RECORD_NUM      1024
#define KMI_BUF_RECORD_NUM_X2   (2 * KMI_BUF_RECORD_NUM)
#define KMI_MAX_STRING_LEN      1024
#define KMI_MAX_QUERY_LEN       256
#define KMI_QUERY_DAEMON_TAG    1000
#define KMI_QUERY_META_TAG      1001
#define KMI_QUERY_DATA_TAG      1002
#define KMI_IQUERY_TAG          1003
#define KMI_QUERY_RSTR_TAG      1004
#define KMI_RATIO_MAX           1.1
#define KMI_Datatype            MPI_Datatype
#define KMI_CHAR                MPI_CHAR
#define KMI_INT                 MPI_INT
#define KMI_MAGIC_CHAR          'Z'
#define KMI_ARRAY_INDEX_LEN     10
#define KMI_RAW_BUFFER_ALLOCATION_GROWTH 1.3

#define MIN(a,b) \
    ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
#define MAX(a,b) \
    ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

/* Macros for choosing binary form of string or the original string.
 * After use this macro, the code is cleaner, but this should be used
 * with care that the name is ALWAYS str_bin or str */
#ifdef KMI_CONFIG_BINARY_STRING
#define KMI_BIN_STR_LEN(x) (((x) * 2 + CHAR_SIZE -  1) / CHAR_SIZE)
#define KMI_STRING str_bin
#else
#define KMI_BIN_STR_LEN(x) x
#define KMI_STRING str
#endif

#ifdef DEBUG_PRINT
/* Memory wrapper fo bookkeeping.
 */
#define kmi_malloc(size) kmalloc(__FILE__, __LINE__, size)
#define kmi_free(ptr) kfree(__FILE__, __LINE__, ptr)
#define kmi_realloc(ptr, size) krealloc(__FILE__, __LINE__, ptr, size)
#else
#define kmi_malloc(size) malloc(size)
#define kmi_free(ptr) free(ptr)
#define kmi_realloc(ptr, size) realloc(ptr, size)
#endif


/** KMI_query_type_t defines all types of querys KMI have.
 */
typedef enum {
    KMI_QUERY_PREFIX,           /* query a prefix */
    KMI_QUERY_PREFIX_DELETE,    /* query a prefix & delete matching results */
    KMI_QUERY_DELETED_FLAG,     /* modify table->deleted_flag[] */
    KMI_QUERY_RANDOM_PICK,      /* randomly pick one string */
    KMI_QUERY_SUFFIX,           /* query a suffix */
    KMI_QUERY_ANY               /* the matching position can be anywhere */
} KMI_query_type_t;

/** The KMI database type.
 */
typedef enum {
    KMI_DB_TYPE_ALPHABET_NT,    /* alphabet nt */
    KMI_DB_TYPE_ALPHABET_NT5,   /* alphabet nt5 */
    KMI_DB_TYPE_ALPHABET_AA,    /* alphabet aa */
    KMI_DB_TYPE_PROTEIN         /* protein */
} KMI_db_type_t;

/** The KMI table type.
 */
typedef enum {
    KMI_TABLE_TYPE_FIXED_LENGTH,        /* fixed length records */
    KMI_TABLE_TYPE_FIXED_LENGTH_IMD,    /* fixed length records, start 
                                           distributing strings when added */
    KMI_TABLE_TYPE_VARIABLE_LENGTH,     /* variable length records */
} KMI_table_type_t;

/** The KMI string type.
 */
typedef enum {
    KMI_STRING_ORIGINAL,        /* the original string */
    KMI_STRING_RC,              /* the reverse complement string */
} KMI_string_type_t;

/** Attribute for a KMI string.
 * e.g. paired-end, reverse-complement
 */
typedef struct KMI_string_attribute {
} KMI_string_attribute_t;

/** Attribute of a KMI table.
 */
typedef struct KMI_table_attribute {
    KMI_table_type_t type;      /* table type */
    int rlen;                   /* read length */
    int qlen;                   /* query length */
} KMI_table_attribute_t;

/** One record of KMI_hash_table.
 */
struct KMI_hash_table_record {
    char *str;                  /* the string itself in binary form */
    int str_len;                /* note it is the original length of the string */
    int count;                  /* count of duplicates in the database */
    int rc_count;               /* count of its reverse string */
    struct KMI_hash_table_record *next; /* pointer to the next record */
};
typedef struct KMI_hash_table_record *KMI_hash_table_record_t;

/** Hash table for string storage and retrieve for each process.
 */
struct KMI_hash_table {
    int key_len;                /* length of the search key */
    int read_len;               /* length of the reads */
    int size;                   /* size of hash table */
    KMI_hash_table_record_t *list;
#ifdef KMI_PROFILING_HASHTABLE
    int max_chain_size;
#endif
};
typedef struct KMI_hash_table *KMI_hash_table_t;

typedef struct KMI_raw_string {
    int str_len;
    int count;
    int rc_count;
    int deleted;                /* whether it is deleted */
    char str[1];
} KMI_raw_string_t;

typedef struct KMI_dh_record {
    long long left;
    long long right;
} KMI_dh_record_t;

struct KMI_table;
struct KMI_db;
typedef struct KMI_table *KMI_table_t;
typedef struct KMI_db *KMI_db_t;

/** A KMI database has many tables. Each table contain many strings.
 * It is a local table, but has a global db_index. Which indicates
 * its connection with tables on other processes.
 */
struct KMI_table {
    int table_len;              /* number of strings */
    MPI_Comm comm;              /* MPI communicator */
    int db_index;               /* parent database index */
    int tb_index;               /* current table index */
    KMI_db_t db;                /* parent database */
    pthread_t recv_thread;      /* thread for receiving data */
    KMI_table_attribute_t attr; /* table attribute */
    int myid;
    int numprocs;
#ifdef KMI_CONFIG_SORTING
    long long  raw_string_count;       /* number of strings in commit buffer */
    int raw_string_buffer_allocated_count;
    int raw_string_len;         /* max length of KMI_raw_string_t */
    KMI_raw_string_t *raw_string_buffer;        /* buffer for commit */
    char *splitter_vector;      /* buffer for commit */
    char *min_string;           /* minimum string of all */
    char *max_string;           /* maximum string of all */
    int *splitter_flag;         /* flag for each pivot in splitter */
    long long *local_histo;     /* buffer for histogram sorting */
    long long *global_histo;    /* buffer for histogram sorting */
    char *s_array;              /* for sorted array */
    long long s_array_len;      /* for sorted array, array length */
    long long s_array_count;    /* for sorted array, record count, each record is 'raw_string_len' long */
    long long s_array_undel;    /* for sorted array, undeleted records count */
    long long s_array_undel_idx;        /* fisrt undeleted record index */
    int all_deleted;            /* 0 if there is still undelted records, 1 if all are deleted */
    int *deleted_flag;          /* indicate whether all the records on a process is deleted */
    int undel_rank_index;
    KMI_dh_record_t *dh;        /* direct hash for indexing sorted array */
    long long dh_len;           /* size of direct hash table */
    int dh_keylen;              /* key len of direct hash table */
    int dh_key_start;           /* the first different letter of all local strings */
    long long dh_left;          /* left position of direct hash table */
#else
    KMI_hash_table_t htbl;      /* hash table for data storage */
#endif
    int *sendcounts;            /* send counts for records */
    char **sendbuf;             /* send buffers for records */
    char **sbuf_end;            /* pointer for adding strings to sendbuf */
    char *mpi_sendbuf;
    int *mpi_sendcounts;
    int *mpi_sdispls;
    char *mpi_recvbuf;
    int *mpi_recvcounts;
    int *mpi_rdispls;
};

/** Used in KMI_query_db to indicate how much error can the query tolerate.
 * Each struct member indicate one kind of error.
 */
typedef struct KMI_error_tolerance {
    int sub;                    /* substitution */
    int ins;                    /* insertion */
    int del;                    /* deletion */
} KMI_error_tolerance_t;

/** Represent a query.
 * If the query result locate in a different process from the caller process,
 * the query is then transmitted to the process that owns the query results.
 */
typedef struct KMI_query {
    char str[KMI_MAX_QUERY_LEN];
    int str_len;
    int from_id;
    int db_index;
    int tb_index;
    KMI_query_type_t type;
    KMI_error_tolerance_t etlr;
} KMI_query_t;

typedef struct KMI_query_meta {
    int count;
    int len;
    int from_id;
} KMI_query_meta_t;

/** A KMI database. There may be multiple databases created by calling
 * KMI_db_dup(db1, db2) function. The low level data, however, has only
 * one copy. In this sense, a database is like a view of the underlying
 * data.
 */
struct KMI_db {
    int db_num_local_tb;        /* number of local tables */
    KMI_table_t tb[KMI_MAX_NUM_TABLE];  /* tables */
    MPI_Comm comm;              /* MPI communicator */
    int myid;
    int numprocs;
    int db_index;               /* index of current db, db of the same index
                                   belongs to the same distributed db */
    pthread_t daemon_thread;    /* thread for receiving data */
    int numfinish;              /* number of finished processes */
    char *mpi_recvbuf;          /* for query transit */
    int mpi_recvlen;            /* for query transit */
    MPI_Request mpi_recvreq;    /* for query transit */
    KMI_hash_table_record_t *list;      /* for query transit */
    int stage;                  /* for query transit */
    KMI_query_meta_t meta_send; /* for query transit */
    MPI_Request mpi_sendreq;    /* for query transit */
    KMI_query_t *query;         /* for query transit */
    char *databuf;              /* for query transit */
};

/** The attribute of a KMI database.
 */
typedef struct KMI_db_attribute {
} KMI_db_attribute_t;

/** Hint used when initializing the database.
 * These hint can help KMI to optimize the database structure.
 */
typedef struct KMI_hint {
    int KMI_hint_no_deletion;
    int KMI_hint_same_length_strings;
    int KMI_hint_no_reverse_complement;
    int KMI_hint_no_spaced_seeds;
    int KMI_hint_no_regex;
} KMI_hint_t;

/** KMI_request_t is used in non-blocking call KMI_iquery.
 */
typedef struct KMI_request {
    int hash_key;
    int myid;
    int stage;
    KMI_db_t db;
    KMI_hash_table_record_t **list;
    int *list_len;
    KMI_query_t query;
    KMI_query_meta_t meta_recv;
    MPI_Status recv_status;
    char *databuf;
    MPI_Request mpi_greq;
    MPI_Request mpi_recvreq;
#ifdef KMI_CONFIG_SORTING
    int req_count;
    int unfinished_count;       /* counting #results from different processors */
    int index;
    int low;
    int high;
    MPI_Request mpi_sendreq[KMI_MAX_NUM_SEND_REQ];
#else
    MPI_Request mpi_sendreq;
#endif
} KMI_request_t;

/** KMI_status_t is used in non-blocking call KMI_wait.
 */
typedef MPI_Status KMI_status_t;


/** Initialize KMI
 */
int KMI_init();

/** Finalize KMI
 */
int KMI_finalize();

/** Initialize KMI database. 
 * @output db
 * @input type
 * @input comm
 */
int KMI_db_init(KMI_db_t *db, KMI_db_type_t type, MPI_Comm comm);

/** Finalize KMI database.
 * @input db
 */
int KMI_db_finalize(KMI_db_t db);

/** Add an hint to a database to help optimization.
 */
int KMI_db_add_hint(KMI_db_t db, KMI_hint_t hint);

/** Add an attribute to a database.
 */
int KMI_db_add_attr(KMI_db_t db, KMI_db_attribute_t db_attr);

/** Create #num tables in database.
 */
int KMI_db_create_tables(KMI_db_t db, int num);

/** Add a table to a database.
 * Note this comm should be a subset of the comm in KMI_db_init.
 * @input table
 * @input comm
 * @output db
 * @output index
 */
int KMI_db_add_table(KMI_db_t db, KMI_table_t table, MPI_Comm comm,
                     int *table_index);

/** Get the number of tables in the database.
 */
int KMI_db_get_num_tables(KMI_db_t db, int *num);

/** Duplicate a KMI database. This db is more like a view of the real data, 
 * should use something like smart pointer to avoid duplication of real data
 * @input db1
 * @input/output db2
 */
int KMI_db_dup(KMI_db_t db1, KMI_db_t db2);

/** Initialize a table.
 * @input tb_attr
 * @input/output table
 * @input db
 */
int KMI_table_init(KMI_db_t db, KMI_table_t *table,
                   KMI_table_attribute_t table_attr);

/** Finalize a table.
 * @input tb
 */
int KMI_table_finalize(KMI_table_t tb);

/** Compare two KMI_raw_string_t strings.
 */
int KMI_compare_raw_string(const void *a, const void *b);

/** Commit a table. All tables sync and stop receiving new strings.
 */
int KMI_table_commit(KMI_table_t table);

/** Add an attribute to a table.
 * @input table_attr
 * @output table
 */
int KMI_table_add_attr(KMI_table_t table, KMI_table_attribute_t table_attr);

/** Adding string to a table.
 * @input str: a char array of DNA
 * @input str_len: the length of string
 * @output table: which KMI_table to add string
 */
int KMI_table_add_string(KMI_table_t table, char *str, long str_len);

/** Add an attribute to a string.
 */
int KMI_table_add_string_attr(KMI_table_t table, char *str, long str_len,
                              KMI_string_attribute_t str_attr);

/** Get string attribute.
 */
int KMI_table_get_string_attr(KMI_table_t table,
                              KMI_string_attribute_t *str_attr);

/** Commit KMI database change. 
 * Must be called once before any query.
 * Can be called repeatedly to redistribute the data when more strings are 
 * added.
 */
int KMI_db_commit(KMI_db_t db);

/** Start query daemon.
 */
int KMI_query_db_start(KMI_db_t db);

/** End query daemon.
 */
int KMI_query_db_finish(KMI_db_t db);

/** Query one table of the database and return a list of records containing 
 * the result.
 * @output list: the output array
 * @output list_len: length of the output array
 * @output: return 1 if found, 0 if not, negative if error
 */
int KMI_query_table(KMI_table_t tb, char *str, long str_len,
                    KMI_query_type_t type,
                    KMI_error_tolerance_t etlr,
                    KMI_hash_table_record_t **list, int *list_len);

/** Delete the query result list.
 */
int KMI_query_delete_list(KMI_hash_table_record_t *list, int list_len);

/** These functions are shared internally in KMI and should not
 * be called by user directly.
 */
int _KMI_query_table_prefix(KMI_table_t tb, char *str, long str_len,
                            KMI_error_tolerance_t etlr,
                            KMI_hash_table_record_t **list, int *list_len);
int _KMI_query_db_check(KMI_db_t db);

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
                       KMI_hash_table_record_t **list, int *list_len);

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
                                    KMI_hash_table_record_t **list, int *list_len, int **offsets);

/** Query the database and return a KMI table containing the result.
 * @output matched_table
 */
int KMI_query_db(KMI_db_t db, char *str, int str_len,
                 KMI_query_type_t type,
                 KMI_error_tolerance_t etlr,
                 KMI_hash_table_record_t **list, int *list_len);

/** Query the database and return a KMI table containing the result.
 * @output matched_table
 */
int KMI_query_db_bin(KMI_db_t db, char *str_bin, int str_len,
                     KMI_query_type_t type,
                     KMI_error_tolerance_t etlr,
                     KMI_hash_table_record_t **list, int *list_len);

/** Get the count of all records in local table.
 */
long long KMI_get_local_total_count(KMI_table_t tb);

/** Get the count of undeleted records in local table.
 */
long long KMI_get_local_count(KMI_table_t tb);

/** Check the determination condition: whether all records in all
 * tables are deleted: return 0 if not, 1 if yes.
 */
int KMI_check_all_deleted(KMI_table_t tb);

int KMI_query_db_random_pick(KMI_table_t tb, char *str, int *str_len);

int KMI_query_db_random_pick_local(KMI_table_t tb, char *str, int *str_len);

/** Non-blocking query the database and return a KMI table containing the
 * result.
 */
int KMI_iquery_db(KMI_db_t db, char *str, long str_len,
                  KMI_query_type_t type,
                  KMI_error_tolerance_t etlr,
                  KMI_request_t *request,
                  KMI_hash_table_record_t **list, int *list_len);

int KMI_test(KMI_request_t *request, int *flag, KMI_status_t *status);
int KMI_get_count(KMI_status_t *status, KMI_Datatype datatype, int *count);
int KMI_testany(int count, KMI_request_t array_of_requests[], int *indx,
                int *flag, KMI_status_t *status);
int KMI_wait(KMI_request_t *request, KMI_status_t *status);

/** Batch query.
 */
KMI_table_t KMI_bquery(KMI_db_t db, char *str, long str_len,
                       KMI_query_type_t type, KMI_error_tolerance_t etlr);

/** Query a string and delete from database.
 */
KMI_table_t KMI_query_and_delete(KMI_db_t db, char *str, long str_len,
                                 KMI_query_type_t type,
                                 KMI_error_tolerance_t etlr);

/** Query reverse complement.
 * KMI_query_db() deals with paired-end, KMI_query_reverse_complement() deals 
 * with reverse_complement.
 */
KMI_table_t KMI_query_reverse_complement(KMI_db_t db, char *str,
                                         long str_len,
                                         KMI_query_type_t type,
                                         KMI_error_tolerance_t etlr);

int KMI_table_get_num_strings(KMI_table_t table, int *num);

/** Get attribute of a table.
 */
int KMI_table_get_attr(KMI_table_t table);

/** Get the length of a string.
 * @output len: the length of string.
 */
int KMI_table_get_string_length(KMI_table_t table, int str_index, int *len);

/** Return a long string containing all the strings on a table, with each string
 * indexed by *displs.
 *
 * The query for strings take two steps: first, it returns a table containing
 * all the strings; second, retrieve all the strings of this table. As the table
 * returned by KMI_query_db may be too big, so it should be a handle in that 
 * case and the actual strings are only returned when this function is called 
 *
 * @input table: the KMI_table containing the strings
 * @output str: the whole long string
 * @output num_str: total number of substrings returned
 * @output str_len: lengths of each substring
 * @output displs: displacements of each substring
 */
int KMI_table_get_string(KMI_table_t table, char *str, int num_str,
                         long *str_len, long *displs);

/** Return reverse-complement strings on a table.
 */
int KMI_table_get_rc_string(KMI_table_t table, char *str, int num_str,
                            long *str_len, long *displs);

/** Return paired-end strings on a table.
 */
int KMI_table_get_paired_string(KMI_table_t table, char *str,
                                int num_str, long *str_len, long *displs);

/** Convert nucleotide sequence from letter to integer representation.
 */
char KMI_str_int2nt(int x);

/** Convert a char* array to a long long integer.
 */
long long KMI_char2longlong(char *str, int str_len);

/** Convert a char* array to a long integer.
 */
long KMI_char2long(char *str, int str_len);

/** Convert nucleotide sequence from letter to xor integer representation.
 * A XOR T = 01
 * C XOR G = 00
 */
int KMI_str_xor_nt2int(char c);

/** Convert nucleotide sequence from letter to integer representation.
 */
int KMI_str_nt2int(char str);

/** Convert a string to a more compact binary form.
 * Use 2 bits for each base letter, {0,1,2,3} for {A,C,G,T}.
 * Another additional 8 bits is used for adjacent edges.
 */
int KMI_str_ltr2bin(char *str, int str_len, char **str_bin, int *str_bin_len);

/** Convert a string to a more compact binary form.
 * Use 2 bits for each base letter, {0,1,2,3} for {A,C,G,T}.
 */
void KMI_str2binchar(const char * str, int str_len, char * str_bin);

/** Convert a binary form string back to letters.
 * Note str_len here is the original length of string.
 */
int KMI_str_bin2ltr(char *str_bin, long str_len, char **str);
int KMI_str_bin2ltr2(char *str_bin, long str_len, char *str);

/* Return a reverse complement letter */
char KMI_str_letter_rc(char c);

/** Hash function for key space partition.
 * Choosed when KMI_CONFIG_SORTING is not defined.
 */
int KMI_keyspace_partition(char *str, long str_len, int numprocs);

/** Splitter function for key space partition.
 * Choosed when KMI_CONFIG_SORTING is defined.
 */
int KMI_keyspace_splitter(KMI_table_t tb, char *str, long str_len,
                          int numprocs, int *low, int *high);

/** Store local strings as a sorted array.
 * Choosed when both KMI_CONFIG_SORTING and KMI_CONFIG_INDEXED_ARRAY 
 * are defined.
 */
int KMI_sorted_array_init(KMI_table_t table, char *old_array,
                          long long array_len);

/** Destroy the sorted array.
 */
int KMI_sorted_array_destroy(KMI_table_t table);

/** Search sorted array.
 */
int KMI_sorted_array_search(KMI_table_t table, char *str, long str_len, int del,
                            KMI_hash_table_record_t **list, int *list_len);

/** Initialize a hash table record.
 * @input/output record
 * @input str
 * @input str_len
 */
int KMI_hash_table_record_init(KMI_hash_table_record_t *record, char *str,
                               int str_len);

/** Delete a hash table record.
 */
int KMI_hash_table_record_delete(KMI_hash_table_record_t record);

/** Initialize a hash table.
 * @input key_len: the length of the search key, counted in char before 
 * converted to compact form.
 * @input read_len: the length of read
 * @input/output htbl: the hash table.
 */
int KMI_hash_table_init(KMI_hash_table_t *htbl, int key_len, int read_len);

/** Print the strings at index. For debug use.
 */
int Test_KMI_hash_table_print_index(KMI_hash_table_t htbl, int index);

/** Search an item in the hash table and return an array of results.
 * It is required that 'len' is initialized to zero when starting a
 * new search. Because we will use KMI_hash_table_search in multiple
 * tables to return a single list, the first call of this function
 * should do the initialization, the other calls will add records to
 * the list.
 * @input htbl
 * @input str
 * @output list
 * @output list_len
 */
int KMI_hash_table_search(KMI_hash_table_t htbl, char *str,
                          KMI_hash_table_record_t **list, int *list_len);

/** Insert an item into the hash table.
 * @input htbl
 * @input str
 * @input str_len
 * @input type
 */
int KMI_hash_table_insert(KMI_hash_table_t htbl, char *str,
                          int str_len, KMI_string_type_t type);

/** Search an item in the hash table.
 */
int Test_KMI_hash_table_search_index(KMI_hash_table_t htbl, char *str,
                                     int *index);

/** Insert a read into the hash table.
 * The read is stored in a KMI_hash_table_record, and the first key_len 
 * letters are used as the hash table key for search.
 */
int Test_KMI_hash_table_insert(KMI_hash_table_t htbl, char *str);

/** Delete an item in the hash table.
 * @input htbl
 * @input str
 */
int KMI_hash_table_delete(KMI_hash_table_t htbl, char *str);

/** Destroy the hash table.
 * @input/output htbl
 */
int KMI_hash_table_destroy(KMI_hash_table_t htbl);

/* Memory wrapper fo bookkeeping.
 */
void *kmalloc(const char *file, int line, size_t size);
void kfree(const char *file, int line, void *ptr);
void *krealloc(const char *file, int line, void *ptr, size_t size);

/* Large binary division, add, and subtraction.
 */
int KMI_binary_cmp(char *s1, char *s2, long len);
int KMI_binary_add(char *s1, char *s2, long str_len, char *s3);
int KMI_binary_sub(char *s1, char *s2, long str_len, char *s3);
int KMI_binary_div(char *s1, long len, int num, char *s2);

/** For debug use. Debugging printf.
 */
#if defined(DEBUG_PRINT) && defined(__GNUC__)
void debug_print(char *format, ...);
#else
#define debug_print(format, args...) ((void)0)
#endif
#if defined(DEBUG_PRINT) && defined(__GNUC__)
void debug_print_color(char *format, ...);
#else
#define debug_print_color(format, args...) ((void)0)
#endif
/** For debug use. Print the result of a query.
 * @output list
 * @output list_len
 * @input rc
 * @input myid
 * @input query
 */
void KMI_print_query_list(KMI_hash_table_record_t *list, int list_len, int rc,
                          int myid, char *query);
void debug_print_bin_num(char *str_bin, long str_len);
void debug_print_bin_str(char *str_bin, long str_len);


/** Global Data Structures for database.
 */
KMI_db_t g_db_all[KMI_MAX_NUM_DB];      /* all the databases.  */
int g_db_num;                   /* number of databases */

#endif                          /* _KMI_ */
