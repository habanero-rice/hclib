#include "kmi.h"

#if defined(DEBUG_PRINT) && defined(__GNUC__)
void debug_print(char *format, ...)
{
#ifdef DEBUG_PRINT
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
#else
#endif                          /* DEBUG_PRINT */
}
#else
#endif                          /* DEBUG_PRINT && __GNUC__ */

#if defined(DEBUG_PRINT) && defined(__GNUC__)
void debug_print_color(char *format, ...)
{
#ifdef DEBUG_PRINT
    va_list args;
    va_start(args, format);
    fprintf(stderr, "%c[%d;%dm", 27, 1, 33);
    vfprintf(stderr, format, args);
    fprintf(stderr, "%c[%dm", 27, 0);
    va_end(args);
#else
#endif                          /* DEBUG_PRINT */
}
#else
#endif                          /* DEBUG_PRINT && __GNUC__ */

void printf_color()
{
    printf("%c[%d;%dm", 27, 1, 33);
    printf("color print\n");
    printf("%c[%dm", 27, 0);
}

void KMI_print_query_list(KMI_hash_table_record_t *list, int list_len, int rc,
                          int myid, char *query)
{
#ifdef KMI_CONFIG_BINARY_STRING
    if (rc == 1) {
        printf("[%d]%s found, %d record(s) match:\n", myid, query, list_len);
        int i;
        for (i = 0; i < list_len; i++) {
            char *str;
            KMI_str_bin2ltr(list[i]->str, list[i]->str_len, &str);
            printf("\t%.*s, count:%d, rc_count:%d, str_len:%d\n",
                   list[i]->str_len, str,
                   list[i]->count, list[i]->rc_count, list[i]->str_len);
            kmi_free(str);
        }
    } else
        printf("[%d]%s not found\n", myid, query);
#else
    if (rc == 1) {
        printf("[%d]%s found, %d record(s) match:\n", myid, query, list_len);
        int i;
        for (i = 0; i < list_len; i++)
            printf("\t%.*s, count:%d, rc_count:%d, str_len:%d\n",
                   list[i]->str_len, list[i]->str,
                   list[i]->count, list[i]->rc_count, list[i]->str_len);
    } else
        printf("[%d]%s not found\n", myid, query);
#endif
}
