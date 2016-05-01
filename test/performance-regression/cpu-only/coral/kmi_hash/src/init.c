#include "kmi.h"

int KMI_init()
{
    g_db_num = 0;
}

int KMI_finalize()
{
    /* delete all databases */
    int i;
    for (i = 0; i < g_db_num; i++) {
        KMI_db_finalize(g_db_all[i]);
    }
}
