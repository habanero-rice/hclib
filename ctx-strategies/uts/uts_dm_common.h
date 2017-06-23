#ifndef _UTS_DM_COMMON_H
#define _UTS_DM_COMMON_H

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

/* Search status */
#define STATUS_HAVEWORK 0
#define STATUS_TERM     1

void        ss_abort(int error);
void        ss_error(char *str, int error);
int         ss_get_thread_num();
int         ss_get_num_threads();
char*       ss_get_par_description();

#endif
