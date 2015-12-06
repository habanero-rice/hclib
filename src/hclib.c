#include <assert.h>
#include <string.h>

#include "hclib.h"
#include "hcpp-rt.h"
#include "hcpp-task.h"
#include "hcpp-asyncStruct.h"

#ifdef __cplusplus
extern "C" {
#endif

void hclib_async(generic_framePtr fp, void *arg, struct ddf_st ** ddf_list,
        struct _phased_t * phased_clause, int property) {
    assert(property == 0);
    assert(ddf_list == NULL);
    assert(phased_clause == NULL);

    task_t *task = (task_t *)malloc(sizeof(task_t));
    task->_fp = fp;
    task->is_asyncAnyType = 0;
    task->ddf_list = NULL;
    memcpy(&task->_args, &arg, sizeof(arg));

    spawn(task);
}

#ifdef __cplusplus
}
#endif
