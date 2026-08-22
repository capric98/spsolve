// Minimal SuiteSparse_config allocation surface required by vendored AMD.
// The full SuiteSparse configuration library is intentionally not linked.

#include "SuiteSparse_config.h"

#include <stdint.h>
#include <stdlib.h>

int (*SuiteSparse_config_printf_func_get(void))(const char *, ...)
{
    return NULL;
}

void *SuiteSparse_malloc(size_t nitems, size_t size_of_item)
{
    if (nitems == 0) nitems = 1;
    if (size_of_item == 0) size_of_item = 1;
    if (nitems > SIZE_MAX / size_of_item) return NULL;
    return malloc(nitems * size_of_item);
}

void *SuiteSparse_free(void *p)
{
    free(p);
    return NULL;
}
