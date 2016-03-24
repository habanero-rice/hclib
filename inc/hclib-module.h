#ifndef HCLIB_MODULE_H
#define HCLIB_MODULE_H

#define HCLIB_MODULE_INITIALIZATION_FUNC(module_init_funcname) static int module_init_funcname()
#define HCLIB_REGISTER_MODULE(module_init_func) const static int ____hclib_register = module_init_func();

#endif
