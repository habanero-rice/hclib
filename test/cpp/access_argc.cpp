#include <hclib_cpp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [=] {
        printf("I see argc = %d, argv contains %s\n", argc, argv[0]);
        assert(argc == 1);
        assert(strcmp(argv[0], "./access_argc") == 0);
        printf("Check results: OK\n");
    });
    return 0;
}
