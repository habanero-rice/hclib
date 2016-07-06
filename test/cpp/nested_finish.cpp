#include <hclib_cpp.h>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] {
        for (int iter = 0; iter < 100; iter++) {
            hclib::async([] {
                hclib::finish([] {
                    hclib::async([] {
                        hclib::finish([] {
                            hclib::async([] {
                                hclib::finish([] {
                                    hclib::async([] {
                                        hclib::finish([] {
                                            hclib::async([] {
                                                printf("Howdy from inside a finish within nested finishes\n");
                                            });
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        }
    });
    return 0;
}
