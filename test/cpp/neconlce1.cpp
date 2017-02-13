#include "hclib_cpp.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] {
        hclib::finish([] {

            int numWorkers = hclib::get_num_workers();
            cout << "Total Workers: " << numWorkers << endl;

            int num_locales = hclib::get_num_locales();
            hclib::locale_t *locales = hclib::get_all_locales();

            for (int i = 0; i < num_locales; i++) {
                hclib::locale_t *locale = locales + i;

                hclib::async_at([=] {
                    cerr << "Hello I'm Worker " << hclib::get_current_worker() << " of " << numWorkers << " workers" << endl;
                }, locale);
            }
        });
    });

    return 0;
}
