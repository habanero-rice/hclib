#include "hclib_cpp.h"
#include "hclib-place.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    hclib::launch(&argc, argv, [] {
        hclib::finish([] {

            int numWorkers = hclib::num_workers();
            cout << "Total Workers: " << numWorkers << endl;

            int num_locales = hclib::get_num_locales();
            hclib::hclib_locale *locales = hclib::get_all_locales();

            for (int i = 0; i < num_locales; i++) {
                hclib::hclib_locale *locale = locales + i;

                hclib::async_at(locale, [=] {
                    cerr << "Hello I'm Worker " << hclib::get_current_worker() << " of " << numWorkers << " workers" << endl;
                });
            }
        });
    });

    return 0;
}
