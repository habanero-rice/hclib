#include <iostream>

#include "hclib_cpp.h"

int main(int argc, char **argv) {
    hclib::launch(&argc, argv, []() {
        hclib::place_t *my_pl = hclib::get_current_place();
        hclib::place_t *root_pl = hclib::get_root_place();

        int num_toplevel;
        hclib::place_t **toplevel = hclib::get_children_of_place(root_pl,
                &num_toplevel);
        for (int i = 0; i < num_toplevel; i++) {
            if (toplevel[i]->type == NVGPU_PLACE) {
                std::cout << "Found GPU place" << std::endl;
                void *d_ptr = hclib::allocate_at(toplevel[i], 10, 0);
            }
        }
    });
    return 0;
}
