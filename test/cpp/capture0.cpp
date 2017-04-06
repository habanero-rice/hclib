/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * DESC: capture a smart pointer (shared)
 */
#include <memory>
#include <iostream>
#include <unistd.h>
#include <cassert>

#include "hclib_cpp.h"

struct SimpleObject {
    int value;
    SimpleObject(): value(1) { }
    ~SimpleObject() { value = 0; }
};

int main (int argc, char ** argv) {
    hclib::launch([]() {
        hclib::finish([]() {{
            auto p = std::make_shared<SimpleObject>();
            std::cout << "Value starts as " << p->value << std::endl;
            hclib::async([=](){
                usleep(100000);
                std::cout << "Value in async is " << p->value << std::endl;
                assert(p->value == 1);
            });
            // p is dead
        }});
    });
    std::cout << "Exiting..." << std::endl;
    return 0;
}
