/* Copyright (c) 2013, Rice University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
3.  Neither the name of Rice University
     nor the names of its contributors may be used to endorse or
     promote products derived from this software without specific
     prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] {
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
