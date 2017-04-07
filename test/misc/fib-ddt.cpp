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

#include "hclib.hpp"
#include <iostream>
using namespace std;

static int threshold = 2;

int fib_serial(int n) {
    if (n <= 2) return 1;
    return fib_serial(n-1) + fib_serial(n-2);
}

void fib(int n, hclib::promise_t<int>* res) {
  int r;
  if (n <= threshold) {
    r = fib_serial(n);
    res->put(r);
    return;
  } 

  // compute f1 asynchronously
  hclib::promise_t<int>* f1 = new hclib::promise_t<int>();
  hclib::async([=]() { 
    fib(n - 1, f1);
  });

  // compute f2 serially (f1 is done asynchronously).
  hclib::promise_t<int>* f2 = new hclib::promise_t<int>();
  hclib::async([=]() { 
    fib(n - 2, f2);
  });

  // wait for dependences, before updating the result
  hclib::async_await([=] {
    int r = f1->get_future()->get() + f2->get_future()->get();
    res->put(r);
  }, f1->get_future(), f2->get_future());
}

int main(int argc, char** argv) {
    hclib::launch([=]() {
        int n = argc == 1 ? 30 : atoi(argv[1]);
        threshold = argc == 2 ? 10 : atoi(argv[2]);
        hclib::promise_t<int>* promise = new hclib::promise_t<int>();
        HCLIB_FINISH {
            fib(n, promise);
        }
        int res = promise->get_future()->get();
        cout << "Fib(" << n << ") = " << res << endl;
    });
    return 0;
}
