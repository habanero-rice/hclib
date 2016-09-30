/* Copyright (c) 2015, Rice University

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

/*
 * hclib-forasync.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Ported from hclib
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCLIB_FORASYNC_H_
#define HCLIB_FORASYNC_H_

#include "hclib.h"
#include "hclib-task.h"

/*
 * Forasync mode to perform static chunking of the iteration space.
 */
#define FORASYNC_MODE_RECURSIVE 1
/*
 * Forasync mode to perform static chunking of the iteration space.
 */
#define FORASYNC_MODE_FLAT 0

namespace hclib {

inline int default_tile_size(const int n, const int nchunks) {
    return (n + nchunks - 1) / nchunks;
}

class loop_domain_1d {
    private:
        hclib_loop_domain_t loop;

    public:
        loop_domain_1d(int N) {
            loop.low = 0; loop.high = N;
            loop.stride = 1; loop.tile = default_tile_size(N, hclib_num_workers());
        }

        loop_domain_1d(int low, int high) {
            loop.low = low; loop.high = high;
            loop.stride = 1; loop.tile = default_tile_size(high - low, hclib_num_workers());
        }

        loop_domain_1d(int low, int high, int nchunks) {
            loop.low = low; loop.high = high;
            loop.stride = 1; loop.tile = default_tile_size(high - low, nchunks);
        }

        hclib_loop_domain_t *get_internal() { return &loop; }
};

class loop_domain_2d {
    private:
        hclib_loop_domain_t loop[2];

    public:
        loop_domain_2d(int N1, int N2) {
            loop[0].low = 0;    loop[0].high = N1;
            loop[0].stride = 1; loop[0].tile = default_tile_size(N1, hclib_num_workers());
            loop[1].low = 0;    loop[1].high = N2;
            loop[1].stride = 1; loop[1].tile = default_tile_size(N2, hclib_num_workers());
        }

        loop_domain_2d(int low1, int high1, int low2, int high2) {
            loop[0].low = low1; loop[0].high = high1;
            loop[0].stride = 1; loop[0].tile = default_tile_size(high1 - low1, hclib_num_workers());
            loop[1].low = low2; loop[1].high = high2;
            loop[1].stride = 1; loop[1].tile = default_tile_size(high2 - low2, hclib_num_workers());
        }

        hclib_loop_domain_t *get_internal() { return loop; }
};

class loop_domain_3d {
    private:
        hclib_loop_domain_t loop[3];

    public:
        loop_domain_3d(int N1, int N2, int N3) {
            loop[0].low = 0;    loop[0].high = N1;
            loop[0].stride = 1; loop[0].tile = default_tile_size(N1, hclib_num_workers());
            loop[1].low = 0;    loop[1].high = N2;
            loop[1].stride = 1; loop[1].tile = default_tile_size(N2, hclib_num_workers());
            loop[2].low = 0;    loop[2].high = N3;
            loop[2].stride = 1; loop[2].tile = default_tile_size(N3, hclib_num_workers());
        }

        loop_domain_3d(int low1, int high1, int low2, int high2, int low3,
                int high3) {
            loop[0].low = low1; loop[0].high = high1;
            loop[0].stride = 1; loop[0].tile = default_tile_size(high1 - low1, hclib_num_workers());
            loop[1].low = low2; loop[1].high = high2;
            loop[1].stride = 1; loop[1].tile = default_tile_size(high2 - low2, hclib_num_workers());
            loop[2].low = low3; loop[2].high = high3;
            loop[2].stride = 1; loop[2].tile = default_tile_size(high3 - low3, hclib_num_workers());
        }

        hclib_loop_domain_t *get_internal() { return loop; }
};


template <typename T>
inline void forasync1D_runner(const _hclib_loop_domain_t* loop, T lambda) {
	const hclib_loop_domain_t loop0 = loop[0];
    for (int i=loop0.low; i<loop0.high; i += loop0.stride) {
        lambda(i);
    }
}

template <typename T>
inline void forasync2D_runner(const _hclib_loop_domain_t loop[2], T lambda) {
	const hclib_loop_domain_t loop0 = loop[0];
	const hclib_loop_domain_t loop1 = loop[1];
	for(int i=loop0.low; i<loop0.high; i+=loop0.stride) {
		for(int j=loop1.low; j<loop1.high; j+=loop1.stride) {
			lambda(i, j);
		}
	}
}

template <typename T>
inline void forasync3D_runner(const _hclib_loop_domain_t loop[3], T lambda) {
	const hclib_loop_domain_t loop0 = loop[0];
	const hclib_loop_domain_t loop1 = loop[1];
	const hclib_loop_domain_t loop2 = loop[2];
	for (int i = loop0.low; i < loop0.high; i += loop0.stride) {
		for (int j = loop1.low; j < loop1.high; j += loop1.stride) {
			for (int k = loop2.low; k < loop2.high; k += loop2.stride) {
				lambda(i, j, k);
			}
		}
	}
}

template <typename T>
inline void forasync1D_recursive(_hclib_loop_domain_t * loop, T lambda,
        hclib::future_t *future) {
	int low = loop->low, high = loop->high, stride = loop->stride, tile = loop->tile;
	//split the range into two, spawn a new task for the first half and recurse on the rest
	if ((high-low) > tile) {
		int mid = (high+low)/2;
		// upper-half
		// delegate scheduling to the underlying runtime
        auto lambda_wrapper = [=]() {
            _hclib_loop_domain_t ld = {mid, high, stride, tile};
            forasync1D_recursive<T>(&ld, lambda, future);
        };

        hclib::async_await(lambda_wrapper, future);
		// update lower-half
		//continue to work on the half task
		_hclib_loop_domain_t ld = {low, mid, stride, tile};
		forasync1D_recursive<T>(&ld, lambda, future);
	} else {
		//compute the tile
		_hclib_loop_domain_t ld = {low, high, stride, tile};
		forasync1D_runner<T>(&ld, lambda);
	}
}

template <typename T>
inline void forasync2D_recursive(const _hclib_loop_domain_t loop[2], T lambda,
        hclib::future_t *future) {
	hclib_loop_domain_t loop0 = loop[0];
	int high0 = loop0.high;
	int low0 = loop0.low;
	int stride0 = loop0.stride;
	int tile0 = loop0.tile;

	hclib_loop_domain_t loop1 = loop[1];
	int high1 = loop1.high;
	int low1 = loop1.low;
	int stride1 = loop1.stride;
	int tile1 = loop1.tile;

	//split the range into two, spawn a new task for the first half and recurse on the rest
	hclib_loop_domain_t new_loop[2];
	bool new_loop_initialized = false;

	if((high0-low0) > tile0) {
		int mid = (high0+low0)/2;
		// upper-half
		new_loop[0] = {mid, high0, stride0, tile0};
		new_loop[1] = {low1, high1, stride1, tile1};
		// update lower-half
		high0 = mid;
		new_loop_initialized = true;
	} else if((high1-low1) > tile1) {
		int mid = (high1+low1)/2;
		// upper-half
		new_loop[0] = {low0, high0, stride0, tile0};
		new_loop[1] = {mid, high1, stride1, tile1};
		// update lower-half
		high1 = mid;
		new_loop_initialized = true;
	}

	// recurse
	if (new_loop_initialized) {
		// delegate scheduling to the underlying runtime
        auto lambda_wrapper = [=]() {
            forasync2D_recursive<T>(new_loop, lambda, future);
        };

        hclib::async_await(lambda_wrapper, future);

		//continue to work on the half task
		hclib_loop_domain_t new_loop_lower_half[2] = {
				{low0, high0, stride0, tile0},
				{low1, high1, stride1, tile1}
		};
		forasync2D_recursive<T>(new_loop_lower_half, lambda, future);
	} else { //compute the tile
		forasync2D_runner<T>(loop, lambda);
	}
}

template <typename T>
inline void forasync3D_recursive(const hclib_loop_domain_t loop[3], T lambda,
        hclib::future_t *future) {
	hclib_loop_domain_t loop0 = loop[0];
	int high0 = loop0.high;
	int low0 = loop0.low;
	int stride0 = loop0.stride;
	int tile0 = loop0.tile;

	hclib_loop_domain_t loop1 = loop[1];
	int high1 = loop1.high;
	int low1 = loop1.low;
	int stride1 = loop1.stride;
	int tile1 = loop1.tile;

	hclib_loop_domain_t loop2 = loop[2];
	int high2 = loop2.high;
	int low2 = loop2.low;
	int stride2 = loop2.stride;
	int tile2 = loop2.tile;

	//split the range into two, spawn a new task for the first half and recurse on the rest
	hclib_loop_domain_t new_loop[3];
	bool new_loop_initialized = false;

	if((high0-low0) > tile0) {
		int mid = (high0+low0)/2;
		// upper-half
		new_loop[0] = {mid, high0, stride0, tile0};
		new_loop[1] = {low1, high1, stride1, tile1};
		new_loop[2] = {low2, high2, stride2, tile2};
		// update lower-half
		high0 = mid;
		new_loop_initialized = true;
	} else if((high1-low1) > tile1) {
		int mid = (high1+low1)/2;
		// upper-half
		new_loop[0] = {low0, high0, stride0, tile0};
		new_loop[1] = {mid, high1, stride1, tile1};
		new_loop[2] = {low2, high2, stride2, tile2};
		// update lower-half
		high1 = mid;
		new_loop_initialized = true;
	} else if((high2-low2) > tile2) {
		int mid = (high2+low2)/2;
		// upper-half
		new_loop[0] = {low0, high0, stride0, tile0};
		new_loop[1] = {low1, high1, stride1, tile1};
		new_loop[2] = {mid, high2, stride2, tile2};
		// update lower-half
		high2 = mid;
		new_loop_initialized = true;
	}
	// recurse
	if (new_loop_initialized) {
		// delegate scheduling to the underlying runtime
        auto lambda_wrapper = [=]() {
			forasync3D_recursive<T>(new_loop, lambda, future);
		};

        hclib::async_await(lambda_wrapper, future);

		//continue to work on the half task
		hclib_loop_domain_t new_loop_lower_half[3] = {
				{low0, high0, stride0, tile0},
				{low1, high1, stride1, tile1},
				{low2, high2, stride2, tile2}
		};
		forasync3D_recursive<T>(new_loop_lower_half, lambda, future);
	} else { //compute the tile
		forasync3D_runner<T>(loop, lambda);
	}
}

template <typename T>
inline void forasync1D_flat(hclib_loop_domain_t* loop, T lambda,
        hclib::future_t *future, const int dist_func_id) {
    const int low = loop->low, high = loop->high, stride = loop->stride,
            tile = loop->tile;
    const loop_dist_func func = hclib_lookup_dist_func(dist_func_id);

	int nb_chunks = (int) (high/tile);
	int size = tile * nb_chunks;
	int low0;
	for (low0 = low; low0 < size; low0 += tile) {
        _hclib_loop_domain_t ld = {low0, low0 + tile, stride, tile};
        auto lambda_wrapper = [=]() {
			forasync1D_runner<T>(&ld, lambda);
		};

        hclib_locale_t *locale = func(1, &ld, loop, FORASYNC_MODE_FLAT);
        hclib::async_await_at(lambda_wrapper, future, locale);
	}
	// handling leftover
	if (size < high) {
        _hclib_loop_domain_t ld = {low0, high, stride, tile};
        auto lambda_wrapper = [=]() {
			forasync1D_runner<T>(&ld, lambda);
		};
        hclib_locale_t *locale = func(1, &ld, loop, FORASYNC_MODE_FLAT);
        hclib::async_await_at(lambda_wrapper, future, locale);
	}
}

template <typename T>
inline void forasync2D_flat(const hclib_loop_domain_t loop[2], T lambda,
        hclib::future_t *future) {
	hclib_loop_domain_t loop0 = loop[0];
	int high0 = loop0.high;
	int low0 = loop0.low;
	int stride0 = loop0.stride;
	int tile0 = loop0.tile;

	hclib_loop_domain_t loop1 = loop[1];
	int high1 = loop1.high;
	int low1 = loop1.low;
	int stride1 = loop1.stride;
	int tile1 = loop1.tile;

	for(int low0a=low0; low0a<high0; low0a+=tile0) {
		int high0a = (low0a+tile0)>high0?high0:(low0a+tile0);
		for(int low1a=low1; low1a<high1; low1a+=tile1) {
			int high1a = (low1a+tile1)>high1?high1:(low1a+tile1);
			hclib_loop_domain_t new_loop0 = {low0a, high0a, stride0, tile0};
			hclib_loop_domain_t new_loop1 = {low1a, high1a, stride1, tile1};
			hclib_loop_domain_t new_loop[2] = {new_loop0, new_loop1};

            auto lambda_wrapper = [=]() {
				forasync2D_runner<T>(new_loop, lambda);
			};

            hclib::async_await(lambda_wrapper, future);
		}
	}
}

template <typename T>
inline void forasync3D_flat(const hclib_loop_domain_t loop[3], T lambda,
        hclib::future_t *future) {
	hclib_loop_domain_t loop0 = loop[0];
	int high0 = loop0.high;
	int low0 = loop0.low;
	int stride0 = loop0.stride;
	int tile0 = loop0.tile;

	hclib_loop_domain_t loop1 = loop[1];
	int high1 = loop1.high;
	int low1 = loop1.low;
	int stride1 = loop1.stride;
	int tile1 = loop1.tile;

	hclib_loop_domain_t loop2 = loop[2];
	int high2 = loop2.high;
	int low2 = loop2.low;
	int stride2 = loop2.stride;
	int tile2 = loop2.tile;

	for(int low0a=low0; low0a<high0; low0a+=tile0) {
		int high0a = (low0a+tile0)>high0?high0:(low0a+tile0);
		for(int low1a=low1; low1a<high1; low1a+=tile1) {
			int high1a = (low1a+tile1)>high1?high1:(low1a+tile1);
			for(int low2a=low2; low2a<high2; low2a+=tile2) {
				int high2a = (low2a+tile2)>high2?high2:(low2a+tile2);
				hclib_loop_domain_t new_loop0 = {low0a, high0a, stride0, tile0};
				hclib_loop_domain_t new_loop1 = {low1a, high1a, stride1, tile1};
				hclib_loop_domain_t new_loop2 = {low2a, high2a, stride2, tile2};
				hclib_loop_domain_t new_loop[3] = {new_loop0, new_loop1, new_loop2};

                auto lambda_wrapper = [=]() {
					forasync3D_runner<T>(new_loop, lambda);
				};

                hclib::async_await(lambda_wrapper, future);
			}
		}
	}
}

template <typename T>
inline void forasync1D_internal(_hclib_loop_domain_t* loop, T lambda, int mode,
        hclib::future_t *future, const int dist_func_id) {
	switch(mode) {
	case FORASYNC_MODE_FLAT:
		forasync1D_flat<T>(loop, lambda, future, dist_func_id);
		break;
	case FORASYNC_MODE_RECURSIVE:
        HASSERT(dist_func_id == HCLIB_DEFAULT_LOOP_DIST);
		forasync1D_recursive<T>(loop, lambda, future);
		break;
	default:
		HASSERT("Check forasync mode" && false);
	}
}

template <typename T>
inline void forasync2D_internal(const hclib_loop_domain_t loop[2], T lambda,
        int mode, hclib::future_t *future) {
	switch(mode) {
	case FORASYNC_MODE_FLAT:
		forasync2D_flat<T>(loop, lambda, future);
		break;
	case FORASYNC_MODE_RECURSIVE:
		forasync2D_recursive<T>(loop, lambda, future);
		break;
	default:
		HASSERT("Check forasync mode" && false);
	}
}

template <typename T>
inline void forasync3D_internal(const hclib_loop_domain_t loop[3], T lambda,
        int mode, hclib::future_t *future) {
	switch(mode) {
	case FORASYNC_MODE_FLAT:
		forasync3D_flat<T>(loop, lambda, future);
		break;
	case FORASYNC_MODE_RECURSIVE:
		forasync3D_recursive<T>(loop, lambda, future);
		break;
	default:
		HASSERT("Check forasync mode" && false);
	}
}

template <typename T>
inline void forasync1D(loop_domain_1d* loop, T lambda,
        bool force_seq = false, int mode = FORASYNC_MODE_RECURSIVE,
        hclib::future_t *future = NULL,
        int dist_func_id = HCLIB_DEFAULT_LOOP_DIST) {
    if (force_seq) {
        hclib_loop_domain_t *internal = loop->get_internal();
        for (int i = internal->low; i < internal->high; i += internal->stride) {
            lambda(i);
        }
    } else {
        forasync1D_internal<T>(loop->get_internal(), lambda, mode,
                future, dist_func_id);
    }
}

template <typename T>
inline void forasync2D(loop_domain_2d* loop, T lambda,
        bool force_seq = false, int mode = FORASYNC_MODE_RECURSIVE,
        hclib::future_t *future = NULL) {
    if (force_seq) {
        hclib_loop_domain_t *internal = loop->get_internal();
        for (int i = internal[0].low; i < internal[0].high;
                i += internal[0].stride) {
            for (int j = internal[1].low; j < internal[1].high;
                    j += internal[1].stride) {
                lambda(i, j);
            }
        }
    } else {
        forasync2D_internal<T>(loop->get_internal(), lambda, mode, future);
    }
}

template <typename T>
inline void forasync3D(loop_domain_3d* loop, T lambda,
        bool force_seq = false, int mode = FORASYNC_MODE_RECURSIVE,
        hclib::future_t *future = NULL) {
    assert(force_seq == false);
    forasync3D_internal<T>(loop->get_internal(), lambda, mode, future);
}

template <typename T>
inline hclib::future_t *forasync1D_future(loop_domain_1d* loop, T lambda,
        bool force_seq = false, int mode = FORASYNC_MODE_RECURSIVE,
        hclib::future_t *future = NULL,
        int dist_func_id = HCLIB_DEFAULT_LOOP_DIST) {
#ifdef VERBOSE
    fprintf(stderr, "forasync1D_future: future=%p\n", future);
#endif
    hclib::promise_t *event = new hclib::promise_t();

    if (force_seq) {
        hclib_loop_domain_t *internal = loop->get_internal();
        for (int i = internal[0].low; i < internal[0].high;
                i += internal[0].stride) {
            lambda(i);
        }
        event->put(NULL);
        return event->get_future();
    } else {
        hclib_start_finish();
        forasync1D_internal<T>(loop->get_internal(), lambda, mode, future,
                dist_func_id);
        hclib_end_finish_nonblocking_helper(&event->internal);
        return event->get_future();
    }
}

template <typename T>
inline hclib::future_t *forasync2D_future(loop_domain_2d* loop, T lambda,
        int mode = FORASYNC_MODE_RECURSIVE, hclib::future_t *future = NULL) {
    hclib_start_finish();
    forasync2D_internal<T>(loop->get_internal(), lambda, mode, future);
    hclib::promise_t *event = new hclib::promise_t();
    hclib_end_finish_nonblocking_helper(&event->internal);
    return event->get_future();
}

template <typename T>
inline hclib::future_t *forasync3D_future(loop_domain_3d* loop, T lambda,
        int mode = FORASYNC_MODE_RECURSIVE, hclib::future_t *future = NULL) {
    hclib_start_finish();
    forasync3D_internal<T>(loop->get_internal(), lambda, mode, future);
    hclib::promise_t *event = new hclib::promise_t();
    hclib_end_finish_nonblocking_helper(&event->internal);
    return event->get_future();
}

}

#endif /* HCLIB_FORASYNC_H_ */
