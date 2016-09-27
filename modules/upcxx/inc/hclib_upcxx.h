#ifndef HCLIB_UPCXX_H
#define HCLIB_UPCXX_H

#include "hclib_upcxx-internal.h"

#include <set>

#if defined(HCLIB_MEASURE_START_LATENCY) || defined(HCLIB_PROFILE)
enum UPCXX_FUNC_LABELS {
    barrier_lbl = 0,
    async_wait_lbl,
    advance_lbl,
    async_after_lbl,
    async_copy_lbl,
    async_lbl,
    N_UPCXX_FUNCS
};
#endif

#ifdef HCLIB_MEASURE_START_LATENCY
extern unsigned long long upcxx_latency_counters[N_UPCXX_FUNCS];
extern unsigned long long upcxx_latency_times[N_UPCXX_FUNCS];
#endif

#ifdef HCLIB_PROFILE
extern unsigned long long upcxx_profile_counters[N_UPCXX_FUNCS];
extern unsigned long long upcxx_profile_times[N_UPCXX_FUNCS];
#endif

#ifdef HCLIB_PROFILE
#define UPCXX_START_PROFILE const unsigned long long __upcxx_profile_start_time = hclib_current_time_ns();

#define UPCXX_END_PROFILE(funcname) { \
    const unsigned long long __upcxx_profile_end_time = hclib_current_time_ns(); \
    upcxx_profile_counters[funcname##_lbl]++; \
    upcxx_profile_times[funcname##_lbl] += (__upcxx_profile_end_time - __upcxx_profile_start_time); \
}
#else
#define UPCXX_START_PROFILE
#define UPCXX_END_PROFILE(funcname)
#endif

#ifdef HCLIB_MEASURE_START_LATENCY
#define UPCXX_START_LATENCY const unsigned long long __upcxx_latency_start_time = hclib_current_time_ns();
#define UPCXX_END_LATENCY(funcname) { \
    const unsigned long long __upcxx_latency_end_time = hclib_current_time_ns(); \
    upcxx_latency_counters[funcname##_lbl]++; \
    upcxx_latency_times[funcname##_lbl] += (__upcxx_latency_end_time - __upcxx_latency_start_time); \
}
#else
#define UPCXX_START_LATENCY
#define UPCXX_END_LATENCY(funcname)
#endif

namespace hclib {

namespace upcxx {

struct event : public ::upcxx::event {
    public:
        event() : ::upcxx::event() { }
};

struct team : public ::upcxx::team {
    public:
        team(::upcxx::team other) : ::upcxx::team(other) { }
        team() : ::upcxx::team() { }

        int split(uint32_t color, uint32_t relrank, team *&new_team) {
            ::upcxx::team *new_internal;
            const int result = ::upcxx::team::split(color, relrank,
                    new_internal);
            new_team = new team(*new_internal);
            return result;
        }
};

template<typename T>
class global_ptr;

template<typename T, typename place_t = ::upcxx::rank_t>
struct global_ref : public ::upcxx::global_ref_base<T, place_t> {
    public:
        global_ref(place_t pla, T *ptr) : ::upcxx::global_ref_base<T, place_t>(pla, ptr) {
        }

        global_ref(::upcxx::global_ref_base<T, place_t> internal) : ::upcxx::global_ref_base<T, place_t>(internal) { }

        global_ref& operator =(const T &rhs) {
            ::upcxx::global_ref_base<T, place_t>::operator=(rhs);
            return *this;
        }

        global_ptr<T> operator &() {
            return global_ptr<T>(::upcxx::global_ref_base<T, place_t>::operator&());
        }
};

template<typename T>
struct global_ref<hclib::upcxx::global_ptr<T> > : public ::upcxx::global_ref_base<hclib::upcxx::global_ptr<T> > {
    public:
        global_ref(::upcxx::rank_t pla, hclib::upcxx::global_ptr<T> *ptr) :
            ::upcxx::global_ref_base<hclib::upcxx::global_ptr<T>, ::upcxx::rank_t >(pla, ptr) {
        }

        global_ref(::upcxx::global_ref_base<hclib::upcxx::global_ptr<T> > internal) : ::upcxx::global_ref_base<hclib::upcxx::global_ptr<T> >(internal) { }

        template<typename T2>
        global_ref<T> operator [](T2 i) {
            hclib::upcxx::global_ptr<T> tmp =
                ::upcxx::global_ref_base<hclib::upcxx::global_ptr<T> >::get();
            return tmp[i];
        }

        inline global_ref operator =(const global_ptr<T>& rhs) {
            return ::upcxx::global_ref_base<global_ptr<T> >::operator=(rhs);
        }
};

template<typename T>
class global_ptr : public ::upcxx::global_ptr<T> {
    public:
        inline explicit global_ptr() : ::upcxx::global_ptr<T>() { }

        inline explicit global_ptr(::upcxx::global_ptr<T> internal) : ::upcxx::global_ptr<T>(internal) { }

        inline explicit global_ptr(T *ptr, ::upcxx::rank_t pla) :
            ::upcxx::global_ptr<T>(ptr, pla) {
        }

        template <typename T2>
        global_ref<T> operator [] (T2 i) const {
            return global_ref<T>(this->where(), (T *)this->raw_ptr() + i);
        }

        template <typename T2>
        global_ptr<T> operator +(T2 i) const {
            return global_ptr<T>(((T *)::upcxx::global_ptr<T>::raw_ptr()) + i,
                    ::upcxx::global_ptr<T>::where());
        }
};



template <typename T, int BLK_SZ>
struct shared_array {
    private:
        ::upcxx::shared_array<T, BLK_SZ> internal;

    public:
        void init(size_t sz, size_t blk_sz) {
            hclib::finish([&] {
                hclib::async_at(nic_place(), [&] {
                    internal.init(sz, blk_sz);
                });
            });
        }

        hclib::upcxx::global_ref<T> operator [] (size_t global_index) {
            return hclib::upcxx::global_ref<T>(internal[global_index]);
        }
};

template<class T>
struct future {
    private:
        ::upcxx::future<T> internal;

    public:
        future(::upcxx::future<T> set) : internal(set) { }
};

template<typename dest = ::upcxx::rank_t>
struct gasnet_launcher {
    private:
        ::upcxx::gasnet_launcher<dest> internal;

    public:
        gasnet_launcher(::upcxx::gasnet_launcher<dest> set_internal) : internal(set_internal) { }

        template<typename Function, typename... Ts>
        inline hclib::upcxx::future<typename std::result_of<Function(Ts...)>::type>
        operator()(Function k, const Ts &... as) {
            return hclib::upcxx::future<typename std::result_of<Function(Ts...)>::type>(internal.operator()(k, as...));
        }
};

extern hclib::upcxx::team team_all;
extern std::set<hclib::upcxx::event *> seen_events;
extern unsigned long long async_copy_latency;
extern unsigned long long async_copy_count;

uint32_t ranks();
uint32_t myrank();

void barrier();
void async_wait();
int advance();

void print_upcxx_profiling_data();

template<typename T>
static void call_lambda(T lambda) {
    lambda();
}

static void advance_callback(hclib_future_t *fut) {
    if (hclib_future_is_satisfied(fut)) return;

    hclib::async([=] {
            hclib::upcxx::advance();
            advance_callback(fut);
        });
}

template<typename T>
void remote_finish(T lambda) {
    hclib::finish([=] {
        lambda();
    });
    hclib::upcxx::async_wait();
}

template<typename T>
void async_after(::upcxx::rank_t rank, hclib::future_t *after,
        T lambda) {
    hclib::async_await_at([=] {
            ::upcxx::async(rank)(call_lambda<T>, lambda);
        }, after, nic_place());
}

template<typename T>
hclib::future_t *async_copy(hclib::upcxx::global_ptr<T> src,
        hclib::upcxx::global_ptr<T> dst, size_t count) {
    return hclib::async_future_at([=] {
            UPCXX_END_LATENCY(async_copy);

            UPCXX_START_PROFILE;
            ::upcxx::copy((::upcxx::global_ptr<T>)src,
                (::upcxx::global_ptr<T>)dst, count);
            UPCXX_END_PROFILE(async_copy);
        }, nic_place());
}

hclib::upcxx::gasnet_launcher<::upcxx::rank_t> async(::upcxx::rank_t rank,
        hclib::upcxx::event *ack);

bool is_memory_shared_with(::upcxx::rank_t r);
int split(uint32_t color, uint32_t relrank, team *&new_team);

template<typename T>
hclib::upcxx::global_ptr<T> allocate(::upcxx::rank_t rank, size_t count) {
    return hclib::upcxx::global_ptr<T>(::upcxx::allocate<T>(rank, count));
}

template<typename T>
void deallocate(global_ptr<T> ptr) {
    ::upcxx::deallocate(ptr);
}

template<typename T>
T* allocate(size_t count) {
    return ::upcxx::allocate(count);
}

}
}

#endif
