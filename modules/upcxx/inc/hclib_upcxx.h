#ifndef HCLIB_UPCXX_H
#define HCLIB_UPCXX_H

#include "hclib_upcxx-internal.h"

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
            internal.init(sz, blk_sz);
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

uint32_t ranks();
uint32_t myrank();

void barrier();
void async_wait();

template<typename T>
int async_copy(hclib::upcxx::global_ptr<T> src,
        hclib::upcxx::global_ptr<T> dst, size_t count,
        hclib::upcxx::event *done_event) {
    hclib::finish([&src, &dst, &count, &done_event] {
        hclib::async_at(nic_place(), [&src, &dst, &count, &done_event] {
            ::upcxx::async_copy((::upcxx::global_ptr<T>)src, (::upcxx::global_ptr<T>)dst, count,
                done_event);
        });
    });
}

hclib::upcxx::gasnet_launcher<::upcxx::rank_t> async_after(::upcxx::rank_t rank,
        hclib::upcxx::event *after, hclib::upcxx::event *ack);

bool is_memory_shared_with(::upcxx::rank_t r);
int split(uint32_t color, uint32_t relrank, team *&new_team);

template<typename T>
hclib::upcxx::global_ptr<T> allocate(::upcxx::rank_t rank, size_t count) {
    return hclib::upcxx::global_ptr<T>(::upcxx::allocate<T>(rank, count));
}

template<typename T>
T* allocate(size_t count) {
    return ::upcxx::allocate(count);
}

}
}

#endif
