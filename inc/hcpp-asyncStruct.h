/*
 * hcpp-asyncStruct.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

#ifndef HCPP_ASYNCSTRUCT_H_
#define HCPP_ASYNCSTRUCT_H_

namespace hcpp {
/*
 * We just need to pack the function pointer and the pointer to
 * the heap allocated lambda. Due to this 16 bytes is sufficient
 * for our lambda based approach.
 */
#define MAX_HCPP_ASYNC_ARG_SIZE 16

template <typename Function, typename T1>
struct async_arguments1 {
	Function kernel;
	T1 arg1;

	async_arguments1(Function k, T1 a1) :
		kernel(k), arg1(a1) { }
};

template<typename Function, typename T1>
void wrapper1(void *args) {
	async_arguments1<Function, T1> *a =
			(async_arguments1<Function, T1> *)args;

	(*a->kernel)(a->arg1);
}

struct hcpp_async_task  {
	char _args[MAX_HCPP_ASYNC_ARG_SIZE];
	struct finish_t* current_finish;
	generic_framePtr _fp;
	/*
	 * Boolean flag specific to HabaneroUPC++ only.
	 * Used to differentiate between a normal hcpp async
	 * and locality flexible asyncAny
	 */
	bool is_asyncAnyType;
	ddf_t ** ddf_list; // Null terminated list
	inline void init_hcpp_async_task(generic_framePtr fp, size_t arg_sz, void *async_args) {
		HASSERT(arg_sz <= MAX_HCPP_ASYNC_ARG_SIZE);
		this->_fp = fp;
		this->is_asyncAnyType = false;
		this->ddf_list = NULL;
		memcpy(&this->_args, async_args, arg_sz);
	}

	inline void set_current_finish(struct finish_t* finish) {
		this->current_finish = finish;
	}

	inline struct finish_t* get_current_finish() {
		return this->current_finish;
	}

	inline void set_ddf_list(ddf_t ** ddf) {
		this->ddf_list = ddf;
	}

	inline ddf_t ** get_ddf_list() {
		return this->ddf_list;
	}

	inline void mark_as_asyncAnyTask() {
		this->is_asyncAnyType = true;
	}

	inline bool is_asyncAnyTask() {
		return this->is_asyncAnyType;
	}

	inline hcpp_async_task() { };

	template<typename Function, typename T1>
	inline hcpp_async_task(Function kernel, const T1 &a1) {
		async_arguments1<Function, T1> args(kernel, a1);
		init_hcpp_async_task(wrapper1<Function, T1>, (size_t)sizeof(args), (void *)&args);
	}
};

#define task_t 	hcpp_async_task

typedef struct hcpp_task_t {
	task_t async_task; 	// the actual task
	ddt_t ddt; 			// ddt meta-information
} hcpp_task_t;

void spawn(task_t * task);
void spawn_at_hpt(place_t* pl, task_t * task);
void spawn_await(task_t * task, ddf_t** ddf_list);
void spawn_commTask(task_t * task);

}

#endif /* HCPP_ASYNCSTRUCT_H_ */
