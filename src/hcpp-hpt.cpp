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
 * hcpp-hpt.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include "hcpp-internal.h"
#include "hcpp-atomics.h"
#include <stdio.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

namespace hcpp {

/**
 * HPT: Try to steal a frame from another worker.
 * 1) First look for work in current place worker deques
 * 2) If nothing is found, check if current place has device
 *    places and if so, try to steal from there.
 * 3) If unsuccessful, start over at step 1) in the parent
 *    place all to the hpt top.
 */
task_t* hpt_steal_task(hc_workerState* ws) {
	MARK_SEARCH(ws->id);
	place_t * pl = ws->pl;
	hcupc_reset_asyncAnyInfo(ws->id);
	while (pl != NULL) {
		hc_deque_t * deqs = pl->deques;
		int nb_deq = pl->ndeques;
		/* Try to steal from right neighbour */

		/* Try to steal once from every other worker first */
		for (int i=1; i<nb_deq; i++) {
			int victim = ((ws->id+i)%nb_deq);
			hc_deque_t* d = &deqs[victim];
			task_t* buff = dequeSteal(&(d->deque));
			if (buff) { /* steal succeeded */
				ws->current = get_deque_place(ws, pl);
				hcupc_check_if_asyncAny_stolen(buff, victim, ws->id);
				return buff;
			}
		}
		hcupc_inform_failedSteal(ws->id);
#if TODO
		/* We also steal from places that represents the device (GPU),
		 * those continuations that are pushed onto the deque by the
		 * device workers
		 */
		place_t *child = pl->child;
		while (child != NULL) {
			if (is_device_place(child)) {
				frame = hc_deque_steal(ws, child->deques);
				if (frame != NULL) {
					ws->current = get_deque_place(ws, pl);
					return buff;
				}
			}
			child = child->nnext;
		}
#endif
		/* Nothing found in this place, go to the parent */
		pl = pl->parent;
	}
	return NULL;
}

/**
 * HPT: Pop items from a worker deque
 * 1) Try to pop from current queue (Q)
 * 2) if nothing found, try to pop downward from worker's child deque.
 * 3) If nothing found down to the bottom, look upward starting from Q.
 */
task_t* hpt_pop_task(hc_workerState * ws) {
	hc_deque_t * current = ws->current; /* go HPT downward and then upward of my own deques */
	hc_deque_t * pivot = current;
	short downward = 1;
	while (current != NULL) {
		task_t* buff = dequePop(&current->deque);
		if (buff) {
			ws->current = current;
			hcupc_check_if_asyncAny_pop(buff, ws->id);
			return buff;
		}
		if (downward) {
			if (current->nnext == NULL) {
				current = pivot->prev;
				downward = 0; /* next, we go upward from pivot */
			} else current = current->nnext;
		} else {
			current = current->prev;
		}
	}
	return NULL;
}

inline short is_cpu_place(place_t * pl) {
	HASSERT(pl);
	return (pl->type == MEM_PLACE || pl->type == CACHE_PLACE);
}

inline short is_device_place(place_t * pl) {
	HASSERT(pl);
	return (pl->type == NVGPU_PLACE || pl->type == AMGPU_PLACE || pl->type == FPGA_PLACE );
}

inline short is_nvgpu_place(place_t * pl) {
	HASSERT(pl);
	return (pl->type == NVGPU_PLACE);
}

place_t* hc_get_current_place() {
	hc_workerState * ws = current_ws_internal();
	HASSERT(ws->current->pl != NULL);
	return ws->current->pl;
}

int hc_get_num_places(short type) {
	hc_workerState * ws = current_ws_internal();
	place_t ** all_places = ws->context->places;
	int np = ws->context->nplaces;
	int i;
	int num = 0;
	for (i=0; i<np; i++)
		if (all_places[i]->type == type) num++;
	return num;
}

void hc_get_places(place_t ** pls, short type) {
	hc_workerState * ws = current_ws_internal();
	place_t ** all_places = ws->context->places;
	int np = ws->context->nplaces;
	int i;
	int num = 0;
	for (i=0; i<np; i++)
		if (all_places[i]->type == type) pls[num++] = all_places[i];
	return;
}

place_t * hc_get_place(short type) {
	hc_workerState * ws = current_ws_internal();
	place_t ** all_places = ws->context->places;
	int np = ws->context->nplaces;
	int i;
	for (i=0; i<np; i++)
		if (all_places[i]->type == type)  return all_places[i];
	return NULL;
}

place_t * hc_get_root_place() {
	hc_workerState * ws = current_ws_internal();
	place_t ** all_places = ws->context->places;
	return all_places[0];
}

inline place_t * get_ancestor_place(hc_workerState * ws) {
	place_t * parent = ws->pl;
	while (parent->parent != NULL) parent = parent->parent;
	return parent;
}

place_t * hc_get_child_place() {
	hc_workerState * ws = current_ws_internal();
	place_t * pl = ws->current->pl;
	HASSERT(pl != NULL);
	if (ws->hpt_path == NULL) return pl;
	return ws->hpt_path[pl->level + 1];
}

place_t * hc_get_parent_place() {
	hc_workerState * ws = current_ws_internal();
	place_t * pl = ws->current->pl;
	HASSERT(pl != NULL);
	if (ws->hpt_path == NULL) return pl;
	return ws->hpt_path[pl->level - 1];
}

place_t ** hc_get_children_places(int * numChildren) {
	place_t * pl = hc_get_current_place();
	*numChildren = pl->nChildren;
	return pl->children;
}

place_t ** hc_get_children_of_place(place_t * pl, int * numChildren) {
	*numChildren = pl->nChildren;
	return pl->children;
}

/*
 * Return my own deque that is in place pl,
 * if pl == NULL, return the current deque of the worker
 */
inline hc_deque_t * get_deque_place(hc_workerState * ws, place_t * pl) {
	if (pl == NULL) return ws->current;
	return &(pl->deques[ws->id]);
}

hc_deque_t * get_deque(hc_workerState * ws) {
	return NULL;
}

/* get the first owned deque from HPT that starts with place pl upward */
hc_deque_t * get_deque_hpt(hc_workerState * ws, place_t * pl) {
	return NULL;

}

bool deque_push_place(hc_workerState *ws, place_t * pl, void * ele) {
	if (is_device_place(pl)) {
#ifdef TODO
		hcqueue_enqueue(ws, pl->hcque, ele); // TODO
#endif
		return true;
	} else {
		hc_deque_t * deq = get_deque_place(ws, pl);
		return dequePush(&deq->deque, ele);
	}
}

inline task_t* deque_pop_place(hc_workerState *ws, place_t * pl) {
	hc_deque_t * deq = get_deque_place(ws, pl);
	return dequePop(&deq->deque);
}

/**
 * Initializes a hc_deque_t
 */
inline void init_hc_deque_t(hc_deque_t * hcdeq, place_t * pl){
	hcdeq->deque.head = hcdeq->deque.tail = 0;
	hcdeq->pl = pl;
	hcdeq->ws = NULL;
	hcdeq->nnext = NULL;
	hcdeq->prev = NULL;
#ifdef BUCKET_DEQUE
	hcdeq->deque.last = 0;
	hcdeq->deque.thief = 0;
	hcdeq->deque.staleMaps = NULL;
#endif
}

/*Initialize deq's buffer (data + capacity)*/
void init_deq_buffer(hc_workerState * ws, deque_t * deq, int capacity) {
	//TODO: in our current design this does not makes sense
}

/* init the hpt and place deques */
void hc_hpt_init(hc_context * context) {
	int i, j;
#ifdef HPT_DESCENTWORKER_PERPLACE /* each place has a deque for each of its descent workers */
	for (i = 0; i < context->nplaces; i++) {
		place_t * pl = context->places[i];
		int nworkers = pl->ndeques;
		pl->deques = malloc(sizeof(hc_deque_t) * nworkers);
		for (j = 0; j < nworkers; j++) {
			hc_deque_t * deq = &(pl->deques[j]);
			init_hc_deque_t(deq, pl);
		}
	}
#else /* HPT_ALLWORKER_PERPLACE each place has a deque for each worker */
	for (i = 0; i < context->nplaces; i++) {
		place_t * pl = context->places[i];
		int ndeques = context->nworkers;
		if (is_device_place(pl)) ndeques = 1;
		pl->ndeques = ndeques;
		pl->deques = (hc_deque_t*) malloc(sizeof(hc_deque_t) * ndeques);
		for (j = 0; j < ndeques; j++) {
			hc_deque_t * hc_deq = &(pl->deques[j]);
			init_hc_deque_t(hc_deq, pl);
#ifndef BUCKET_DEQUE
#ifndef UNLOCK_DEQUE
			//pthread_mutex_init(&(hc_deq->deque.lock), NULL); // TODO:Check
#endif
#endif
		}
	}
#endif

	/* init the deque in the HPT tree, set the current place of each worker,
	 * and link the deque for the worker
	 */
#if 0 /* this is the algorithm in which we only have a deque for the ancestor places of a worker */
	for (i=0; i<context->nworkers; i++) {
		hc_workerState * ws = context->workers[i];
		place_t * parent = ws->pl;
		hc_deque_t * tmp = NULL;
		hc_deque_t * deqp = NULL;
		hc_deque_t * endp = NULL;
		while (parent != NULL) {
			for (j=0; j<parent->ndeques; j++) {
				tmp = &(parent->deques[j]);
				if (tmp->ws == NULL) { /* find the first deque not owned by a worker */
					tmp->ws = ws;
					ws->current = tmp; /* we will finally set it to the topmost one */
					printf("worker %d deque is at slot %d of place %d\n", ws->id, j, parent->id);
					if (deqp == NULL) {
						deqp = tmp;
						endp = tmp;
					} else {
						endp->prev = tmp;
						tmp->nnext = endp;
						endp = tmp;
					}
					break;
				}
			}
			parent = parent->parent;
		}
		ws->deques = deqp;
	}
#endif

	//int deq_capacity = context->options->deqsize;
	/* link the deques for each cpu workers. the deque index is the same as ws->id to simplify the search.
	 */
	for (i = 0; i < context->nworkers; i++) {
		hc_workerState * ws = context->workers[i];
		int id = ws->id;
		for (j=0; j<context->nplaces; j++) {
			place_t * pl = context->places[j];
			hc_deque_t * hc_deq;
			if (is_cpu_place(pl)) {
				hc_deq = &(pl->deques[id]);
				hc_deq->ws = ws;
				/* We dont have deq buffers
				init_deq_buffer(ws, ((deque_t *) hc_deq), deq_capacity);
			} else if (is_device_place(pl) && pl->deques->deque.buffer == NULL) {
				hc_deq = pl->deques;
				init_deq_buffer(ws, ((deque_t *) hc_deq), deq_capacity);
				 */
			} else {
				/* unhandled or ignored situation */
			}
		}

		/* here we link the deques of the ancestor places for this worker */
		place_t * parent = ws->pl;
		place_t * current = parent;
		ws->deques = &(current->deques[id]);
		while (parent->parent != NULL) {
			parent = parent->parent;
			current->deques[id].prev = &(parent->deques[id]);
			parent->deques[id].nnext = &(current->deques[id]);
			current = parent;
		}
		ws->current = &(current->deques[id]);
	}
#if 0
	/*Print HPT*/
	int level = context->places[0]->level;
	for (i=0; i<context->nplaces; i++) {
		place_t * pl = context->places[i];
		if (level != pl->level) {
			printf("\n");
			level = pl->level;
		}

		printf("Place %d ", pl->id);
		hc_workerState * w = pl->workers;
		if (w != NULL) {
			printf("[ ");
			while (w != NULL) {
				printf("%d ", w->id);
				w = w->nnext;
			}
			printf("] ");
		}

		place_t * c = pl->child;
		if (c != NULL) {
			printf("{ ");
			while (c != NULL) {
				printf("%d ", c->id);
				c = c->nnext;
			}
			printf("} ");
		}
		printf("\t");
	}
	printf("\n");
#endif
}

/* init dev places of the hpt, create a pthread and a reverse-deque for each GPU place */
void hc_hpt_dev_init(hc_context * context) {
#ifdef TODO
	// Not supported currently
	int i;
	for (i = 0; i < context->nplaces; i++) {
		place_t * pl = context->places[i];
		if (is_device_place(pl)) {
			hc_workerState * dws = pl->workers;
			// NOTE: the following inits are not used at this moment, may be later, so we do it
			dws->num = 1;
			dws->deques = pl->deques;
			dws->current = pl->deques;
			dws->deques->ws = dws;

			// init the (dev) task queue
			pl->hcque = hcqueue_init(dws);
		}
	}
#endif
}

void hc_hpt_dev_cleanup(hc_context * context) {
#ifdef TODO
	int i;
	for (i = 0; i < context->nplaces; i++) {
		place_t * pl = context->places[i];
		if (is_device_place(pl)) {
			hc_workerState * dws = pl->workers;
			// NOTE: the following inits are not used at this moment, may be later, so we do it
			dws->num = 1;
			dws->deques = pl->deques;
			dws->current = pl->deques;

			// init the (dev) task queue
			hcqueue_destroy(dws, pl->hcque);
		}
	}
#endif
}

void hc_hpt_cleanup_1(hc_context * context) {
	/* clean up HPT deques for cpu workers*/
	for (int i = 0; i < context->nplaces; i++) {
		place_t * pl = context->places[i];
		if (is_device_place(pl)) continue;
		free(pl->deques);
	}
}

void hc_hpt_cleanup_2(hc_context * context) {
	/* clean up the HPT, places and workers */
	if (getenv("HCPP_HPT_FILE")) {
		freeHPT(context->hpt);
	} else {
		for(int i=0; i<context->nproc; i++) {
			free(context->workers[i]);
		}
		free(context->workers);
		free(context->hpt);
	}
}

/*
 * Interfaces to read the xml files and parse correctly to generate the place data-structures
 */
hc_workerState * parseWorkerElement(xmlNode * wkNode) {
	hc_workerState * wk = (hc_workerState *) malloc(sizeof(hc_workerState));
	xmlChar * num = xmlGetProp(wkNode, xmlCharStrdup("num"));
	xmlChar * didStr = xmlGetProp(wkNode, xmlCharStrdup("did"));
	xmlChar * type = xmlGetProp(wkNode, xmlCharStrdup("type"));
	/*
	printf("Worker(%x): num: %s, type: %s\n", wk, num, type);
	 */

	if (num != NULL) wk->id = atoi((char*)num);
	else wk->id = 1;
	if (didStr != NULL) wk->did = atoi((char*)didStr);
	else wk->did = 0;
	wk->nnext = NULL;
	/* TODO: worker/deque type */

	xmlFree(num);
	xmlFree(didStr);
	xmlFree(type);
	return wk;
}

place_t * parsePlaceElement(xmlNode * plNode) {
	int num = 1;
	int did = 0;
	place_t * pl = (place_t *) malloc(sizeof(place_t));
	xmlChar * numStr = xmlGetProp(plNode, xmlCharStrdup("num"));
	xmlChar * didStr = xmlGetProp(plNode, xmlCharStrdup("did"));
	xmlChar * typeStr = xmlGetProp(plNode, xmlCharStrdup("type"));
	xmlChar * sizeStr = xmlGetProp(plNode, xmlCharStrdup("size"));
	xmlChar * unitSize = xmlGetProp(plNode, xmlCharStrdup("unitSize"));

	/*
	printf("Place(%x): num: %s, type: %s, size: %s, unitSize: %s\n", pl, numStr, typeStr, sizeStr, unitSize);
	 */
	if (numStr != NULL) {
		num = atoi((char*)numStr);
    }

	if (didStr != NULL) {
        did = atoi((char*)didStr);
    }

	short type = CACHE_PLACE;
	if (typeStr != NULL) {
		if (!xmlStrcmp(typeStr, (const xmlChar *) "mem")) {
			type = MEM_PLACE;
		} else if (!xmlStrcmp(typeStr, (const xmlChar *) "cache")) {
			type = CACHE_PLACE;
		} else if (!xmlStrcmp(typeStr, (const xmlChar *) "nvgpu")) {
			type = NVGPU_PLACE;
		} else if (!xmlStrcmp(typeStr, (const xmlChar *) "amgpu")) {
			type = AMGPU_PLACE;
		} else if (!xmlStrcmp(typeStr, (const xmlChar *) "fpga")) {
			type = FPGA_PLACE;
		} else if (!xmlStrcmp(typeStr, (const xmlChar *) "pgas")) {
			type = PGAS_PLACE;
		} else {
			/* warnning, unknown type specified */
		}
	} else {
		/* default to be cache type */
		type = CACHE_PLACE;
	}

    // Not supported yet
    HC_ASSERT(type != AMGPU_PLACE && type != FPGA_PLACE);

	xmlFree(numStr);
	xmlFree(didStr);
	xmlFree(typeStr);
	xmlFree(sizeStr);
	xmlFree(unitSize);

	pl->id = num;
	pl->did = did;
	pl->type = type;
	pl->psize = (sizeStr == NULL) ? 0 : atoi((char*)sizeStr);
	pl->unitSize = (unitSize == NULL) ? 0 : atoi((char*)unitSize);
	pl->level = 0;
	pl->nChildren = 0;
	pl->children = NULL;
	pl->child = NULL;
	pl->workers = NULL;
	pl->nnext = NULL;
	pl->ndeques = 0;

	xmlNode *child = plNode->xmlChildrenNode;

	place_t * pllast = NULL;
	hc_workerState * wslast = NULL;

	while (child != NULL) {
		if (!xmlStrcmp(child->name, (const xmlChar *) "place")) {
			place_t * tmp = parsePlaceElement(child);
			tmp->parent = pl;
			if (pl->child == NULL) pl->child = tmp;
			else pllast->nnext = tmp;
			pllast = tmp;
		} else if (!xmlStrcmp(child->name, (const xmlChar *) "worker")) {
			hc_workerState * tmp = parseWorkerElement(child);
			tmp->pl = pl;
			if (pl->workers == NULL) pl->workers = tmp;
			else wslast->nnext = tmp;
			wslast = tmp;
		}
		child = child->next;
	}
	return pl;
}

/*
 * Generates an in-memory representation of an HPT, and returns the first element in the top layer.
 *
 * In memory, an HPT is stored as something like:
 *
 *      + -> + -> +
 *      |         |
 *      |         |
 *      + -> +    + -> +
 *
 * with each layer consisting of a linked list of memory spaces/processing
 * units, and each of those nodes pointing down to the layer below it.
 */
place_t * parseHPTDoc(xmlNode * hptNode) {
	xmlChar * version = xmlGetProp(hptNode, xmlCharStrdup("version"));
	xmlChar * info = xmlGetProp(hptNode, xmlCharStrdup("info"));
	xmlFree(version);
	xmlFree(info);

	xmlNode *child = hptNode->xmlChildrenNode;

	place_t * hpt = NULL;
	place_t * pllast = NULL;

    // Iterative over top-level place nodes in the XML file
	while (child != NULL) {
		if (!xmlStrcmp(child->name, (const xmlChar *) "place")) {
			place_t * tmp = parsePlaceElement(child);
			tmp->parent = NULL;
			if (hpt == NULL) {
				hpt = tmp;
				pllast = tmp;
			} else {
				pllast->nnext = tmp;
				pllast = tmp;
			}
		}
		child = child->next;
	}
	return hpt;
}

typedef struct place_node {
	place_t * data;
	struct place_node * next;
}place_node_t;

typedef struct worker_node {
	hc_workerState *data;
	struct worker_node * next;
}worker_node_t;

void find_leaf(place_t * pl, place_node_t **pl_list_cur, worker_node_t ** wk_list_cur) {
	place_t * child = pl->child;
	if (child == NULL) {
		place_node_t *new_pl = (place_node_t*)malloc(sizeof(place_node_t));
		new_pl->data = pl;
		new_pl->next = NULL;
		(*pl_list_cur)->next = new_pl;
		*pl_list_cur = new_pl;
	} else {
		while (child != NULL) {
			find_leaf(child, pl_list_cur, wk_list_cur);
			child = child->nnext;
		}
	}

	hc_workerState * wk = pl->workers;
	while(wk != NULL){
		worker_node_t *new_ws = (worker_node_t*) malloc(sizeof(worker_node_t));
		new_ws->next = NULL;
		new_ws->data = wk;
		(*wk_list_cur)->next = new_ws;
		*wk_list_cur = new_ws;
		wk = wk->nnext;
	}
}

place_t * clonePlace(place_t *pl, int * num_pl, int * num_wk);
void setupWorkerHptPath(hc_workerState * worker, place_t * pl);

void unrollHPT(place_t *hpt, place_t *** all_places, int * num_pl, int * nproc, hc_workerState *** all_workers, int * num_wk) {
	int i;
	place_node_t leaf_places = {NULL, NULL};
	worker_node_t leaf_workers = {NULL, NULL};
	place_node_t * pl_list_cur = &leaf_places;
	worker_node_t * wk_list_cur = &leaf_workers;
	place_t * plp = hpt;
	while(plp != NULL) {
		find_leaf (plp, &pl_list_cur, &wk_list_cur);
		plp = plp->nnext;
	}

	*num_wk = 0;
	*num_pl = 0;
	*nproc = 0;
	/* unroll workers */
	wk_list_cur = leaf_workers.next;
	while (wk_list_cur != NULL) {
		hc_workerState * ws = wk_list_cur->data;
		int num = ws->id;
		for (i=0; i<num-1; i++) {
			hc_workerState * tmp = (hc_workerState*) malloc(sizeof(hc_workerState));
			tmp->pl = ws->pl;
			tmp->did = ws->did + num - i - 1; /* please note the way we add to the list, and the way we allocate did */
			tmp->nnext = ws->nnext;
			ws->nnext = tmp;
		}
		(*nproc) += num;
		ws->pl->ndeques = ws->pl->ndeques + num;
		worker_node_t * tmpwknode = wk_list_cur;
		wk_list_cur = wk_list_cur->next;
		free(tmpwknode);
	}

	/* unroll places, we need to clone HPT subtree when we duplicate a place*/
	place_node_t * lastpl = pl_list_cur;
	pl_list_cur = leaf_places.next;
	while (pl_list_cur != NULL) {
		place_t * pl = pl_list_cur->data;

		/* we first check whether this is the last one of its parent that needs unrolling */
		place_t * parent = pl->parent;
		int unfinished = 0;
		if (parent != NULL) {
			place_t * pllast = parent->child;

			while (pllast != NULL) {
				if (pllast->id != -1) unfinished ++;
				pllast = pllast->nnext;
			}
		}

		/* now do the clone and link them as sibling */
		(*num_pl) ++;
		if (pl->parent != NULL) pl->parent->ndeques += pl->ndeques; /*////////////////////////////////////// This will add too many deques */
		int num = pl->id;
		pl->id = -1; /* clear this out, we only reset this after processing the whole HPT */
		for (i=0; i<num-1; i++) {
			place_t * clpl = clonePlace(pl, num_pl, nproc);
			if (pl->parent != NULL) pl->parent->ndeques += pl->ndeques; /*////////////////////////////////////// This will add too many deques */

			/* now they are sibling */
			clpl->did = pl->did + num - i - 1; /* please note the way we add to the list, and the way we allocate did */
			clpl->nnext = pl->nnext;
			pl->nnext = clpl;
		}

		if (unfinished == 1) {/* the one we just unrolled should be this one */
			place_node_t *new_pl = (place_node_t*)malloc(sizeof(place_node_t));
			new_pl->data = parent;
			new_pl->next = NULL;
			lastpl->next = new_pl;
			lastpl = new_pl;
		}

		place_node_t * tmppl = pl_list_cur;
		pl_list_cur = tmppl->next;
		free(tmppl);
	}

	/* re-sequence the unrolled HPT with a breadth-first traversal */
	place_node_t * start = NULL;
	place_node_t * end = NULL;
	place_node_t * tmp;
	plp = hpt;

	*all_places = (place_t **)malloc(sizeof(place_t *) * (*num_pl));
	int num_dev_wk = 0;
	/* allocate worker objects for CPU workers */
	*all_workers = (hc_workerState **)malloc(sizeof(hc_workerState *) * (*nproc));

	while (plp != NULL) {
		tmp= (place_node_t*) malloc(sizeof(place_node_t));
		tmp->data = plp;
		if (start == NULL) start = tmp;
		else end->next = tmp;
		end = tmp;
		plp->level = 1;
		plp = plp->nnext;
	}
	if (end != NULL) end->next = NULL;

	int wkid = 0;
	int plid = 0;
	while(start != NULL) {
		plp = start->data;
		if (plp->level == 0) {
			fprintf(stderr, "Place level is not set!\n");
			return;
		}
		(*all_places)[plid] = plp;
		if (is_device_place(plp)) num_dev_wk++;
		plp->id = plid++;
		plp->nChildren = 0;
		place_t * child = plp->child;
		while (child != NULL) {
			tmp = (place_node_t*) malloc(sizeof(place_node_t));
			tmp->data = child;
			tmp->next = NULL;
			end->next = tmp;
			end = tmp;
			plp->nChildren++;
			child->level = plp->level + 1;
			child = child->nnext;
		}

		plp->children = (place_t **)malloc(sizeof(place_t *) * plp->nChildren);
		child = plp->child;
		for (i = 0; i < plp->nChildren; i++) {
			plp->children[i] = child;
			child = child->nnext;
		}

		hc_workerState * ws = plp->workers;
		while (ws != NULL) {
			(*all_workers)[wkid] = ws;
			ws->id = wkid ++;
			setupWorkerHptPath(ws, plp);
			ws = ws->nnext;
		}
		tmp = start->next;
		free(start);
		start = tmp;
	}
	if (num_dev_wk) {
		*num_wk = *nproc + num_dev_wk;
		hc_workerState ** tmp = *all_workers;
		*all_workers = (hc_workerState **)malloc(sizeof(hc_workerState *) * (*num_wk));
		memcpy(*all_workers, tmp, sizeof(hc_workerState*)*(*nproc));
		free(tmp);

		/* link the device workers with the place and the all_workers list */
		for (i = 0; i < num_dev_wk; i++) {
			hc_workerState * ws = (hc_workerState*) malloc(
					sizeof(hc_workerState));
			ws->id = *nproc + i;
			(*all_workers)[ws->id] = ws;
			ws->nnext = NULL;
		}
		if (num_dev_wk) {
			int index = 0;
			for (i = 0; i < *num_pl; i++) {
				place_t * dev_pl = ((*all_places)[i]);
				if (is_device_place(dev_pl)) {
					dev_pl->workers = (*all_workers)[*nproc + index];
					(*all_workers)[*nproc + index]->pl = dev_pl;
					(*all_workers)[*nproc + index]->did = dev_pl->did;
					setupWorkerHptPath((*all_workers)[*nproc + index], dev_pl);
					index++;
				}
			}
		}
	} else {
		*num_wk = *nproc;
	}
}

place_t * clonePlace(place_t *pl, int * num_pl, int * nproc) {
	place_t * clone = (place_t*) malloc(sizeof(place_t));
	clone->type = pl->type;
	clone->psize = pl->psize;
	clone->unitSize = pl->unitSize;
	clone->id = pl->id;
	clone->did = pl->did;
	clone->ndeques = pl->ndeques;
	clone->parent = pl->parent;
	clone->child = NULL;
	clone->workers = NULL;
	clone->level = 0;
	clone->nChildren = 0;
	clone->children = NULL;
	place_t * child = pl->child;
	place_t * pllast = NULL;
	while(child != NULL) {
		place_t * tmp = clonePlace(child, num_pl, nproc);
		tmp->parent = clone;
		if (clone->child == NULL) clone->child = tmp;
		else pllast->nnext = tmp;
		pllast = tmp;

		child = child->nnext;
	}
	if (pllast != NULL) pllast->nnext = NULL;

	hc_workerState * ws = pl->workers;
	hc_workerState * wslast = NULL;

	while (ws != NULL) {
		hc_workerState * tmp = (hc_workerState *) malloc(sizeof(hc_workerState));
		tmp->pl = clone;
		tmp->did = ws->did;
		if (clone->workers == NULL) clone->workers = tmp;
		else wslast->nnext = tmp;
		wslast = tmp;
		(*nproc) ++;
		ws = ws->nnext;
	}
	if (wslast != NULL) wslast->nnext = NULL;
	(*num_pl) ++;

	return clone;
}

void setupWorkerHptPath(hc_workerState * worker, place_t * pl) {
	int i;
	if (pl == NULL) return;
	int level = pl->level;
	worker->hpt_path = (place_t **)malloc(sizeof(place_t *) * (level+2));
	worker->hpt_path[level+1] = pl;
	for (i = level; i > 0; i--) {
		if (pl == NULL) {
			fprintf(stderr, "Place level does not match with tree depth!\n");
			return;
		}
		worker->hpt_path[i] = pl;
		pl = pl->parent;
	}
	worker->hpt_path[0] = worker->hpt_path[1];
}

/*
 * The hierarchical place tree for a shared memory system is specified as an XML
 * file containing the hierarchy of memory spaces it contains. In general, this
 * hierarchy is multi-layered and may include a separate layer for system
 * memory, L3 caches, L2 caches, L1 caches, etc.
 *
 * Each layer is described by two properties: a type and a count/num. For
 * example, the type may be "mem" whose contents are explicitly managed by the
 * programmer, or a "cache" whose contents are automatically managed. The count
 * refers to the fan-out at a given level. For example, a machine with main
 * memory feeding to two sockets each with their own L3 cache would have a count
 * of 2 for the L3 layer.
 *
 * An example HPT specification is below. This specification describes a machine
 * with a single system memory, two L3 caches, each of which fans out to 6 L1/L2
 * caches.
 *
 * <HPT version="0.1" info="2 hex core2 Intel Westmere processors">
 *   <place num="1" type="mem">
 *     <place num="2" type="cache"> <!-- 2 sockets with common L3 in each -->
 *       <place num="6" type="cache"> <!-- 6 L2/L1 cache per socket -->
 *         <worker num="1"/> 
 *       </place>
 *     </place>
 *   </place>
 * </HPT>
 *
 * While homogeneous systems will usually have this singly-nested structure, a
 * machine with multiple types of memory sitting below system memory may have
 * multiple elements at the same nesting level, e.g. both L3 and GPU device
 * memory at the same level in the hierarchy.
 *
 * readhpt parses one of these XML files and produces the equivalent place
 * hierarchy, returning the root of that hierarchy.
 */
place_t* readhpt(place_t *** all_places, int * num_pl, int * nproc,
        hc_workerState *** all_workers, int * num_wk) {
    const char *filename = getenv("HCPP_HPT_FILE");
    HASSERT(filename);

    /* create a parser context */
    xmlParserCtxt* ctxt = xmlNewParserCtxt();
    if (ctxt == NULL) {
        fprintf(stderr, "Failed to allocate parser context\n");
        return NULL;
    }
    /* parse the file, activating the DTD validation option */
    xmlDoc* doc = xmlCtxtReadFile(ctxt, filename, NULL, XML_PARSE_DTDVALID);
    /* check if parsing succeeded */
    if (doc == NULL) {
        fprintf(stderr, "Failed to parse %s\n", filename);
        return NULL;
    }

    /* check if validation suceeded */
    if (ctxt->valid == 0) {
        fprintf(stderr, "Failed to validate %s\n", filename);
        return NULL;
    }

    xmlNode *root_element = xmlDocGetRootElement(doc);

    place_t *hpt = parseHPTDoc(root_element);
    unrollHPT(hpt, all_places, num_pl, nproc, all_workers, num_wk);

    /*free the document */
    xmlFreeDoc(doc);

    /* free up the parser context */
    xmlFreeParserCtxt(ctxt);
    return hpt;
}

void freeHPT(place_t * hpt) {
	place_node_t * start = NULL;
	place_node_t * end = NULL;
	place_node_t * tmp;
	place_t *plp = hpt;

	while (plp != NULL) {
		tmp = (place_node_t*)  malloc(sizeof(place_node_t));
		tmp->data = plp;
		if (start == NULL)
			start = tmp;
		else
			end->next = tmp;
		end = tmp;
		plp = plp->nnext;
	}
	if (end != NULL) end->next = NULL;

	while (start != NULL) {
		plp = start->data;
		place_t * child = plp->child;
		while (child != NULL) {
			tmp = (place_node_t*) malloc(sizeof(place_node_t));
			tmp->data = child;
			tmp->next = NULL;
			end->next = tmp;
			end = tmp;
			child = child->nnext;
		}

		hc_workerState * ws = plp->workers;
		while (ws != NULL) {
			hc_workerState * tmpwk = ws;
			ws = ws->nnext;
			free(tmpwk);
		}
		tmp = start->next;
		free(start);
		free(plp);
		start = tmp;
	}
}

}
