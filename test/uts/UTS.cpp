/*
 *         ---- The Unbalanced Tree Search (UTS) Benchmark ----
 *  
 *  Copyright (c) 2010 See AUTHORS file for copyright holders
 *
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.  See AUTHORS file for more information.
 *
 */

#include "crt.h"
#include "uts.h"

int nproc;
counter_t* nNodes;
counter_t* nLeaves;
counter_t* maxTreeDepth;
double walltime;

/***********************************************************
 *  UTS Implementation                                     *
 ***********************************************************/

void ss_init() {
	nproc = crt::numWorkers();
	nNodes = new counter_t[nproc];
	nLeaves = new counter_t[nproc];
	maxTreeDepth = new counter_t[nproc];

	for (int i = 0; i < nproc; i++) {
		nNodes[i] = 0;
		nLeaves[i] = 0;
		maxTreeDepth[i] = 0;
	}
}

/*
 * parallel search of UTS trees locally
 *
 */

void parTreeSearch(Node parent)
{
	int status;
	int parentHeight;
	int numChildren;
	int childType;
	int wid = crt::get_hc_wid();

	maxTreeDepth[wid] = max(maxTreeDepth[wid], parent.height);
	nNodes[wid]++;

	parentHeight = parent.height;
	numChildren = uts_numChildren(&parent);
	childType   = uts_childType(&parent);
	parent.numChildren = numChildren;

	if (numChildren > 0) {
		for (int i = 0; i < numChildren; i++) {
			Node child;
			child.type = childType;
			child.height = parentHeight + 1;
			for (int j = 0; j < computeGranularity; j++) {
				// TBD:  add parent height to spawn
				// computeGranularity controls number of rng_spawn calls per node
				rng_spawn(parent.state.state, child.state.state, i);
			}
			crt::async([child]() {
				parTreeSearch(child);
			});
		}
	} else {
		nLeaves[wid]++;
	}
}

void ss_start() {
	Node root;

	/* initialize root node and push on thread 0 stack */
	uts_initRoot(&root, type);

	/* time parallel search */
	double t1 = uts_wctime();

	crt::start_finish();
	//crt::finish([root]() {
		parTreeSearch(root);
	//});
	crt::end_finish();

	double t2 = uts_wctime();
	walltime = t2 - t1;
}

void showStats() {
	counter_t tnodes = 0, tleaves = 0, mheight = maxTreeDepth[0];
	double elapsedSecs = walltime;

	// combine measurements from all local threads
	for (int i = 0; i < nproc; i++) {
		tnodes  += nNodes[i];
		tleaves += nLeaves[i];
		if(i+1 < nproc) {
			mheight  = max(mheight, maxTreeDepth[i+1]);
		}
	}

	uts_showStats(nproc, elapsedSecs, tnodes, tleaves, mheight);
}

void ss_finalize() {
	delete nNodes;
	delete nLeaves;
	delete maxTreeDepth;
	walltime = 0;
}

int main(int argc, char *argv[]) 
{
	crt::init(&argc, argv);
	/* determine benchmark parameters */
	uts_parseParams(argc, argv);

	/* show parameter settings */
	uts_printParams();

	/* initialize */
	ss_init();

	/* Go */
	ss_start();

	/* display results */
	showStats();

	ss_finalize();

	int pi, s, po;
        crt::gather_commWorker_Stats(&po, &pi, &s);
        std::cout << "Pushes = " << pi << " Steals = " << s << std::endl;

	crt::finalize();
	return 0;
}

char * ss_get_par_description() {
	return "HabaneroUPC++ par";
}

void ss_abort(int error) {
	exit(-1);
}

/* Fatal error */
void ss_error(char *str, int error) {
	ss_abort(error);
}

int ss_get_num_threads() {
	return nproc;
}
