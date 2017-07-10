/*
 * Function to evaluate argv[]. specs is a 0 terminated array of command 
 * line options and types an array that stores the type as one of 
 */
#define INTARG 1
#define DOUBLEARG 2
#define LONGARG 3
#define BOOLARG 4
#define STRINGARG 5
#define BENCHMARK 6
/*
 * for each specifier. Benchmark is specific for cilk samples. 
 * -benchmark or -benchmark medium sets integer to 2
 * -benchmark short returns 1
 * -benchmark long returns 2
 * a boolarg is set to 1 if the specifier appears in the option list ow. 0
 * The variables must be given in the same order as specified in specs. 
 */

void get_options(int argc, char *argv[], const char *specs[], int *types,...);
extern int hc_rand(void);
