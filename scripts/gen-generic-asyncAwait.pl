#!/usr/bin/perl -w

#########################################
# Author: Vivek Kumar (vivekk@rice.edu)
#########################################

if($#ARGV != 0) {
	print "USAGE: perl gen-generic-asyncAwait.pl <Total Args>\n";
 	exit();
}
####################################################
#template <typename T>
#void asyncAwait(hclib_promise_t* promise0, T lambda) {
#	int promises = 2;
#	hclib_promise_t** promiseList = (hclib_promise_t**) malloc(sizeof(hclib_promise_t *) * promises);
#	promiseList[0] = promise0; 
#	promiseList[1] = NULL; 
#	_asyncAwait<T>(promiseList, lambda);
#}
#template <typename T>
#void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, T lambda) {
#	int promises = 3;
#	hclib_promise_t** promiseList = (hclib_promise_t**) malloc(sizeof(hclib_promise_t *) * promises);
#	promiseList[0] = promise0; 
#	promiseList[1] = promise1; 
#	promiseList[2] = NULL; 
#	_asyncAwait<T>(promiseList, lambda);
#}
####################################################

print "#include \"hclib-promise.h\"\n";
print "#include \"hclib-async.h\"\n";
print "\n";
print "namespace hclib {\n";
print "\n";

for (my $j=0; $j<$ARGV[0]; $j++) {
	print "template <typename T>\n";
	print "void asyncAwait(hclib_promise_t* promise0";

	#Printing the hclib_promise_t parameters
	for (my $i=1; $i<=$j; $i++) {
  		print ", hclib_promise_t* promise$i";
	}
	print ", T lambda) {\n";

	my $promises = $j + 2;
	print "\tint promises = $promises;\n";
	print "\thclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);\n";

	for (my $i=0; $i<=$j; $i++) {
  		print "\tpromiseList[$i] = promise$i; \n";
	}
	$promises = $promises - 1;
	print "\tpromiseList[$promises] = NULL; \n";


	print "\thclib::_asyncAwait<T>(promiseList, lambda);\n";
	print "}\n";
}
print "\n";
print "}\n";
print "\n";
