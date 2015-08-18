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
#void asyncAwait(DDF_t* ddf0, T lambda) {
#	int ddfs = 2;
#	DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
#	ddfList[0] = ddf0; 
#	ddfList[1] = NULL; 
#	_asyncAwait<T>(ddfList, lambda);
#}
#template <typename T>
#void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, T lambda) {
#	int ddfs = 3;
#	DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
#	ddfList[0] = ddf0; 
#	ddfList[1] = ddf1; 
#	ddfList[2] = NULL; 
#	_asyncAwait<T>(ddfList, lambda);
#}
####################################################

for (my $j=0; $j<$ARGV[0]; $j++) {
	print "template <typename T>\n";
	print "void asyncAwait(DDF_t* ddf0";

	#Printing the DDF_t parameters
	for (my $i=1; $i<=$j; $i++) {
  		print ", DDF_t* ddf$i";
	}
	print ", T lambda) {\n";

	my $ddfs = $j + 2;
	print "\tint ddfs = $ddfs;\n";
	print "\tDDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);\n";

	for (my $i=0; $i<=$j; $i++) {
  		print "\tddfList[$i] = ddf$i; \n";
	}
	$ddfs = $ddfs - 1;
	print "\tddfList[$ddfs] = NULL; \n";


	print "\t_asyncAwait<T>(ddfList, lambda);\n";
	print "}\n";
}
