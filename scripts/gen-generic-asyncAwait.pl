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
#void asyncAwait(hclib_ddf_t* ddf0, T lambda) {
#	int ddfs = 2;
#	hclib_ddf_t** ddfList = (hclib_ddf_t**) malloc(sizeof(hclib_ddf_t *) * ddfs);
#	ddfList[0] = ddf0; 
#	ddfList[1] = NULL; 
#	_asyncAwait<T>(ddfList, lambda);
#}
#template <typename T>
#void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, T lambda) {
#	int ddfs = 3;
#	hclib_ddf_t** ddfList = (hclib_ddf_t**) malloc(sizeof(hclib_ddf_t *) * ddfs);
#	ddfList[0] = ddf0; 
#	ddfList[1] = ddf1; 
#	ddfList[2] = NULL; 
#	_asyncAwait<T>(ddfList, lambda);
#}
####################################################

print "#include \"hclib-ddf.h\"\n";
print "#include \"hclib-async.h\"\n";
print "\n";
print "namespace hclib {\n";
print "\n";

for (my $j=0; $j<$ARGV[0]; $j++) {
	print "template <typename T>\n";
	print "void asyncAwait(hclib_ddf_t* ddf0";

	#Printing the hclib_ddf_t parameters
	for (my $i=1; $i<=$j; $i++) {
  		print ", hclib_ddf_t* ddf$i";
	}
	print ", T lambda) {\n";

	my $ddfs = $j + 2;
	print "\tint ddfs = $ddfs;\n";
	print "\thclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);\n";

	for (my $i=0; $i<=$j; $i++) {
  		print "\tddfList[$i] = ddf$i; \n";
	}
	$ddfs = $ddfs - 1;
	print "\tddfList[$ddfs] = NULL; \n";


	print "\thclib::_asyncAwait<T>(ddfList, lambda);\n";
	print "}\n";
}
print "\n";
print "}\n";
print "\n";
