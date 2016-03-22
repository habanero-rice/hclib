#!/usr/bin/perl -w

if($#ARGV != 0) {
	print "USAGE: perl gen-generic-asyncAwait.pl <Total Args>\n";
 	exit();
}
####################################################
##include "hcpp-utils.h"
#namespace hcpp
#{
#  using namespace std;
#
#  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, std::function<void()> &&lambda)
#  {
#    int ddfs = 2+1;
#    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
#    ddfList[0] = ddf0;
#    ddfList[1] = ddf1;
#    ddfList[2] = NULL;
#    
#    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
#    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
#  }
#
#}
####################################################

print "#include \"hcpp-utils.h\"\n";
print "namespace hcpp { \n";
print "  using namespace std; \n";

for (my $j=0; $j<$ARGV[0]; $j++) {
print "  void asyncAwait(DDF_t* ddf0";

#Printing the DDF_t parameters
for (my $i=1; $i<=$j; $i++) {
  print ", DDF_t* ddf$i";
}
print ", std::function<void()> &&lambda) {\n";

my $ddfs = $j + 2;
print "    int ddfs = $ddfs;\n"; 
print "    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);\n";

for (my $i=0; $i<=$j; $i++) {
  print "    ddfList[$i] = ddf$i; \n";
}
$ddfs = $ddfs - 1;
print "    ddfList[$ddfs] = NULL; \n";

print "    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);\n";
print "    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);\n";
print "  }\n";
}
print "}\n";
