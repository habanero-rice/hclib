#!/usr/bin/perl -w

if($#ARGV != 0) {
	print "USAGE: perl gen-generic-asyncPhased.pl <Total Args>\n";
 	exit();
}
####################################################
##include "hcpp-utils.h"
#namespace hcpp { 
#  using namespace std; 
#  void asyncPhased(std::function<void()> &&lambda) {
#    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
#    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, NO_DDF, NO_PHASER, PHASER_TRANSMIT_ALL);
#  }
#............
#............
#............
#  void asyncPhased(PHASER_t ph0, PHASER_m m0, PHASER_t ph1, PHASER_m m1, std::function<void()> &&lambda) {
#    int total = 2;
#    PHASER_t* phaser_type_arr = new PHASER_t[total];
#    phaser_type_arr[0] = ph0;
#    phaser_type_arr[1] = ph1;
#    PHASER_m* phaser_mode_arr = new PHASER_m[total];
#    phaser_mode_arr[0] = m0;
#    phaser_mode_arr[1] = m1;
#    phased_t phased;
#    phased.count = total;
#    phased.phasers = phaser_type_arr;
#    phased.phasers_mode = phaser_mode_arr;
#    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
#    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, NO_DDF, &phased, 0); 
#  }
#}
####################################################

print "#include \"hcpp-utils.h\"\n";
print "namespace hcpp { \n";
print "  using namespace std; \n";
print "  void asyncPhased(std::function<void()> &&lambda) {\n";
print "    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda); \n";
print "    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, NO_DDF, NO_PHASER, PHASER_TRANSMIT_ALL); \n";
print "  } \n";

for (my $j=0; $j<$ARGV[0]; $j++) {
print "  void asyncPhased(PHASER_t* ph0, PHASER_m m0";

#Printing the PHASER_t, PHASER_m parameters
for (my $i=1; $i<=$j; $i++) {
  print ", PHASER_t* ph$i";
  print ", PHASER_m m$i";
}
print ", std::function<void()> &&lambda) {\n";

my $ph = $j + 1;
print "    int total = $ph;\n"; 

print "    PHASER_t* phaser_type_arr = new PHASER_t[total]; \n";
for (my $i=0; $i<=$j; $i++) {
  print "    phaser_type_arr[$i] = *ph$i; \n";
}

print "    PHASER_m* phaser_mode_arr = new PHASER_m[total]; \n";
for (my $i=0; $i<=$j; $i++) {
  print "    phaser_mode_arr[$i] = m$i; \n";
}

print "    phased_t phased;\n";
print "    phased.count = total;\n";
print "    phased.phasers = phaser_type_arr;\n";
print "    phased.phasers_mode = phaser_mode_arr;\n";
print "    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);\n";
print "    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, NO_DDF, &phased, NO_PROP);\n";
print "  }\n";
}
print "}\n";
