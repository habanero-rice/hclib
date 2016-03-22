#!/usr/bin/perl -w

if($#ARGV != 0) {
	print "USAGE: perl gen-generic-asyncAwait.pl <Total Args>\n";
 	exit();
}
####################################################
#namespace hcpp
#{
#  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, std::function<void()> &&lambda);
#}
####################################################

print "namespace hcpp { \n";

for (my $j=0; $j<$ARGV[0]; $j++) {
print "  void asyncPhased(DDF_t* ddf0";

#Printing the DDF_t parameters
for (my $i=1; $i<=$j; $i++) {
  print ", DDF_t* ddf$i";
}
print ", std::function<void()> &&lambda);\n";
}
print "}\n";
