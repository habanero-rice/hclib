#!/usr/bin/perl -w

if($#ARGV != 0) {
	print "USAGE: perl gen-generic-asyncAwait.pl <Total Args>\n";
 	exit();
}
####################################################

print "namespace hcpp { \n";

print "  void asyncPhased(std::function<void()> &&lambda); \n";
for (my $j=0; $j<$ARGV[0]; $j++) {
print "  void asyncPhased(PHASER_t* ph0, PHASER_m m0";

for (my $i=1; $i<=$j; $i++) {
  print ", PHASER_t* ph$i";
  print ", PHASER_m m$i";
}
print ", std::function<void()> &&lambda);\n";
}
print "}\n";
