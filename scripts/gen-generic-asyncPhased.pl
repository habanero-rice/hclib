#!/usr/bin/perl -w

if($#ARGV != 0) {
	print "USAGE: perl gen-generic-asyncPhased.pl <Total Args>\n";
 	exit();
}
####################################################
#  template<typename T>
#  void asyncPhased(PHASER_t ph0, PHASER_m m0, PHASER_t ph1, PHASER_m m1, T lambda) {
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
#    _asyncPhased<T>(&phased, lambda);
#  }
####################################################

for (my $j=0; $j<$ARGV[0]; $j++) {
        print "template <typename T>\n";
        print "void asyncPhased(PHASER_t ph0, PHASER_m m0 ";

        #Printing the promise parameters
        for (my $i=1; $i<=$j; $i++) {
                print ", PHASER_t ph$i, PHASER_m m$i";
        }
        print ", T lambda) {\n";

        my $total = $j + 1;
        print "\tint total = $total;\n";
        print "\tPHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);\n";
        print "\tPHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);\n";

        for (my $i=0; $i<=$j; $i++) {
                print "\tphaser_type_arr[$i] = ph$i; \n";
                print "\tphaser_mode_arr[$i] = m$i; \n";
        }
	print "\tphased_t phased;\n";
	print "\tphased.count = total;\n";
	print "\tphased.phasers = phaser_type_arr;\n";
	print "\tphased.phasers_mode = phaser_mode_arr;\n";
        print "\t_asyncPhased<T>(&phased, lambda);\n";
        print "}\n";
}
