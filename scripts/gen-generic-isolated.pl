#!/usr/bin/perl -w

if($#ARGV != 0) {
	print "USAGE: perl gen-generic-isolated.pl <Total Args>\n";
 	exit();
}
####################################################
#inline void isolated(void* object0, void* object1, std::function<void()> &&lambda) {
#  const int n = 2;
#  void *array[n];
#  array[0] = object0;
#  array[1] = object1;
#  isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
#}
####################################################

for (my $j=0; $j<$ARGV[0]; $j++) {
        print "inline void isolated(void* object0";

        #Printing the remaining objects
        for (my $i=1; $i<=$j; $i++) {
                print ", void* object$i";
        }
        print ", std::function<void()> &&lambda) {\n";

        my $total = $j + 1;
        print "\tconst int n = $total;\n";
        print "\tvoid *array[n];\n";

        for (my $i=0; $i<=$j; $i++) {
                print "\tarray[$i] = object$i; \n";
        }
        print "\tisolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);\n";
        print "}\n";
}
