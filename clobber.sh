#!/bin/sh

#
# Cleanup the project 
#

rm -Rf compileTree hclib-install

rm -Rf config autom4te.cache aclocal.m4 configure COPYING depcomp config.log config.status libtool

for file in `find src -name Makefile.in`; do
    rm -Rf $file
done

for file in `find src -name Makefile`; do
    rm -Rf $file
done

for file in `find src -regex '.*\.o$'`; do
    rm -Rf $file
done

for file in `find src -regex '.*\.lo$'`; do
    rm -Rf $file
done

for file in `find src -regex '.*\.la$'`; do
    rm -Rf $file
done

for file in `find src -regex '.*deps$'`; do
    rm -Rf $file
done

for file in `find src -regex '.*libs$'`; do
    rm -Rf $file
done

rm -f Makefile Makefile.in 
