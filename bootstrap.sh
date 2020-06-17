#
# Search for libtoolize
#

LIBTOOLIZE=`which libtoolize`

if [ $? -ne 0 ]; then
   LIBTOOLIZE=`which glibtoolize`
fi

if [ $? -ne 0 ]; then
   echo "ERROR: can't find libtoolize nor glibtoolize"
   exit 1
fi


aclocal -I config;

eval "$LIBTOOLIZE --force --copy"

autoreconf -vfi;
