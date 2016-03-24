rm cholesky.out
HCLIB_WORKERS=2 ./cholesky 500 20 ./input/m_500.in
if [ `diff cholesky.out input/cholesky_out_500.txt | wc -l` -eq 0 ]; then
	echo "Test=Success"
else
	echo "Test=Fail"
fi
