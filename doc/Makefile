help:
	@echo "Makefile to build SciGPU-GEMM documentation"

docs:
	doxygen scigpugemm.doxy

docs-public:
	( cat scigpugemm.doxy ; echo "EXTRACT_PRIVATE=NO" ; echo "INTERNAL_DOCS=NO" ) | doxygen -

clean:
	-rm -rf latex/
	-rm -rf html/
