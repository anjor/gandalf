##################################################
#           Makefile of GandAlf 				 #
#                                                #
#  NOTE: environmental variables                 #
#     CUDAARCH, CUDA_INCLUDE                           #
#  need to be properly defined                   #
##################################################
TARGET    = gandalf
CU_DEPS := \
	c_fortran_namelist3.c \
	reduc_kernel.cu \
	device_funcs.cu \
	k_funcs.cu \
	reduc_kernel.cu \
	work_kernel.cu \
	diag_kernel.cu \
	maxReduc.cu \
	init_func.cu \
	nlps_kernel.cu \
	init_kernel.cu \
	timestep_kernel.cu \
	nlps.cu \
	courant.cu \
	timestep.cu \
	sumReduc_nopad.cu \
	forcing.cu \
	damping_kernel.cu \
	zderiv_kernel.cu \
	slowmodes.cu \
	diagnostics.cu
#courant.cu and maxReduc.cu
FLDFOL_DEPS := \
	c_fortran_namelist3.c \
	device_funcs.cu \
	k_funcs.cu \
	fldfol_funcs.cu

DNE_DBPAR_DEPS := \
	c_fortran_namelist3.c \
	device_funcs.cu \
	k_funcs.cu \
	work_kernel.cu \
	nlps_kernel.cu \
	diag_kernel.cu \
	fldfol_funcs.cu 

FILES     = *.cu *.c *.cpp Makefile
VER       = `date +%y%m%d`

NVCC      = nvcc
NVCCFLAGS = -arch=$(CUDAARCH) -use_fast_math 
NVCCINCS  = $(CUDA_INCLUDE)
NVCCLIBS  = -lcufft -lcudart

ifeq ($(debug),on)
  NVCCFLAGS += -g -G2
else
  NVCCFLAGS += -O3
endif

.SUFFIXES:
.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $(NVCCINCS) $< 

# main program
$(TARGET): gandalf.o
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS) 

gandalf.o: $(CU_DEPS)

test_make:
	@echo TARGET=    $(TARGET)
	@echo CUDA_INCLUDE=    $(CUDA_INCLUDE)
	@echo NVCC=      $(NVCC)
	@echo NVCCFLAGS= $(NVCCFLAGS)
	@echo NVCCINCS=  $(NVCCINCS)
	@echo NVCCLIBS=  $(NVCCLIBS)

clean:
	rm -rf *.o *~ \#*

distclean: clean
	rm -rf $(TARGET)

tar:
	@echo $(TARGET)-$(VER) > .package
	@-rm -fr `cat .package`
	@mkdir `cat .package`
	@ln $(FILES) `cat .package`
	tar cvf - `cat .package` | bzip2 -9 > `cat .package`.tar.bz2
	@-rm -fr `cat .package` .package

fldfol: fldfol.o
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS)

fldfol.o:  $(FLDFOL_DEPS)

dne_dbpar: dne_dbpar.o
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS)

dne_dbpar.o:  $(DNE_DBPAR_DEPS)
