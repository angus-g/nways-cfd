# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

CC=	nvc++
CFLAGS=	-lm -I/apps/nvidia-hpc-sdk/21.3/Linux_x86_64/21.3/cuda/11.2/include -L/apps/nvidia-hpc-sdk/21.3/Linux_x86_64/21.3/cuda/11.2/lib64 -lnvToolsExt -cuda
LFLAGS=


# System independent definitions

MF=	Makefile

EXE=	cfd

INC= \
	arraymalloc.h \
	boundary.h \
	cfdio.h \
	jacobi.h

SRC= \
	arraymalloc.cu \
	boundary.cu \
	cfd.cu \
	cfdio.cu \
	jacobi.cu

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .cu .o

OBJ=	$(SRC:.cu=.o)

.cu.o:
	$(CC) $(CFLAGS) -c $<

all:	$(EXE)

$(OBJ):	$(INC)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

tar:
	tar cvf cfd.tar $(MF) $(INC) $(SRC)

clean:
	rm -f $(OBJ) $(EXE) velocity.dat colourmap.dat cfd.plt core
