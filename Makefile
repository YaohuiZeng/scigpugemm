# General Build file for SciGPU-GEMM

# (c) Copyright 2009 President and Fellows of Harvard College

#  This file is part of the SciGPU-GEMM Library
#
#  SciGPU-GEMM is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SciGPU-GEMM is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with SciGPU-GEMM.  If not, see <http://www.gnu.org/licenses/>.

EXAMPLES := $(shell find examples -name Makefile)

%.ph_build : lib
	make -C $(dir $*)

%.ph_clean :
	make -C $(dir $*) clean

%.ph_clobber :
	make -C $(dir $*) clobber

%.ph_nuke :
	make -C $(dir $*) nuke

all: $(addsuffix .ph_build,$(EXAMPLES)) docs lib
	@echo "Finished building all"


docs:
	make -C doc/ docs 

lib:
	make -C src/

tidy:
	@find | egrep "#" | xargs rm -f
	@find | egrep "\~" | xargs rm -f

nuke: $(addsuffix .ph_nuke,$(EXAMPLES))
	make -C src/ nuke
	make -C doc/ clean
	make tidy
