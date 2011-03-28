# Makefile for GEMMcleaver library

# Based in part on the NVIDIA SDK common/common.mk file

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

LIBNAME := libscigpugemm.a

CCFILES    =  gpuerror.cpp\
              gemmcleaver.cpp\
              mgemm.cpp\
              wrappers.cpp


# Redefine locations
ROOTDIR    ?= ..
SRCDIR     ?= $(ROOTDIR)/src/
INCDIR     ?= $(ROOTDIR)/include/
LIBDIR     ?= $(ROOTDIR)/lib/
OBJDIR     ?= $(SRCDIR)/obj/


# Compilers
CXX        := g++
LINK       := g++ -fPIC


# Includes
INCLUDES  += -I$(INCDIR)


# Warning flags
CXXWARN_FLAGS := \
	-W -Wall -Wextra -Weffc++ \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function

# Compiler-specific flags
CXXFLAGS  := $(INCLUDES) $(CXXWARN_FLAGS) -DUNIX -O3 -fno-strict-aliasing

# Do a verbose build by default
verbose    = 1


# Add some general dependencies
CXX_DEPS += $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.h)


LINKLINE = ar rv $(LIBDIR)/$(LIBNAME) $(OBJS)


# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

# Add common flags
CXXFLAGS  += $(COMMONFLAGS)


################################################################################
# Set up object files
################################################################################
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(CCFILES)))

################################################################################
# Rules
################################################################################

$(OBJDIR)/%.o : $(SRCDIR)%.cpp $(CXX_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<


$(LIBNAME): makedirectories $(OBJS) $(CUBINS) Makefile
	$(VERBOSE)$(LINKLINE)



makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(LIBDIR)


tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -rf $(LIBDIR)

clobber : clean
	$(VERBOSE)rm -rf $(OBJDIR)


nuke : clobber
	-rm -rf ${BINDIR}