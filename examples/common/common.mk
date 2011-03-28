# Common makefile for SciGPU-GEMM Examples
# Borrows from NVIDIA common.mk

# (c) Copyright 2009 President and Fellows of Harvard College
#
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


ROOTDIR    ?= ../../
SRCDIR     ?= ./
LIBDIR     ?= $(ROOTDIR)/lib/
INCDIR     ?= $(ROOTDIR)/include/
OBJDIR     ?= ./obj/
BINDIR     ?=  $(ROOTDIR)/bin

CCFILES    += gemm_gold.cpp

CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Add some general dependencies
C_DEPS += $(wildcard *.hpp) $(wildcard $(COMMONDIR)/*.hpp)


CXXFLAGS +=
LIB      += -lgsl -lcblas -latlas -lpthread -lcuda -lcublas -L$(LIBDIR) -lscigpugemm


# Includes
INCLUDES  += -I. -I$(INCDIR) -I$(COMMONDIR)



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
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \



# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX -O3 -fno-strict-aliasing


TARGET   := $(BINDIR)/$(EXECUTABLE)
LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB)



# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################

CXXFLAGS  += $(COMMONFLAGS) $(CXXWARN_FLAGS)
CFLAGS    += $(COMMONFLAGS) $(CWARN_FLAGS)


################################################################################
# Set up object files
################################################################################
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.o,$(notdir $(CFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.o : $(SRCDIR)%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.o : $(SRCDIR)%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

# Rule for sources in the common directory
$(OBJDIR)/%.o : $(COMMONDIR)/%.cpp
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TARGET): makedirectories $(OBJS) lib Makefile
	$(VERBOSE)$(LINKLINE)


makedirectories:
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(BINDIR)


tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)

lib :
	$(VERBOSE)make -C $(ROOTDIR) lib

clobber : clean
	$(VERBOSE)rm -rf $(OBJDIR)

nuke: clobber
	-rm -rf ${BINDIR}
