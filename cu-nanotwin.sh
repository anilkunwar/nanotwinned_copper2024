#!/bin/bash

a=3.59 #Angstrom
atomsk --create fcc $a Cu orient [11-2] [111] [-110] copper_unit.xsf cfg
atomsk copper_unit.xsf -duplicate 8 5 8 copper_super.xsf cfg

# Apply mirror symmetry
atomsk copper_super.xsf -mirror 0 Y -wrap copper_mirror.xsf
# Merge the original super cell with the mirror supercell
atomsk --merge Y 2 copper_super.xsf copper_mirror.xsf cu_nanotwin.xsf cfg
