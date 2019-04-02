#!/usr/bin/env python
# coding=utf-8
import sys
import traceback

if len(sys.argv) != 2:
    print 'usage: %s input'
    sys.exit(0)

input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[1]+".clean", "w")

for line in input_file:
    clean_line = line.replace("\/", "/")
    output_file.write("%s" % clean_line)



