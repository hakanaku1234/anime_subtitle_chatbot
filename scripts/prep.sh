#!/bin/sh

# make a file that has removed the indenting character in beginning of every line
cut -c 2- < input.txt > input2.txt

# remove the original file with indenting character
rm input.txt

# remove last line to make the input file
sed '$d' input2.txt > input.txt

# remove first line to make the output file
tail -n +2 input2.txt > output.txt

rm input2.txt
