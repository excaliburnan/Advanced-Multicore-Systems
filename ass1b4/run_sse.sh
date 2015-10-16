#!/bin/sh

matrix_size=$1

echo "compile application"

gcc -g -lm -msse -fopenmp  matrix.c -o matrix.exe

echo "executing the application"
./matrix.exe $matrix_size

rm -fr *~ matrix.exe
