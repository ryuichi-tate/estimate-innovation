#!/usr/bin/bash
for i in "22" "23" "24" "25"
do
	python learn_triple_ver6.py --data_seed $i
done
