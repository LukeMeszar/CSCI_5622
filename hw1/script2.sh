#!/bin/bash
rm results2.txt
echo "k = 1" >> results2.txt
python knn.py --k 1 --limit 600 | grep Accuracy >> results2.txt
echo "k = 3" >> results2.txt
python knn.py --k 3 --limit 600 | grep Accuracy >> results2.txt
echo "k = 5" >> results2.txt
python knn.py --k 5 --limit 600 | grep Accuracy >> results2.txt
echo "k = 7" >> results2.txt
python knn.py --k 7 --limit 600 | grep Accuracy >> results2.txt
echo "k = 9" >> results2.txt
python knn.py --k 9 --limit 600 | grep Accuracy >> results2.txt
echo "k = 11" >> results2.txt
python knn.py --k 11 --limit 600 | grep Accuracy >> results2.txt
echo "k = 13" >> results2.txt
python knn.py --k 13 --limit 600 | grep Accuracy >> results2.txt
