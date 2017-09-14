#!/bin/bash
rm results.txt
echo "limit 50" >> results.txt
python knn.py --limit 50 | grep Accuracy >> results.txt
echo "limit 100" >> results.txt
python knn.py --limit 100 | grep Accuracy >> results.txt
echo "limit 500" >> results.txt
python knn.py --limit 500 | grep Accuracy >> results.txt
echo "limit 1000" >> results.txt
python knn.py --limit 1000 | grep Accuracy >> results.txt
echo "limit 5000" >> results.txt
python knn.py --limit 5000 | grep Accuracy >> results.txt
echo "limit 10000" >> results.txt
python knn.py --limit 10000 | grep Accuracy >> results.txt
echo "limit 50000" >> results.txt
python knn.py --limit 50000 | grep Accuracy >> results.txt
