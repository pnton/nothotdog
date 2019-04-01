#!/bin/sh

REQUESTNUMBER=500
MAXCONCURRENTREQUESTS=50

shopt -s globstar
for file in hotdogs/test/1/*.jpg hotdogs/test/2/*.jpg hotdogs/train/1/*.jpg hotdogs/train/2/*.jpg hotdogs/valid/1/*.jpg hotdogs/valid/2/*.jpg ; do
    nohup ab -n $REQUESTNUMBER -c $MAXCONCURRENTREQUESTS -p $file -T application/x-www-form-urlencoded http://nothotdog.elumitas.com:5000/is-hot-dog &
done