#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Wrong number of parameters"
    echo "Usage: download.sh <starttime> <endtime>"
    exit 1
fi

start=`date -Iseconds -d $1`
end=`date -Iseconds -d $2`

echo $start
echo $end

d="$start"
while [ "$d" != "$end" ]; do
    echo $d
    dat=`date -d "$d" "+%d-%m-%Y"`
    tim=`date -d "$d" "+%H-%M"`
    wget "http://neige.meteociel.fr/satellite/archives/$dat/satir-$tim.gif" \
        -O `date -d "$d" "+%Y-%m-%d-%H-%M.jpg"`
    d=`date -Iseconds -d "$d + 15 minutes"`
    sleep $[ ( $RANDOM % 3 )  + 1 ]s
    # http://neige.meteociel.fr/satellite/archives/04-04-2016/satir-02-30.gif
done
