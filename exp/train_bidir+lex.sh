#!/usr/bin/env bash
workspace=`pwd`
datadir=$workspace/../data
tooldir=../bidir+lex/root_model/treelstm/
function run()
{
    nohup $tooldir/$1 $datadir/glove.sentiment.large.pretrained.vec $datadir/trees/train.txt.clean $datadir/trees/dev.txt.clean $datadir/trees/test.txt.clean 1>$workspace/$2.log 2>&1 &
#   gdb --args $tooldir/$1 $datadir/glove.sentiment.large.pretrained.vec $datadir/trees/train.txt.clean $datadir/trees/dev.txt.clean $datadir/trees/test.txt.clean
}
run BiTreeSentimentZhu  9.bidir.adam.6.replicated
exit
for i in {1..3}
do
    run BiTreeSentimentZhu  9.bidir.adam.$i
done
wait
for i in {4..6}
do
    run BiTreeSentimentZhu  9.bidir.adam.$i
done
wait
for i in {7..10}
do
    run BiTreeSentimentZhu  9.bidir.adam.$i
done
wait
