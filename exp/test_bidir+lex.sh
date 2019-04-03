#!/usr/bin/env bash
workspace=`pwd`
datadir=$workspace/../data
tooldir=../bidir+lex/root_model/treelstm/
function run()
{
    nohup $tooldir/$1 $datadir/glove.sentiment.large.pretrained.vec $datadir/trees/train.txt.clean $datadir/trees/dev.txt.clean $datadir/trees/test.txt.clean $2 1>$workspace/$3.log 2>&1 &
#   gdb --args $tooldir/$1 $datadir/glove.sentiment.large.pretrained.vec $datadir/trees/train.txt.clean $datadir/trees/dev.txt.clean $datadir/trees/test.txt.clean
}
#change the model name here 
model=bitree.zhu.fix2._300_150_1-pid31746.params
run BiTreeSentimentZhu  $model bidir
