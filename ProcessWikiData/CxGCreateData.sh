export min=10
export max=500
if [ $# != 0 ] 
then
    export min=$1
    export max=$2
fi
echo "Running with min of $min and max of $max, will pause to allow termination"
sleep 1

export end=45
export outdir=processed-$min-$max

python createCxGData.py                                \
    ../../WikiText-103/processed/sents-cxg-tagged-#.pk \
    ../../WikiText-103/processed/sentences.txt         \
    ../../WikiText-103/$outdir/                        \
    --do_feat_select      \
    --cxg_split           \
    --start 0             \
    --end           $end  \
    --run_name $min-$max-45 
