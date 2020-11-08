

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
if [ -d ../../WikiText-103/$outdir ] 
then
    echo "Outdir ../../WikiText-103/$outdir exists! Exiting ..."
    exit
fi

mkdir -p ../../WikiText-103/$outdir
mkdir -p ../../WikiText-103/$outdir/samples
python createCxGData.py                                \
    ../../WikiText-103/processed/sents-cxg-tagged-#.pk \
    ../../WikiText-103/processed/sentences.txt         \
    ../../WikiText-103/$outdir/                        \
    --do_feat_select      \
    --feat_max      $max  \
    --feat_min      $min  \
    --force_feat_sel      \
    --cxg_split           \
    --start 0             \
    --end           $end  \
    --run_name $min-$max-45  

