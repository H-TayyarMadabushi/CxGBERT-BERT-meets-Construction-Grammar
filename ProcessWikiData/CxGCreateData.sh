export min=10
export max=500
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
