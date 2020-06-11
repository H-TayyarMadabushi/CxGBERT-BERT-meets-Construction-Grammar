export outdir=processed10-500
python createCxGData.py                                \
    ../../WikiText-103/processed/sents-cxg-tagged-#.pk \
    ../../WikiText-103/processed/sentences.txt         \
    ../../WikiText-103/$outdir/                        \
    --do_feat_select      \
    --cxg_split           \
    --start 0             \
    --end   45            \
    --run_name 10000_50_45 
