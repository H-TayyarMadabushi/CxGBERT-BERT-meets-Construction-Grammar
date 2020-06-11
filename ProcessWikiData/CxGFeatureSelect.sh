export outdir=processed10-500
python createCxGData.py                                \
    ../../WikiText-103/processed/sents-cxg-tagged-#.pk \
    ../../WikiText-103/processed/sentences.txt         \
    ../../WikiText-103/$outdir/                        \
    --do_feat_select      \
    --feat_max       500  \
    --feat_min        10  \
    --force_feat_sel      \
    --cxg_split           \
    --start 0             \
    --end   45            \
    --run_name 10000_50_45 
