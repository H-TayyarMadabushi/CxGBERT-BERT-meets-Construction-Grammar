import os
import sys

import pickle
import argparse

from tqdm          import tqdm

def parse_args() : 
    parser = argparse.ArgumentParser( description='Read parsed WikiText-25 data and tag with constructions.' ) 
    parser.add_argument('in_path' , type=str, help='path, including file name of parsed training data from WikiText-25')
    parser.add_argument('out_path', type=str, help='output path')
    parser.add_argument('workers' , type=int, help='CPUs to use')
    parser.add_argument('done' , type=int, default=-1, help='index of previoudly done shard (-1 for none)')
    args = parser.parse_args()
    return args.in_path, args.out_path, args.workers, args.done

def split_docs( sentences ) : 
    ## There is a blank line between documents
    docs = list()
    this_doc = list()
    for sent in tqdm( sentences, total=len(sentences), desc="Splitting into documents" ) : 
        if sent == '' : 
            if len( this_doc ) > 0 : 
                # Should be the case except the first time. 
                docs.append( this_doc ) 
                this_doc = list()
        else :
            this_doc.append( sent ) 
    return docs

def compress_docs( sentences ) : 
    new_sentences = list()
    for sent in tqdm( sentences, total=len(sentences), desc="Compress documents" ) : 
        if sent == '' : 
            sent = 'MY NEW LINE, MY NEW LINE, MY NEW LINE'
        new_sentences.append( sent ) 
    return new_sentences

        
def getsents( in_path ) : 
    sentences = pickle.load( open( in_path, 'rb' ) )
    return compress_docs( sentences ) 
    # return split_docs( sentences ) 

# https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def tag_and_write( lines, workers, out_path, shard_at=100000, shards_done=-1 ) :
    
    if len( lines ) > shard_at :
        shards = chunks( lines, shard_at )
        processed = 0
        for index, shard in enumerate( shards ) : 

            if index <= shards_done : 
                print( "Skipping ", index, " previously done" )
                continue
                
            processed += len( shard ) 
            
            from c2xg import C2xG
            CxG = C2xG( './data/', 'eng', False, '', '', 'eng.Grammar.v2.p'  )
    
            tagged_lines = CxG.parse_return( shard, 'lines', workers ) 
            outfile      = out_path + 'sents-cxg-tagged-' + str( index ) + '.pk'
            pickle.dump( tagged_lines, open( outfile, 'wb' ) )
            print( "Wrote File: ", outfile )

            if processed > 500000 : 
                print( "Memory overload possible, quitting." ) 
                sys.exit()

    else : 
        from c2xg import C2xG
        CxG = C2xG( './data/', 'eng', False, '', '', 'eng.Grammar.v2.p'  )
    
        tagged_lines = CxG.parse_return( lines, 'lines', workers ) 
        outfile      = out_path + 'sents-cxg-tagged.pk'
        pickle.dump( tagged_lines, open( outfile, 'wb' ) )
        print( "Wrote File: ", outfile )
    return 

if __name__ == '__main__' : 

    in_path, out_path, workers, done = parse_args() 
    tag_and_write( getsents( in_path ), workers, out_path, 100000, done )
