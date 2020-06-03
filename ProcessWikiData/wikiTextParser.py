import os
import sys

import pickle
import argparse

from tqdm          import tqdm
from nltk.tokenize import sent_tokenize

def readin( path ) : 
    num_lines = sum(1 for line in open( path,'r') )
    read_data = list()
    with open( path ) as fh : 
        for line in tqdm( fh, total=num_lines, desc="Reading Data" ): 
            sents = sent_tokenize( line ) 
            sents = [ i.lstrip().rstrip() for i in sents ]
            sents = [ i for i in sents if i != '' ]
            if len( sents ) == 0 : 
                continue
            if len( sents ) == 1 : 
                if sents[0][0] == sents[0][-1] == '=' : 
                    ## New sentence
                    read_data.append( '' )
            read_data += sents
    return read_data
        
        

def parse_args() : 
    parser = argparse.ArgumentParser(description='Parse WikiText-25 into BERT pre-train format.')
    parser.add_argument('in_path', type=str, help='path, including file name of training data from WikiText-25')
    parser.add_argument('out_path', type=str, help='output path')
    args = parser.parse_args()
    return args.in_path, args.out_path


def writeout( sents, out_path ) : 
    pickle.dump( sents, open( out_path + 'sentences.pk', 'wb' ) )
    with open( out_path + 'sentences.txt', 'w' ) as fh : 
        for sent in tqdm( sents, total=len( sents ), desc="Writing out" ) : 
            fh.write( sent ) 
            fh.write( '\n' ) 
    return
    

if __name__ == '__main__' : 

    in_path, out_path = parse_args() 
    writeout( readin( in_path ), out_path )
