import os
import sys
import pickle
import random
import argparse

from tqdm                      import tqdm
from collections               import defaultdict
from scipy.sparse              import coo_matrix
from sklearn.feature_selection import VarianceThreshold

random.seed( 42 ) 

class createCxGData: 
    def __init__( self ) : 
        self.parse_args()
        return 

    def parse_args( self ) : 
        parser = argparse.ArgumentParser(description='Create CxG version of WikiText-25 for BERT pre-train.')
        parser.add_argument('cxg_location'     , type=str, help='path, including file name of cxg pickle data')
        parser.add_argument('text_location'    , type=str, help='path, including file name of full original data.')
        parser.add_argument('out_path'         , type=str, help='output path')
        
        parser.add_argument('--cxg_freq'       , action='store_true', help='if set, will print number of sentences in each construction and quit'                               , default=False ) 

        parser.add_argument('--do_feat_select' , action='store_true', help='if set, will pick a subset of cxg based on counts (see -feat_max, min). (default false)'            , default=False ) 
        parser.add_argument('--feat_max'  , type=int                , help='Will ignore constructions with more than this many sentences (default 100)'                         , default=1000  ) 
        parser.add_argument('--feat_min'  , type=int                , help='Will ignore constructions with less than this many sentences (default   0)'                         , default=50    ) 

        parser.add_argument('--force_feat_sel' , action='store_true', help='if set, recalculate features even if pickled version exists. (default false)'                       , default=False ) 
        
        parser.add_argument('--cxg_split'      , action='store_true', help='are the cxg pickle files split?'                                                                    , default=False )

        parser.add_argument('--start'          , type=int, help='if split, start of cxg file name of cxg should have HASH where numbers between start and end will be replaced.', default=None ) 
        parser.add_argument('--end'            , type=int, help='if split, end of cxg file name of cxg should have HASH where numbers between start and end will be replaced.'  , default=None ) 
        
        parser.add_argument('--run_name'       , type=str, help='Some name to identify output files with - will be appended to end of output files. (default '' )'              , default='' ) 
         

        self.args = parser.parse_args()

        return 


    def _load( self, location, is_pkl=False ) : 
        if is_pkl : 
            return pickle.load( open( location, 'rb' ) ) 
        return open( location ).read()

    def get_cxg_locations( self ) : 

        start = self.args.start
        end   = self.args.start
        
        if start is None or end is None : 
            print( "ERROR: You must set start and end when using --cxg_split" ) 
            sys.exit()
        
        return [ str( i ).join( self.args.cxg_location.split( '#' ) ) for i in range( self.args.start, self.args.end + 1 ) ]

            

    def _load_cxg_locations( self, cxg_locations ) :
        if len( cxg_locations ) == 1 : 
            return self._load( cxg_locations[0], True )
            
        cxg_data = list() 
        for elem in tqdm( cxg_locations, desc="Loading CxG files" ) : 
            cxg_data += self._load( elem, True ) 
        return  cxg_data
    
    def load_cxg_data( self, force_single=False ) : 
        cxg_locations = [ self.args.cxg_location ]
        if self.args.cxg_split : 
            cxg_locations = self.get_cxg_locations()
        if force_single : 
            cxg_locations = cxg_locations[:1] 
        self.cxg_list = self._load_cxg_locations( cxg_locations )


    def load_text_data( self ) : 
        print( "Loading Text ... ", end="" ) 
        sys.stdout.flush()
        text = self._load( self.args.text_location, False ).split( '\n' )[:-1] ## Remove trailing empty line
        print( "Done." )
        sys.stdout.flush()

        if len( self.cxg_list ) != len( text ) :
            if len( self.cxg_list ) < len( text ) :
                print( "WARNING: Truncating Text to CxG" ) 
                text = text[ :len(self.cxg_list) ]
            else : 
                raise Exception( "CxG longer than text!" ) 
        assert len( self.cxg_list ) == len( text ) 
        self.text = text
        return
        
    def compile_data( self, include_text ) : 
        cxg  = self.cxg_list
        text = None
        if include_text :
            text = self.text
            assert len( cxg ) == len( text ) 
        
        data = defaultdict( list ) 
        for index in tqdm( range( len( cxg ) ), desc="Compile CxG and Text" ) : 
            if include_text and text[ index ] == '' :
                continue
            if len( cxg[ index ] ) == 0 : 
                continue
            for construction_no in cxg[ index ] : 
                if include_text :
                    data[ construction_no ].append( text[ index ] )
                else : 
                    data[ construction_no ].append( 1 ) 
        self.cxg_dict    = data
        self.sorted_cxgs = sorted( data.keys(), key=lambda x:len(data[x]), reverse=True )
        return

    def write_base_data( self, prune_picked_sents=False ) : 

        data        = self.text
        
        if prune_picked_sents : 
            data         = list()
            prev_newline = True
            for index in tqdm( range( len( self.text ) ), desc="Prune data for base" ) : 
                this_is_newline = False
                if self.text[ index ] == '' or not self.included_sentences[ index ] : 
                    this_is_newline = True
                if this_is_newline and prev_newline : 
                    continue
                line = self.text[ index ] 
                if not self.included_sentences[ index ] : 
                    line = ''
                data.append( line ) 
                prev_newline = this_is_newline
        data_len    = len( data ) 
        multiplier  = int( self.picked_cxg_sents / data_len )
        part_mult   = ( self.picked_cxg_sents / data_len ) - multiplier
        trun_index  = int( data_len * part_mult )
        new_data    = data[ : trun_index ]
        
        data        = data * multiplier
        data       += new_data
            
        
        file_info = '_all_' 
        if prune_picked_sents : 
            file_info = '_cxg_only_'

        outfile = os.path.join( self.args.out_path, 'base' + file_info + 'train_data_' + self.args.run_name + '.txt' ) 
        with open( outfile, 'w' ) as fh : 
                fh.write( '\n'.join( data ) ) 
        print( "Wrote base train data to: ", outfile ) 

        print( "Shuffling Base Data ... ", end="" ) 
        sys.stdout.flush()
        random.shuffle( data ) 
        print( "Done." )
        outfile = os.path.join( self.args.out_path, 'rand' + file_info + 'train_data_' + self.args.run_name + '.txt' ) 
        with open( outfile, 'w' ) as fh : 
                fh.write( '\n'.join( data ) ) 
        print( "Wrote rand train data to: ", outfile ) 

            
    def write_cxg_data( self ) : 

        ## Used to make sure we pick same number of base sentences
        self.picked_cxg_sents = 0
        ## Used to make sure we pick same base sentences
        self.included_sentences = list()
        
        ## Each picked cxg is a "document" and there is an extra one for the rst.
        cxg_text = [ [] for i in range( len( self.selected_features ) + 1 ) ] 
        for index in tqdm( range( len( self.text ) ), desc="Creating CxG Output" ) : 
            this_sent_picked = False
            this_cxgs = self.cxg_list[ index ]
            for cxg in this_cxgs : 
                if cxg in self.selected_features : 
                    cxg_text[ self.selected_features.index( cxg ) ].append( self.text[ index ] )
                    self.picked_cxg_sents += 1
                    this_sent_picked = True
            if not this_sent_picked : 
                cxg_text[-1].append( self.text[ index ] )
                self.picked_cxg_sents += 1
            self.included_sentences.append( this_sent_picked )
                    
        for index in tqdm( range( len( cxg_text ) ), desc="Shuffle CxG docs" ) : 
            random.shuffle( cxg_text[ index ] ) 

        outfile = os.path.join( self.args.out_path, 'cxg_only_train_data_' + self.args.run_name + '.txt' )
        with open( outfile, 'w' ) as fh : 
            for doc in tqdm( cxg_text[:-1], desc="Writing CxG Only data" ) : 
                fh.write( '\n'.join( doc ) ) 
                fh.write( '\n' )
        print( "Wrote CxG Only train data to: ", outfile ) 

        outfile = os.path.join( self.args.out_path, 'cxg_all_train_data_' + self.args.run_name + '.txt' )
        with open( outfile, 'w' ) as fh : 
            for doc in tqdm( cxg_text, desc="Writing CxG all data" ) : 
                fh.write( '\n'.join( doc ) ) 
                fh.write( '\n' )
        print( "Wrote CxG all train data to: ", outfile ) 
            

    def print_cxg_freq( self ) : 
        output    = ''
        output_pk = list()
        for elem in self.sorted_cxgs : 
            output += str( elem ) + ", " + str(len( self.cxg_dict[ elem ] )) + "\n"
            output_pk.append( ( elem, len( self.cxg_dict[ elem ] ) ) )
        outfile = self.args.out_path + 'frequencies'
        with open( outfile + '.csv', 'w' ) as fh : 
            fh.write( output )
        import pickle
        pickle.dump( output_pk, open( outfile + '.pk', 'wb' ) ) 
        print( "Wrote ", outfile, ".pk and .txt" )
        return

    def do_feat_select( self ) : 

        store_features = os.path.join( self.args.out_path, 'selected_features_' + self.args.run_name + '.pk' )
        if os.path.exists( store_features ) and not self.args.force_feat_sel : 
            print( "WARNING: Loading precalculated picked features, use --force_feat_sel to recalculate." )
            self.selected_features = pickle.load( open( store_features, 'rb' ) ) 
            return 

        self.selected_features = list()
        for construction in self.sorted_cxgs :
            if len( self.cxg_dict[ construction ] ) > self.args.feat_max :
                continue
            if len( self.cxg_dict[ construction ] ) < self.args.feat_min : 
                break
            self.selected_features.append( construction ) 

        pickle.dump( self.selected_features, open( store_features, 'wb' ) ) 

        print( "Wrote to ", store_features )

        print( "Picked constructions: ", len( self.selected_features ) )
        sent_count  = 0
        check_on    = 50000
        if check_on > len( self.text ) :
            check_on = len( self.text ) 
        for index in tqdm( range( check_on ), desc="Prune Sents" ) : 
            if any( [ ( i in self.selected_features ) for i in self.cxg_list[ index ] ] ) : 
                sent_count += 1

        print( "Sentences that contain picked constructions: ", sent_count ) 
        print( "%: ", ( sent_count / check_on ) * 100 )

        print( "Will exit, simply rerun to use precalculated features." ) 
        sys.exit()
            

        ## Feature Selection using VarianceThreshold, no longer used. 

        """

        ## Must now change format
        ##   X : Features (0 not included, 1 included)
        ##   y : Labels, # of construction


        X, y = list(), list()
        total_constructions = 22628 ## V2 from C2xG github repo
        this_rows, this_cols, this_data = list(), list(), list()
        for index in tqdm( range( len( self.text ) ), desc="Feature Selection Prep" ) : 
            for contained_cxg in self.cxg_list[ index ] : 
                this_rows.append( index         ) 
                this_cols.append( contained_cxg ) 
                this_data.append( 1 ) 
        features = coo_matrix(
            (this_data, (this_rows, this_cols)), 
            shape=( len( self.text ), total_constructions )
        ) 

        
        threshold = 0.99
        selector = VarianceThreshold(threshold=( threshold * (1 - threshold) ) ) # 99% (see: https://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold )
        selector.fit( features )
        picked_cxgs = selector.get_support()
        picked_cxgs = [ i[0] for i in enumerate( picked_cxgs ) if i[1] ] 

        self.selected_features = picked_cxgs
        pickle.dump( picked_cxgs, open( store_features, 'wb' ) ) 

        print( "Wrote to ", store_features )

        print( "Picked constructions: ", len( picked_cxgs ) )
        sent_count  = 0
        for index in tqdm( range( len( self.text ) ), desc="Prune Sents" ) : 
            if any( [ ( i in picked_cxgs ) for i in self.cxg_list[ index ] ] ) : 
                sent_count += 1
        print( "Sentences that contain picked constructions: ", sent_count ) 

        print( "Will exit, simply rerun to use precalculated features." ) 
        sys.exit()
        """

    def main( self ) : 

        print()
        print( "Running with args: ", self.args ) 
        
        extract_features_from_single_file = False
        self.load_cxg_data( force_single=extract_features_from_single_file )

        if self.args.cxg_freq : 
            self.compile_data( include_text=False ) 
            self.print_cxg_freq()
            return

        self.load_text_data()
        self.compile_data( include_text=True ) 
        if self.args.do_feat_select : 
            self.do_feat_select()
        else : 
            self.selected_features = self.cxg_list
        
        if extract_features_from_single_file : 
            self.load_cxg_data()
            self.load_text_data()
            self.compile_data( include_text=True ) 
            
        self.write_cxg_data()
        self.write_base_data( prune_picked_sents=False )
        self.write_base_data( prune_picked_sents=True  )
        
        return

if __name__ == '__main__' : 

    data_creator = createCxGData()
    data_creator.main()

    ## Base: 1000, 50 on 500,000
