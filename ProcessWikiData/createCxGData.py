import os
import csv
import sys
import copy
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

        data        = copy.copy( self.text ) 

        ## If prune_picked_sents then cxg_only

        text_articles = len( [ x for x in self.text if x == '' ] ) 
        data_articles = len( [ x for x in data      if x == '' ] ) 

        version_to_use = 1
        if version_to_use == 1 :
            ## Version 1: If there is a missed line, break document
            ignored_doc_breaks = 0 
            added_doc_breaks   = 0
            if prune_picked_sents : 
                data         = list()
                prev_newline = True
                for index in tqdm( range( len( self.text ) ), desc="Prune data for base" ) : 
                    line = self.text[ index ] 
                    this_is_newline = False
                    if self.text[ index ] == '' : 
                        this_is_newline = True
                    if not self.included_sentences[ index ] : 
                        if not this_is_newline : 
                            ## Add new line
                            line = ''
                            this_is_newline = True
                            added_doc_breaks += 1
                    if this_is_newline and prev_newline : 
                        ignored_doc_breaks += 1
                        continue
                    data.append( line ) 
                    prev_newline = this_is_newline

                ## Sanity Check
                data_articles = len( [ x for x in data      if x == '' ] ) 
                data_articles = data_articles  -  added_doc_breaks + ignored_doc_breaks


        if version_to_use == 2 : 
            ## Version 2: Ignore missed lines
            if prune_picked_sents : 
                data         = list()
                for index in tqdm( range( len( self.text ) ), desc="Prune data for base" ) : 
                    this_is_newline = False
                    if self.text[ index ] == '' :
                        data.append( self.text[ index ] ) 
                    if not self.included_sentences[ index ] : 
                        continue
                    data.append( self.text[ index ] ) 

                data_articles = len( [ x for x in data      if x == '' ] ) 


        ## Sanity check.
        assert data_articles == text_articles
        
        ## Repeat or truncate to make sure we get the same length as CxG
        
        picked_len  = self.picked_cxg_all_sents
        if prune_picked_sents : 
            picked_len = self.picked_cxg_sents
        data_len    = len( data ) 
        
        if picked_len < data_len : 
            data    = data[ :picked_len ]
        else : 
            multiplier  = int( picked_len / data_len )
            part_mult   =    ( picked_len / data_len ) - multiplier
            trun_index  = int( data_len * part_mult )
            new_data    = data[ : trun_index ]
        
            data        = data * multiplier
            data       += new_data


        ## Create a probe for base.
        probe_name = 'base_all'
        if prune_picked_sents : 
            ## CxG only
            probe_name = 'base_cxg'
        # Create docs
        docs = list()
        this_doc = list()
        for line in data : 
            if line == '' : 
                if len( this_doc ) > 0 : 
                    docs.append( this_doc ) 
                    this_doc = list()
                    continue
            this_doc.append( line ) 
        self._create_probes( docs, probe_name, limit_docs=True ) 


        
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

        
    def _create_probes( self, cxg_text, cxg_what, limit_docs=False ) : 
        
        constructions = len( cxg_text )
        # Train count will be 4 times this. 
        dev_count   = constructions * 2
        if dev_count * 4 < 10000 : 
            dev_count = int( 10000 / 4 ) + 10 ## round up 
        if limit_docs : 
            if dev_count * 4 > 100000 : 
                dev_count = int( 100000 / 4 ) 
        train = list()
        dev   = list()
        test  = list()

        docs  = cxg_text
        
        sent_picked_count = [ 0        for i in docs ]
        sent_lens         = [ len( i ) for i in docs ]
        passes_over_data  = 0 
        while ( len( train ) < dev_count  * 4 ) or \
              ( len( dev   ) < dev_count      ) or \
              ( len( test  ) < dev_count      ) : 
            passes_over_data += 1
            if passes_over_data > 10 and limit_docs : 
                print( "WARNING: Ran out of data to create probes, have train {}, test {}, dev {}. Will continue.".format( len( train ), len( test ), len( dev ) ) ) 
                break
            if passes_over_data > 50 : 
                print( "WARNING: Ran out of data to create probes, have train {}, test {}, dev {}. Will continue.".format( len( train ), len( test ), len( dev ) ) ) 
                break

            for doc_index in tqdm( range( len( docs ) ), desc="Building Probe" ) : 
                ## First add positive
                positives = list()
                for _ in range( 3 ) : 
                    sim           = 1

                    doc_sent_id_1 = str( doc_index ) + "." + str( sent_picked_count[ doc_index ] )
                    if sent_picked_count[ doc_index ] == sent_lens[ doc_index ] :
                        break
                    sent_1        = cxg_text[ doc_index ][ sent_picked_count[ doc_index ] ]
                    sent_picked_count[ doc_index ] += 1

                    if sent_picked_count[ doc_index ] == sent_lens[ doc_index ] :
                        ## "Put back" sent and break
                        sent_picked_count[ doc_index ] -= 1 
                        break
                    doc_sent_id_2 = str( doc_index ) + "." + str( sent_picked_count[ doc_index ] )
                    sent_2        = cxg_text[ doc_index ][ sent_picked_count[ doc_index ] ]
                    sent_picked_count[ doc_index ] += 1

                    positives.append( [ [ sim, doc_sent_id_1, doc_sent_id_2, sent_1, sent_2 ], doc_index ] ) 

                if len( positives ) > 0 : 
                    train.append( positives[0][0] ) 
                if len( positives ) > 1 : 
                    if len( dev ) != dev_count :
                        dev.append(   positives[1][0] )
                    else : 
                        ## "Put back" sents
                        sent_picked_count[ positives[1][1] ] -= 1
                if len( positives ) > 2 : 
                    if len( test ) != dev_count : 
                        test.append(  positives[2] ) 
                    else : 
                        ## "Put back" sents
                        sent_picked_count[ positives[2][1] ] -= 1
                        

                if len( positives ) == 0 : 
                    continue
                ## Now the negatives
                negatives = list()
                for _ in range( 3 ) : 
                    sim           = 0

                    doc_sent_id_1 = str( doc_index ) + "." + str( sent_picked_count[ doc_index ] )
                    if sent_picked_count[ doc_index ] == sent_lens[ doc_index ] :
                        break
                    sent_1        = cxg_text[ doc_index ][ sent_picked_count[ doc_index ] ]
                    sent_picked_count[ doc_index ] += 1
                    
                    doc_sent_id_2   = None
                    sent2           = None
                    for _ in range( 10 ) : 
                        ## Pick a sent from another doc
                        other_doc_index = doc_index
                        while other_doc_index == doc_index : 
                            other_doc_index = random.randrange( 0, constructions )

                        doc_sent_id_2 = str( other_doc_index ) + "." + str( sent_picked_count[ other_doc_index ] )
                        if sent_picked_count[ other_doc_index ] == sent_lens[ other_doc_index ] :
                            continue
                        sent_2        = cxg_text[ other_doc_index ][ sent_picked_count[ other_doc_index ] ]
                        sent_picked_count[ other_doc_index ] += 1
                        need_other_sent = False
                        break

                    negatives.append( [ [ sim, doc_sent_id_1, doc_sent_id_2, sent_1, sent_2 ], doc_index ] )

                        
                if len( negatives ) > 0 : 
                    train.append( negatives[0][0] ) 
                if len( negatives ) > 1 : 
                    if len( dev ) != dev_count : 
                        dev.append(   negatives[1][0] ) 
                    else : 
                        ## "Put back" sents
                        sent_picked_count[ negatives[1][1] ] -= 1
                if len( negatives ) > 2 : 
                    if len( test ) != dev_count : 
                        test.append(  negatives[2][0] ) 
                    else : 
                        ## "Put back" sents
                        sent_picked_count[ negatives[2][1] ] -= 1


                    
        random.shuffle( train ) 
        random.shuffle( dev   ) 
        random.shuffle( test  ) 
        
        # Add header 
        header = [ 'SameCxG', 'DocID.SentID_1', 'DocID.SentID_2' 'Sent_1', 'Sent_2' ]
        train  = [ header ] + train
        dev    = [ header ] + dev 
        test   = [ header ] + test

        for ( data, name ) in [ ( train, 'train' ), ( dev, 'dev' ), ( test, 'test' ) ] : 
            file_name = os.path.join( self.args.out_path, 
                                      cxg_what + '_probe_' + self.args.run_name + '_' + name + '.tsv' )
            with open( file_name, 'w' ) as csvfile :
                writer = csv.writer( csvfile, delimiter='\t' ) 
                writer.writerows( data ) 
            print( "Wrote probe {}".format( file_name ) )
                
                
        return
        
            
    def write_cxg_data( self ) : 

        ## Used to make sure we pick same number of base sentences
        self.picked_cxg_sents      = 0
        self.picked_cxg_all_sents = 0 
        ## Used to make sure we pick same base sentences
        self.included_sentences = list()

        selected_features_dict = dict() ## Do NOT make this defaultdict
        for index in range( len( self.selected_features ) ) : 
            selected_features_dict[ self.selected_features[ index ] ] = index
        
        ## Each picked cxg is a "document" and there is an extra one for the rest.
        cxg_text = [ [] for i in range( len( self.selected_features ) + 1 ) ] 
        for index in tqdm( range( len( self.text ) ), desc="Creating CxG Output" ) : 
            this_sent_picked = False
            this_cxgs = self.cxg_list[ index ]
            for cxg in this_cxgs : 
                cxg_index = None
                try : 
                    cxg_index = selected_features_dict[ cxg ]
                except : 
                    continue
                cxg_text[ cxg_index ].append( self.text[ index ] )
                self.picked_cxg_sents += 1
                self.picked_cxg_all_sents += 1
                this_sent_picked = True
            if not this_sent_picked : 
                if not self.text[ index ] == '' : 
                    cxg_text[-1].append( self.text[ index ] )
                    self.picked_cxg_all_sents += 1
            self.included_sentences.append( this_sent_picked )
                    
        for index in tqdm( range( len( cxg_text ) ), desc="Shuffle CxG docs" ) : 
            random.shuffle( cxg_text[ index ] ) 

        count = 0 
        for index in range( len( cxg_text ) - 1 ) : 
            count += len( cxg_text[ index ] ) 
        assert self.picked_cxg_sents == count 
        count += len( cxg_text[ -1 ] ) 
        assert self.picked_cxg_all_sents == count 

        outfile = os.path.join( self.args.out_path, 'cxg_only_train_data_' + self.args.run_name + '.txt' )
        with open( outfile, 'w' ) as fh : 
            docs_to_consider = len(cxg_text[:-1])
            for index in tqdm( range( docs_to_consider ), desc="Writing CxG Only data" ) : 
                doc = cxg_text[ index ]
                fh.write( '\n'.join( doc ) ) 
                fh.write( '\n' )
                if index != ( docs_to_consider - 1 ) : 
                    # Avoid trailing newline. 
                    fh.write( '\n' )
        print( "Wrote CxG Only train data to: ", outfile ) 

        outfile = os.path.join( self.args.out_path, 'cxg_all_train_data_' + self.args.run_name + '.txt' )
        with open( outfile, 'w' ) as fh : 
            docs_to_consider = len(cxg_text)
            for index in tqdm( range( docs_to_consider ), desc="Writing CxG all data" ) : 
                doc = cxg_text[ index ]
                fh.write( '\n'.join( doc ) ) 
                fh.write( '\n' )
                if index != ( docs_to_consider - 1 ) : 
                    ## Avoid trailing newline.
                    fh.write( '\n' )
        print( "Wrote CxG all train data to: ", outfile ) 

        self._create_probes( cxg_text[:-1], 'cxg_only' ) 
        self._create_probes( cxg_text     , 'cxg_all'  ) 

        ## Write samples
        self._write_samples( cxg_text[:-1] ) 

        return

    def _write_samples( self, docs ) : 
        
        for file_id, doc in tqdm( enumerate( docs ), total=len( docs ), desc="Writing samples" ) : 
            outfile = os.path.join( self.args.out_path, 'samples', str( file_id ) + '.txt' )
            with open( outfile, 'w' ) as fh : 
                if len( doc ) > 1000 : 
                    doc = doc[:1000]
                fh.write( '\n'.join( doc ) )
        
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
        selected_features_dict = defaultdict( bool )
        for feature in self.selected_features : 
            selected_features_dict[ feature ] = True
        for index in tqdm( range( check_on ), desc="Prune Sents" ) : 
            for cxg in self.cxg_list[ index ] : 
                if selected_features_dict[ cxg ] : 
                    sent_count += 1
                    break

        print( "Sentences that contain picked constructions (out of {0}) :{1} ({2}%)".format( check_on, sent_count, (( sent_count / check_on ) * 100 ) ) ) 

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

        print( "All Done." ) 
        sys.stdout.flush()
        
        return

if __name__ == '__main__' : 

    data_creator = createCxGData()
    data_creator.main()

