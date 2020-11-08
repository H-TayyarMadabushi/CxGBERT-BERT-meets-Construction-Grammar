
# Tagging Sentences with Constructions

To tag sentences with constructions we make use of the package c2xg 1.0 with some minor modifications. 

**If you are interested in sentences already tagged with construtional information, see**

For details on how the constructions are identified and tagged, see *[Computational learning of construction grammars](https://www.cambridge.org/core/journals/language-and-cognition/article/computational-learning-of-construction-grammars/43E9BA63CD01CB2912029FF32721076E)*  and the associated program code [c2xg 1.0](https://github.com/jonathandunn/c2xg)

# Tagged Sentences

The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. (See [Here](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) for original release) 

**We release the raw WikiText 103 data with associated constructional information tagged.**

 - You can find files containing 1 sentence per line in [THIS](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/tree/master/C2xG/data/sentences) folder  
 - The pickle (Python 3) files in [THIS](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/tree/master/C2xG/data/cxg) folder contains a List where the
   element in the ith index contains a (second) list of all
   constructions that sentence i from the corresponding file is an
   instance of.


## Data Examples

	  
Here are the first 5 lines of the file sentences00
	
	<BLANK LINE>
	= Valkyria Chronicles III =
	Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァ	ルキュリア3 , lit .
	Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable .
	Released in January 2011 in Japan , it is the third game in the Valkyria series .

Here are the first 5 elements of the array stored in sents-cxg-tagged-0.pk

	[]
	[]
	[]
	[6076, 7976, 15090, 21689, 16, 647, 21305]
	[21913, 7192, 7479, 13952, 14327, 14638, 8559, 16, 647, 21305]

This means that the first three lines (an empty line, the title and the line with Japanese text in it) are instances of no constructions and that line 4 (*Valkyria of the Battlefield 3 ) ...*) is an instance of the constructions numbered *[6076, 7976, 15090, 21689, 16, 647, 21305]*

Note that what these constructions are is not available. 

## Tagging Sentences

Tagging sentences is done using the C2xG with the changes listed in THIS folder. These changes will soon be merged with the original package. These changes do NOT in any way change the actual constructions, they are used only to greatly speed up tagging. 

[tagger.py](https://github.com/H-TayyarMadabushi/CxGBERT-BERT-meets-Construction-Grammar/blob/master/C2xG/tagger.py) can be used to tag sentences in 100,000 shards 
