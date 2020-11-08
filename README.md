# CxGBERT: BERT meets Construction Grammar

This repository contains data and code associated with the publication  *CxGBERT: BERT meets Construction Grammar"* (COLING 2020). 

In the paper we show that **BERT has access to a information that linguists typically call constructional**. 

## What is Construction Grammar

Construction Grammars are a set of linguistic theories that take constructions as their primary object of study. Constructions are either patterns that are frequently occurring or their meaning is not predictable from the sum of their parts (e.g. Idioms). This study focuses more on constructions that consist of frequently occuring patterns. 

These patterns can be  Low level and simple: Noun +s (Plural Construction) 
Another low level construction is *The Xer the Yer*. Some sentences that are instances of this construction include *a) The more you think about it, the less you understand, b) The Higher You Fly, the Further You Fall*

An example of a higher level pattern is  *Personal Pronoun + didn’t + V + how* and sentences that are instances of this construction include: *a) She didn’t understand how I could do so poorly, and b) One day she picked up a book and as she opened it, a white child took it away from her, saying she didn’t know how to read.*

There are also *schematic constructions* such as the *Caused-Motion Construction*. Some instances: *a) Norman kicked the ball into the room. b) Mary hit the ball out of the park. c) She sneezed the foam off the cappuccino.*

## Why Construction Grammar and BERT?

Consider the final example in the previous section *She sneezed the foam off the cappuccino.* It has been shown that humans can understand sentences with such novel uses of words (or novel words) using the construction the sentences are instances of. 

Given that BERT has access to PoS information, parse trees, and mBERT can even reproduce labels for syntactic dependencies that largely agree with universal grammar, we ask: **How much constructional information does BERT have access to?**

## Preprocessing

There are three preprocessed data sets that are required for the experiments described. 

 1. The tagging of WikiText-103 with constructional information.
 2. Creating pre-training data for CxGBERT, BERT Base Clone and BERT
 3. Random Creating data to evaluate BERT on Constructions.

