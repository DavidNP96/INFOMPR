from gingerit.gingerit import GingerIt
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import string
from typing import List
import pyter
from helpers import get_word_ngrams
from numpy import mean
import pandas as pd
from numba import cuda
from numba import jit
import pysbd
print(cuda.gpus)

def spelling_error(text: str) -> int:
    """
    Description
    -----------
    Metric return the number of spelling errors maded in a string. Does not
    take into account the relevance, grammatical errors, or weird sentences.
    Is quite slow...

    Parameters
    ----------
    text : str
        Generated text to evaluatue.

    Returns
    -------
    int
        Number of spelling erros.
    """
    errors = 0
    parser = GingerIt()
    for i in range(0, len(text), 299):
        try:
            t = text[i:i+300]
        except:
            t = text[i:]
        result = parser.parse(t)['corrections']
        errors+= len(result)

    return 1 - errors/len(text.split())

def sentence_bleu_score(ref: List[str], gen: str) -> float:
    """
    Description
    -----------
    Calculating the bleu score for a single sample and its reference
    Adapted from: https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1
    Documentation: https://www.kite.com/python/docs/nltk.bleu_score.corpus_bleu
    For tips in report: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Indifferent to punctuation  
    """
    
    """remove punctuation"""
    ref = [r.translate(str.maketrans('', '', string.punctuation)).lower() for r in ref]
    gen = gen.translate(str.maketrans('', '', string.punctuation)).lower()
    
    ref_bleu = [r.split() for r in ref]
    gen_bleu = gen.split()
        
    cc = SmoothingFunction()
    
    """references = list(str) e.g. can be [review1, review2, review3], 
    hypothesis = str e.g. can be 'This is a cat'"""
    return sentence_bleu(ref_bleu, gen_bleu, smoothing_function=cc.method2)

def mean_sentence_bleu(ref: List[List[str]], gen: List[str]) -> float:
    """
    ""
    Description
    -----------
    Different from corpus_bleu_score as it averages scores after division, 
    not before as in corpus_bleu
    """
    assert len(ref) == len(gen)
    
    return mean([sentence_bleu_score(r, g) for r, g in zip(ref, gen)])
    

def corpus_bleu_score(ref: List[List[str]], gen: List[str]) -> float:
    """
    Description
    -----------
    Calculating the bleu score across all samples, summing across the samples before division
    Adapted from: https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1
    Documentation: https://www.kite.com/python/docs/nltk.bleu_score.corpus_bleu
    For tips in report: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Indifferent to punctuation
    """
    gen = [g.translate(str.maketrans('', '', string.punctuation)) for g in gen]
    
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
        
    for refs in ref:
        refs = [r.translate(str.maketrans('', '', string.punctuation)) for r in refs]
        ref_bleu.append([r.split() for r in refs])
        
    cc = SmoothingFunction()
    
    """references = list(list(str)) e.g. [[ref1.1, ref1.2], [ref2.1]]
    hypothesis = list(str) e.g. ['some generated sentence', 'another']"""
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, smoothing_function=cc.method2)
    
    return score_bleu

def rouge_score(references: List[str], generated: List[str], n=2):
	"""
	Computes ROUGE-N of two text collections of sentences.
	Source: http://research.microsoft.com/en-us/um/people/cyl/download/
	papers/rouge-working-note-v1.3.1.pdf
	Args:
		evaluated_sentences: The sentences that have been picked by the summarizer
		reference_sentences: The sentences from the referene set
		n: Size of ngram.  Defaults to 2.
	Returns:
		recall rouge score(float)
    """
	evaluated_ngrams = get_word_ngrams(n, generated)
	reference_ngrams = get_word_ngrams(n, references)
	reference_count = len(reference_ngrams)
      
	# Gets the overlapping ngrams between evaluated and reference
	overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
	overlapping_count = len(overlapping_ngrams)
	
	if reference_count == 0:
		return 0.0
	return overlapping_count / reference_count # Recall


def ter_score(references: List[str], generated: List[str]):
    '''
    averaged TER score over all sentence pairs
    '''
    assert len(references) == len(generated)
    
    total_score = 0
    for i in range(len(generated)):
        total_score = total_score + pyter.ter(generated[i], references[i])
    total_score = total_score/len(generated)
    return total_score

def determine_baseline(data: pd.DataFrame, function) -> float:
    """
    Description
    -----------
    Iterate over reviews, use other reviews of same product as reference and 
    review in question as generated text. Return average score
    """
    result = []
    for i in range(len(data)):
        try:
            if (i%1000 == 0):
                print(i)
            review = data.loc[i]
            references = data[(data.asin == review.asin) & (data.reviewText != review.reviewText)]
            references = references.reviewText.to_list()
            i+= len(references)
            result.append(function(references, review.reviewText))
        except Exception as e:
            print(e)
    
    return mean(result)
        
        

ref = "The NASA Opportunity rover is battling a massive dust storm on Mars"
cand1 = "The Opportunity rover is combating a big sandstorm on Mars"
cand2 = "A NASA rover is fighting a massive storm on Mars"

ref1a = 'It is a guide to action that ensures that the military will forever heed Party commands.'
ref1b = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
ref1c = 'It is the practical guide for the army always to heed the directions of the party'
ref2a = 'he was interested in world history because he read the book'

hyp1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
hyp2 = 'he read the book because he was interested in world history'


if __name__ == '__main__':
    
    "Example scores"
    print('se', spelling_error(cand1))
    print('se', spelling_error(cand2))
    print('sb', sentence_bleu_score([ref1a, ref1b, ref1c], hyp1))
    print('sb', sentence_bleu_score([ref2a], hyp2))
    print('msb', mean_sentence_bleu([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2]))
    print('cb', corpus_bleu_score([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2]))
    print('r',  rouge_score([ref1a, ref1b, ref1c], [hyp1]))
    print('r',  rouge_score([ref2a], [hyp2]))
    
    # "Determine baselines"
    # print('cb bl', determine_baseline(data, sentence_bleu_score)) 0.1479257873995623
    # print('r bl', determine_baseline(data, rouge_score)) 0.0006166477313487892
    


    
    
