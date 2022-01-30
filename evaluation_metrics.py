from gingerit.gingerit import GingerIt
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import string
from typing import List
from helpers import get_word_ngrams
from numpy import mean
import pandas as pd
from bert_score import score
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

#scorer = BERTScorer(lang="en", rescale_with_baseline=True)

def spelling_error(text: str) -> float:
    """
    Description
    -----------
    Metric return the number of spelling errors maded in a string. Does not
    take into account the relevance, grammatical errors, or weird sentences.
    Is quite slow...
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

def sentence_bleu_score(references: List[str], candidate: str) -> float:
    """
    Description
    -----------
    Calculating the bleu score for a single sample and its reference
    Adapted from: https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1
    Documentation: https://www.kite.com/python/docs/nltk.bleu_score.corpus_bleu
    For tips in report: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Indifferent to punctuation  
    """
    
    # remove punctuation
    references = [r.translate(str.maketrans('', '', string.punctuation)).lower() for r in references]
    candidate = candidate.translate(str.maketrans('', '', string.punctuation)).lower()
    
    ref_bleu = [r.split() for r in references]
    cand_bleu = candidate.split()
        
    cc = SmoothingFunction()
    
    # references = list(str) e.g. can be [review1, review2, review3], 
    # hypothesis = str e.g. can be 'This is a cat'
    return sentence_bleu(ref_bleu, cand_bleu, smoothing_function=cc.method2)

def mean_sentence_bleu(references: List[List[str]], candidates: List[str]) -> float:
    """
    Description
    -----------
    Different from corpus_bleu_score as it averages scores after division, 
    not before as in corpus_bleu
    """
    assert len(references) == len(candidates), "Must have the same number of reference lists and hypotheses"
    
    return mean([sentence_bleu_score(r, c) for r, c in zip(references, candidates)])
    

def corpus_bleu_score(references: List[List[str]], candidates: List[str]) -> float:
    """
    Description
    -----------
    Calculating the bleu score across all samples, summing across the samples before division
    Adapted from: https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1
    Documentation: https://www.kite.com/python/docs/nltk.bleu_score.corpus_bleu
    For tips in report: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Indifferent to punctuation
    """
    
    # Remove punctuation and split into words while maintaining shape
    cand_bleu = [c.translate(str.maketrans('', '', string.punctuation)).split() for c in candidates]
    ref_bleu = [[r.translate(str.maketrans('', '', string.punctuation)).split() for r in refs] for refs in references]
    
    cc = SmoothingFunction()
    
    # references = list(list(str)) e.g. [[ref1.1, ref1.2], [ref2.1]]
    # hypothesis = list(str) e.g. ['some generated sentence', 'another sentence']
    return corpus_bleu(ref_bleu, cand_bleu, smoothing_function=cc.method2)
    

def sentence_rouge_score(references: List[str], candidate: str, n: int = 2) -> float:
	"""
	Computes ROUGE-N of two text collections of sentences.
	Source: http://research.microsoft.com/en-us/um/people/cyl/download/
	papers/rouge-working-note-v1.3.1.pdf
	Args:
		candidate: The generated sentence
		references: The sentences from the reference set
		n: Size of ngram.  Defaults to 2.
	Returns:
		recall rouge score(float)
    """
	evaluated_ngrams = get_word_ngrams(n, [candidate])
	reference_ngrams = get_word_ngrams(n, references)
	reference_count = len(reference_ngrams)
      
	# Gets the overlapping ngrams between evaluated and reference
	overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
	overlapping_count = len(overlapping_ngrams)
	
	if reference_count == 0:
		return 0.0
	return overlapping_count / reference_count # Recall


def corpus_rouge_score(references: List[List[str]], candidates: List[str], n: int = 2) -> float:
    
    assert len(references) == len(candidates), "Must have the same number of reference lists and hypotheses"
    
    result = []
    for refs, cand in zip(references, candidates):
        result.append(sentence_rouge_score(refs, cand, n))
        
    return mean(result)

def sentence_bert_score(references: List[str], candidates: str, metric: str = 'p') -> float:
    """
    Description
    ----------
    https://towardsdatascience.com/machine-translation-evaluation-with-sacrebleu-and-bertscore-d7fdb0c47eb3
    https://github.com/Tiiiger/bert_score
    """
    #if type(candidates) == str:
    #    return scorer.score([candidates], [references])
    precision, recall, f1 = score(candidates, references, lang="en")

    if metric == 'p':
        return precision.mean()
    if metric == 'r':
        return recall.mean()
    if metric == 'f':
        return f1.mean()

def corpus_bert_score(references: List[List[str]], candidates: List[str], metric: str = 'p') -> float:
    """
    Description
    ----------
    https://towardsdatascience.com/machine-translation-evaluation-with-sacrebleu-and-bertscore-d7fdb0c47eb3
    """
    
    assert len(references) == len(candidates), "Must have the same number of reference lists and hypotheses"
    
    #if type(candidates) == str:
    #    return scorer.score([candidates], [references])
    precision, recall, f1 = score(candidates, references, lang="en")
    
    if metric == 'p':
        return precision.mean()
    if metric == 'r':
        return recall.mean()
    if metric == 'f':
        return f1.mean()

def determine_baseline(data: pd.DataFrame, function) -> float:
    """
    Description
    -----------
    Iterate over reviews, use other reviews of same product as reference and 
    review in question as generated text. Return average score
    """
    refs = []
    hyps = []
    
    for i in range(len(data)):
        
        review = data.loc[i]
        references = data[(data.asin == review.asin) & (data.reviewText != review.reviewText)]
        references = references.reviewText.to_list()
        
        refs.append(references)
        hyps.append(review.reviewText)

    return function(refs, hyps)     


ref = "The NASA Opportunity rover is battling a massive dust storm on Mars"
cand1 = "The Opportunity rover is combating a big sandstorm on Mars"
cand2 = "A NASA rover is fighting a massive storm on Mars"

ref1a = 'It is a guide to action that ensures that the military will forever heed Party commands.'
ref1b = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
ref1c = 'It is the practical guide for the army always to heed the directions of the party'
ref2a = 'he was interested in world history because he read the book'

hyp1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
hyp2 = 'he read the book because he was interested in world history'

refs = [['The dog bit the guy.', 'The dog had bit the man.'],
        ['It was not unexpected.', 'No one was surprised.'],
        ['The man bit him first.', 'The man had bitten the dog.']]

hyps = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

if __name__ == '__main__':
    
    "Example scores"

    print('b ', sentence_bleu_score( [ref1a, ref1b, ref1c], hyp1))
    print('b ', sentence_bleu_score( [ref2a], hyp2))
    print('r ', sentence_rouge_score([ref1a, ref1b, ref1c], hyp1))
    print('r ', sentence_rouge_score([ref2a], hyp2))
    print('bs', sentence_bert_score( [ref1a, ref1b, ref1c], hyp1))
    print('bs', sentence_bert_score( [ref2a], hyp2))
    print()
    print('cb ', corpus_bleu_score( [[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2]))
    print('cbs', corpus_bert_score( [[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2]))
    print('cr ', corpus_rouge_score([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2]))

    # "Determine baselines"
    # print('b bl ', determine_baseline(data, corpus_bleu_score))  # 0.11775293410059064
    # print('r bl ', determine_baseline(data, corpus_rouge_score)) # 0.008019608502661493
    # print('bs bl', determine_baseline(data, corpus_bert_score))  # 0.0899



    
    
