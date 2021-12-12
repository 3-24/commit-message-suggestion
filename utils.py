from rouge import Rouge
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu

class Metric:
    @staticmethod
    def get_rouge_score(gen_commits, real_commits):
        gen_txt = [' '.join(msg) for msg in gen_commits]
        real_txt = [' '.join(msg) for msg in real_commits]
        rouge = Rouge()
        score = rouge.get_scores(gen_txt, real_txt, avg = True)
        return (score["rouge-1"]["f"],score["rouge-2"]["f"],score["rouge-l"]["f"])
    
    @staticmethod
    def duplicate_vocab(gen_commits):
        return sum(map(lambda msg: len(msg)-len(Counter(msg)),gen_commits))/len(gen_commits)
    
    @staticmethod
    def get_BLEU(gen_commits,real_commits):
        assert len(gen_commits) == len(real_commits)
        BLEU_sum = 0
        for i in range(len(gen_commits)):
            BLEU_sum+=sentence_bleu([real_commits[i]],gen_commits[i],weights = (1,0,0,0))
        return BLEU_sum/len(gen_commits)