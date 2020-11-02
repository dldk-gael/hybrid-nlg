from hybrid_nlg import GPT2Score, BERTScore

##### GPT2 Score #####
print("Initialize and load in memory the gpt2-based sentence scorer")
gpt2_scorer = GPT2Score(model_name="gpt2", batch_size=2)

print("Compute sentence scores")
gpt2_scorer.set_context("Does Bas know Gael ?")
sentences = ["Bas knows Judith.", "Bas knows Gael."]
scores = gpt2_scorer.compute_score(sentences)

for sentence, score in zip(sentences, scores):
    print ("%s : %f" % (sentence, score))


##### BERT Score #####
print("Initialize and load in memory the bert-based sentence scorer")
bert_score = BERTScore(model_name="bert-base-uncased", batch_size=1)

sentences = [
    "He knows Gael.",
    "He know Gael.",
]

scores = bert_score.compute_score(sentences)

for sentence, score in zip(sentences, scores):
    print ("%s : %f" % (sentence, score))