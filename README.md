## Setup

```
git clone https://github.com/dldk-gael/hybrid-nlg
cd hybrid-nlg
pip install .
```

## Package description 

This package implements a hybrid framework for natural language generation : formal grammars define the set of possible sentence, sentence scoring functions - built on top of state-of-the art language models -  evaluate sentences' naturaless, and finally, Monte Carlo Tree Search algorithm guides the generation process toward the production of the most natural sentence given some previous context. 

The package is divised in four modules : 
- **grammar** : implements basic functions and structures to handle feature-based grammars 
- **grammar_tree** : enables the representation of grammar under the form of a tree structure
- **lm_scorer** : uses Huggingface library to build sentence scoring functions on top of two state-of-the-art language models (GPT2 and BERT)
- **mcts** : implements a version of the MCTS algorithm that is optimized for grammar search