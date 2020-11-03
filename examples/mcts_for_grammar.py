from hybrid_nlg import GrammarNode, MCTS, GPT2Score

toy_grammar = """
S -> N V O 
N -> "Gael"
V -> "knows" | "knowing"
O -> "Bas" 
"""

grammar_root = GrammarNode.rootnode_from_grammar(grammar_as_str=toy_grammar, start_symbol="S")
gpt2_scorer = GPT2Score(normalization_strategy="MeanLP")
mcts = MCTS(lm_scorer=gpt2_scorer, allocation_strategy="ALL_FROM_ROOT", buffer_size=2)
mcts.search(grammar_root, nb_of_tree_walks=4)

