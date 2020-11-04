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
best_sentence, best_score, mcts_root = mcts.search(grammar_root, nb_of_tree_walks=4)

def print_counter_node_info(counter_node):
    print("Node : %s" % str(counter_node))
    print("Best sentence from this node : %s" % str(counter_node.top_leaf_node))
    print("Nb of visits : %d" % counter_node.count)
    print("---")

print_counter_node_info(mcts_root)
for child in mcts_root.children():
    print_counter_node_info(child)