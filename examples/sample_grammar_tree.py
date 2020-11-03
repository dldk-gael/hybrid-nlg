from hybrid_nlg import GrammarNode

def print_all_leaves(root):
    queue = [root]
    while len(queue) > 0:
        head, new_queue = queue[0], queue[1:]
        if head.is_terminal():
            print(head)
        else:
            new_queue += head.children()
        queue  = new_queue

toy_cfg_grammar = """
s -> np vp
np -> 'gael'
vp -> v o 
v -> 'knows'
o -> 'bas' | 'judith'
"""

grammar_root = GrammarNode.rootnode_from_grammar(toy_cfg_grammar, start_symbol="s")
print(toy_cfg_grammar)
print_all_leaves(grammar_root)
print("-"*50)

toy_feature_grammar = """
S -> NP[pos=subj] V NP[pos=obj] 
NP[pos=Pos] -> Pronoun[pos=Pos] | Proper[pos=Pos] 
V -> "knows"
Pronoun[pos=subj] -> "he"
Pronoun[pos=obj] -> "him"
Proper[pos=subj] -> "Gael"
Proper[pos=obj] -> "Bas"
"""

print(toy_feature_grammar)
grammar_root = GrammarNode.rootnode_from_grammar(toy_feature_grammar, start_symbol="S")
print_all_leaves(grammar_root)
print("-"*50)


feature_grammar_with_dead_end_branches = """
S[sem=SEM] -> NP[sem=SEM,pos=subj] V NP[sem=SEM,pos=obj] 
NP[sem=s1,pos=subj] -> "Gael"
NP[sem=s1,pos=obj] -> "Bas"
NP[sem=s2,pos=subj] -> "Bas"
V -> "knows"
"""

print(feature_grammar_with_dead_end_branches)
grammar_root = GrammarNode.rootnode_from_grammar(feature_grammar_with_dead_end_branches, start_symbol="S")
print_all_leaves(grammar_root)
print("-"*50)