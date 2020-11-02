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
ROOT -> np vp
np -> 'gael'
vp -> v o 
v -> 'knows'
o -> 'bas' | 'judith'
"""

grammar_root = GrammarNode.from_string(toy_cfg_grammar)
print(toy_cfg_grammar)
print_all_leaves(grammar_root)
print("-"*50)

toy_feature_grammar = """
ROOT -> NP[pos=subj] V NP[pos=obj] 
NP[pos=Pos] -> Pronoun[pos=Pos] | Proper[pos=Pos] 
V -> "knows"
Pronoun[pos=subj] -> "he"
Pronoun[pos=obj] -> "him"
Proper[pos=subj] -> "Gael"
Proper[pos=obj] -> "Bas"
"""

print(toy_feature_grammar)
grammar_root = GrammarNode.from_string(toy_feature_grammar)
print_all_leaves(grammar_root)
print("-"*50)
