from copy import deepcopy
import random 
import numpy as np

from .grammar import parse_grammar, PStruct, is_symbol_terminal, revert, copy_features

class GrammarNode:
    """
    Let N be a GrammarNode object that represents the sequence of symbols : 
    t1 ... tk n s1...sm where n represents the leftest non-terminal symbol

    Let M be another grammar node. M is a child of N if M represents the sequence of symbols 
    t1 ... tk d1 ... dj s1...sm and that there exists a production rule
    n -> d1 ... dj in the grammar 
    """
    def __init__(self, symbols: tuple, grammar):
        self.symbols = symbols  # a symbol <- {"str": str, "features": PStruct | PVar | PConst}
        self.grammar = grammar
        self._children = None

    @classmethod
    def from_string(cls, grammar_as_str):
        grammar = parse_grammar(grammar_as_str)
        return cls(({"str": "ROOT", "features": PStruct({})},), grammar)

    def children(self):
        if not self._children:
            self._children = self.compute_children()
        return self._children

    def compute_children(self):
        child_nodes = []

        # Go from left to right to the first non terminal symbol
        idx_left_nt_symb = 0
        for symbol in self.symbols:
            if not is_symbol_terminal(symbol):
                break
            idx_left_nt_symb += 1
        symbol = self.symbols[idx_left_nt_symb]

        bodies = self.grammar.get(symbol["str"], [])

        if not bodies:
            # Error in grammar -> miss a non terminal symbol
            return [GrammarNode(({"str": "DEAD_END", "features": PStruct({})},), self.grammar)]

        for body in bodies:
            # body <- {'head_feature':PStruct, 'body_features': List[PStruct], 'body_symbols': List[str]}
            feature_copies = copy_features(body) # <- List[{'str': str, 'features': PStruct}]

            head_bindings = []
            if not symbol["features"].unify(feature_copies[0]["features"], head_bindings):
                revert(head_bindings)
                continue

            new_node = GrammarNode(
                symbols=deepcopy(self.symbols[:idx_left_nt_symb] + tuple(feature_copies[1:]) + self.symbols[idx_left_nt_symb + 1 :]),
                grammar=self.grammar,
            )
            child_nodes.append(new_node)
            revert(head_bindings)

        return child_nodes

    def is_terminal(self):
        for symbol in self.symbols:
            if not is_symbol_terminal(symbol):
                return False
        return True

    def __str__(self):
        if self.is_terminal():
            as_str = ""
            for symbol in self.symbols:
                if not len(symbol["str"]) == 2: # to remove "" or ''
                    str_symbol = symbol["str"][1:-1] # to remove " " from "word"
                    if as_str != "":
                        as_str += " " + str_symbol
                    else:
                        as_str += str_symbol
            return as_str
        else: # for debug / illustration only
            return " ".join(map(str, self.symbols))

    def estimate_mean_depth(self, nb_samples: int = 1):
        depths = []
        for _ in range(nb_samples):
            depth = 1  # by choice root's depth = 1 (and not 0)
            node = self
            while not node.is_terminal():
                node = random.choice(self.children())
                depth += 1
            depths.append(depth)
        return np.mean(depths)
    