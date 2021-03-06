"""
grammar_tree module implements tree structure for grammar.
By simply overriding the compute_children method, it can be used to interface 
various kind of grammar implementation. 
"""

from copy import deepcopy
from typing import *
import random
import numpy as np

from .grammar import parse_grammar, PStruct, is_symbol_terminal, revert, copy_features


class GrammarNode:
    """
    Let N be a GrammarNode object that represents the sequence of symbols : 
    t1 ... tk n s1...sm where n represents the leftest non-terminal symbol

    Let M be another grammar node. M is a child of N if M represents the sequence of symbols 
    t1 ... tk d1 ... dj s1...sm and that there exists, in the grammar, a production rule
    n -> d1 ... dj 

    Remarks : 
    1- if a node has only one single child, we will - by default - skip this child and 
    directly return the grandchildren. This makes the MCTS run more efficiently. 

    2- if a node represents a sequence that has at least one non-terminal symbol, but that 
    this node has not any child (= dead-end branch), we artificially construct a special dead-end node
    and make it a child of this node; 
    """

    def __init__(self, symbols: tuple, grammar: Dict):
        self.symbols = symbols  # a symbol <- {"str": str, "features": PStruct | PVar | PConst}
        self.grammar = grammar
        self._children = None
        self._is_terminal = None 

    def dead_end_node(self):
        return GrammarNode(({"str": "'DEAD_END'", "features": PStruct({})},), self.grammar)

    def is_dead_end(self):
        return self.symbols[0]["str"] == "'DEAD_END'"

    @classmethod
    def rootnode_from_grammar(cls, grammar_as_str: str, start_symbol: str):
        grammar = parse_grammar(grammar_as_str)
        return cls(({"str": start_symbol, "features": PStruct({})},), grammar)

    def children(self) -> List["GrammarNode"]:
        if not self._children:
            self._children = self.compute_children()
            if (len(self._children) == 1) and (not self._children[0].is_terminal()):
                self._children = self._children[0].children()
        return self._children

    def compute_children(self) -> List["GrammarNode"]:
        if self.is_terminal() :
            return []

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
            return [self.dead_end_node()]

        for body in bodies:
            # body <- {'head_feature':PStruct, 'body_features': List[PStruct], 'body_symbols': List[str]}
            feature_copies = copy_features(body)  # <- List[{'str': str, 'features': PStruct}]

            head_bindings = []
            if not symbol["features"].unify(feature_copies[0]["features"], head_bindings):
                revert(head_bindings)
                continue

            new_node = GrammarNode(
                symbols=deepcopy(
                    self.symbols[:idx_left_nt_symb] + tuple(feature_copies[1:]) + self.symbols[idx_left_nt_symb + 1 :]
                ),
                grammar=self.grammar,
            )
            child_nodes.append(new_node)
            revert(head_bindings)

        return child_nodes if len(child_nodes) > 0 else [self.dead_end_node()]

    def is_terminal(self) -> bool:
        # in the sense the node represents a sequence composed of only terminal symbols
        # not in the sense this node does not have any child 
        if self._is_terminal is None:   
            self._is_terminal = True
            for symbol in self.symbols:
                if not is_symbol_terminal(symbol):
                    self._is_terminal = False
                    break
        return self._is_terminal

    def random_child(self) -> Union["GrammarNode", None]:
        _children = self.children()
        return random.choice(_children) if len(_children) >= 0 else None

    def __str__(self) -> str:
        if self.is_terminal():
            as_str = ""
            for symbol in self.symbols:
                if not len(symbol["str"]) == 2:  # to remove the empty string : "" or ''
                    str_symbol = symbol["str"][1:-1]  # to remove " " from "word"
                    if as_str != "":
                        as_str += " " + str_symbol
                    else:
                        as_str += str_symbol
            return as_str
        else:  # for debug / illustration only
            return " ".join(map(str, self.symbols))

    def __repr__(self):
        return self.__str__()

    def estimate_mean_depth(self, nb_samples: int = 1) -> float:
        depths = []
        for _ in range(nb_samples):
            depth = 1  # by default root's depth = 1 (and not 0)
            node = self
            while node and not node.is_terminal():
                node = node.random_child()
                depth += 1
            depths.append(depth)
        return np.mean(depths)
