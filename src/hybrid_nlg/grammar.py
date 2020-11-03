"""
grammar module implements basic functions and structures to handle feature-based grammars
(code from Bas)
"""

import re 

class GrammarDS:
    VARIABLE = 0
    CONSTANT = 1
    STRUCTURE = 2
    KGNODE = 3
    FUNCTION = 4
    def __init__(self, type):
        self.type = type
        
class PVar(GrammarDS):
    def __init__(self, name):
        super().__init__(GrammarDS.VARIABLE)
        self.ref = None
        self.name = name 
        
    def unify(self, other, bindings, expand_features=False):
        if self == other: 
            return True
        if self.ref == None:
            self.ref = other
            bindings.append(self)
            return True
        else:
            return self.ref.unify(other, bindings, expand_features)
            
    def show(self):
        if self.ref == None:
            return self.name
        return self.ref.show() 
        
    def __str__(self):
        refstr = 'UNBOUND'
        if self.ref != None:
            refstr = self.ref.show()
        return '(VAR ' + self.name + '=' + refstr + ')'
        
    def get_mirror(self, variables):
        if self.name not in variables:
            variables[self.name] = PVar(self.name)
        return variables[self.name]
        
    def derefcopy(self):
        if self.ref != None:
            return self.ref.derefcopy()
        return PConstant(None)
        
        
class PConstant(GrammarDS):
    def __init__(self, value):
        super().__init__(GrammarDS.CONSTANT)
        self.value = value 
        
    def unify(self, other, bindings, expand_features=False):
        if other.type == GrammarDS.VARIABLE and other.ref == None:
            other.ref = self 
            bindings.append(other)
            return True
        elif other.type == GrammarDS.VARIABLE:
            return self.unify(other.ref, bindings, expand_features)
        return other.type == GrammarDS.CONSTANT and other.value == self.value
        
    def show(self):
        return str(self.value)
        
    def __str__(self):
        return self.show()
        
    def get_mirror(self, variables):
        return self
        
    def derefcopy(self):
        return self

class PStruct(GrammarDS):
    
    def __init__(self, features):
        super().__init__(GrammarDS.STRUCTURE)
        self.features = features 
        
    def unify(self, other, bindings, expand_features=False):
        if other.type == GrammarDS.VARIABLE and other.ref == None:
            other.ref = self 
            bindings.append(other)
            return True 
        elif other.type == GrammarDS.VARIABLE:
            return self.unify(other.ref, bindings, expand_features)
        elif other.type == GrammarDS.STRUCTURE and not other.features:
            return True
        elif other.type == GrammarDS.STRUCTURE:
            to_check = [f for f in self.features if f in other.features]
            for f in to_check:
                if not self.features[f].unify(other.features[f], bindings, expand_features):
                    return False
            if expand_features:
                new_features = [f for f in other.features if f not in self.features]
                for f in new_features:
                    self.features[f] = other.features[f]
                bindings.append([self, new_features])
            return True 
        return False
    
    def show(self):
        s = '['
        first = True 
        for f in self.features:
            if first:
                first = False
            else:
                s += ', '
            s += f + '='+self.features[f].show()
        return s + ']'
        
    def __repr__(self): 
        return self.show()
        
    def get_mirror(self, variables): 
        mirror_features = {f: self.features[f].get_mirror(variables) for f in self.features} 
        return PStruct(mirror_features)
    
    def derefcopy(self):
        copy_features = {f: self.features[f].derefcopy() for f in self.features}
        return PStruct(copy_features)

class PFunction(GrammarDS):
    def __init__(self, functor, arguments, infix=False):
        super().__init__(GrammarDS.FUNCTION)
        self.functor = functor
        self.arguments = arguments
        self.infix = infix
    
    def show(self):
        if self.infix:
            return self.arguments[0].show() + ' ' + self.functor + ' ' + self.arguments[1].show()
        else:
            return self.functor + '(' + ','.join([a.show() for a in self.arguments]) + ')'
    
    def __str__(self):
        return self.show()
        
class PKGNode(GrammarDS):
    def __init__(self, functor, pstruct):
        super().__init__(GrammarDS.KGNODE)
        self.functor = functor
        self.pstruct = pstruct 
    
    def show(self):
        result = self.functor
        if self.pstruct != None:
            result += self.pstruct.show()
        else:
            result += '[]'
        return result
        
    def unify(self, other, bindings, expand_features=False):
        if other.type == GrammarDS.VARIABLE and other.ref == None:
            other.ref = self 
            bindings.append(other)
            return True 
        elif other.type == GrammarDS.VARIABLE:
            return self.unify(other.ref, bindings, expand_features)
        elif other.type == GrammarDS.KGNODE:
            if self.functor != other.functor: 
                return False
            if self.pstruct == None or other.pstruct == None:
                return True 
            return self.pstruct.unify(other.pstruct, bindings, expand_features)  
        return False
        
    def derefcopy(self):
        if self.pstruct == None:
            return PKGNode(self.functor, None)
        return PKGNode(self.functor, self.pstruct.derefcopy())
        
def revert(bindings):
    for i in range(len(bindings)):
        if isinstance(bindings[i], list):
            for feature in bindings[i][1]:
                del bindings[i][0].features[feature]
        else:
            binding = bindings[-i]
            binding.ref = None
        
def copy_features(body):
    variables = {}
    mirrors = []
    mirrors.append({'features': body['head_feature'].get_mirror(variables)})
    for i in range(len(body['body_symbols'])):
        mirrors.append({'str': body['body_symbols'][i], 'features': body['body_features'][i].get_mirror(variables)})
    return mirrors 

def is_symbol_terminal(symbol):
    return (symbol["str"].startswith('"') or symbol["str"].startswith("'"))


# PARSE A FEATURE-BASED GRAMMAR GIVEN A INPUT STR

def parse_grammar(as_str):
    lines = as_str.splitlines()
    grammar = {}
    for line in lines:
        line_split = line.split('->')
        if len(line_split) == 1 or line_split[0].startswith('#'):
            continue
        head = "".join(line_split[0].split())
        if '[' in head:
            head = head[:head.index('[')]
        grammar[head] = grammar.get(head, [])
        for body_split in line_split[1].split('|'):
            symbols, features = parse_rule("".join(line_split[0].split()) + '->'+body_split)
            grammar[head].append({
                'head_feature': features[0],
                'body_symbols': symbols[1:],
                'body_features': features[1:]
            })
    return grammar
    
def parse_rule(str):
    variables = {}
    for name in re.findall(r'[A-Z][A-Za-z0-9]*', str):
        variables[name] = PVar(name)
    str = str.strip()
    rule_split = str.split('->')
    head, head_feature = get_feature_struct(rule_split[0].strip(), variables)
    symbols = [head]
    features = [head_feature]
    for b in rule_split[1].strip().split(r' '):
        if not b:
            continue
        symbol, feature = get_feature_struct(b, variables)
        symbols.append(symbol)
        features.append(feature)
    return symbols, features
    
def parse_separate_feature(str, old_variables = {}):
    variables = {}
    for name in re.findall(r'[A-Z][A-Za-z0-9]+', str):
        if name in old_variables:
            variables[name] = old_variables[name]
        else:
            variables[name] = PVar(name)
    _, feature = get_feature_struct(str.strip(), variables)
    return feature, variables
    
def get_feature_struct(str, variables): 
    if '[' not in str:
        return str, PStruct({})
    str2 = str[str.index('[') + 1: -1]
    arguments = []
    record = ''
    recording = True
    nested = 0
    for i in str2:
        if recording and nested == 0 and i == ',':
            recording = False
            arguments.append(record)
            record = ''
        if i == '[':
            nested += 1
        if i == ']': 
            nested -= 1
        if not recording and (i != ',' and i != ' '):
            recording = True
        if recording:
            record += i
    if record:
        arguments.append(record)  
    features = {}
    for a in arguments:
        if '=' not in a:
            print(str, 'X',a,'x')
        s = a.index('=')
        name = a[:s]
        value = a[s+1:]
        if value in variables:
            features[name] = variables[value]
        elif value[0] == '[':
            _, features[name] = get_feature_struct(value, variables)
        else:
            features[name] = PConstant(value)
    return str[:str.index('[')], PStruct(features)
    