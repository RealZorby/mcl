##################################################################
#--- IMPORTS                                                  ---#
##################################################################

import colorama, string
colorama.init()

##################################################################
#--- CONSTANTS                                                ---#
##################################################################

DIGITS = "0123456789"
VOWELS = "aeiouAEIOU"
LETTERS = string.ascii_letters
CHARS_LIMITED = LETTERS + "_"
CHARS = CHARS_LIMITED + DIGITS

def pointers(start_pos, end_pos):
    out = "".join([" " for _ in range(start_pos)])
    out += "".join(["^" for _ in range(end_pos - start_pos)])

    return out

##################################################################
#--- ERRORS                                                   ---#
##################################################################

class Error:

    def __init__(self, start_pos, end_pos, err_name, repr, details, event):
        self.err_name = err_name
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.details = details
        self.event = event
        self.repr = repr
    
    def __str__(self):
        line = self.start_pos.file_txt.split("\n")[self.start_pos.line]
        return f"A{'n' if self.err_name[0] in VOWELS else ''} {colorama.Fore.RED}{self.err_name}{colorama.Fore.RESET} occured while {self.event}.\nLine {colorama.Fore.YELLOW}{self.start_pos.line}{colorama.Fore.RESET} in {colorama.Fore.YELLOW}{self.start_pos.file_name}{colorama.Fore.RESET}\n{line}\n{colorama.Fore.RED}{pointers(self.start_pos.column, self.end_pos.column)}{colorama.Fore.RESET}\n{self.repr}: {self.details}"

class IllegalCharError(Error):

    def __init__(self, start_pos, end_pos, details, event):
        super().__init__(start_pos, end_pos, "IllegalCharError", "Illegal Character", details, event)

class InvalidSyntaxError(Error):

    def __init__(self, start_pos, end_pos, details, event):
        super().__init__(start_pos, end_pos, "InvalidSyntaxError", "Syntax Error", details, event)

##################################################################
#--- POSITION                                                 ---#
##################################################################

class Position:

    def __init__(self, index, line, column, file_name, file_txt):
        self.index = index
        self.line = line
        self.column = column
        self.file_name = file_name
        self.file_txt = file_txt
    
    def advance(self, cur_char=None):
        self.index += 1
        self.column += 1

        if cur_char == "\n":
            self.column = 0
            self.line += 1
        
        return self

    def copy(self):
        return self.__class__(self.index, self.line, self.column, self.file_name, self.file_txt)

##################################################################
#--- LEXER                                                    ---#
##################################################################

TT_INT       = "int"
TT_FLOAT     = "float"
TT_PLUS      = "plus"
TT_MINUS     = "minus"
TT_MUL       = "mul"
TT_DIV       = "div"
TT_POW       = "pow"
TT_IDENF     = "idenf"
TT_KEYW      = "keyw"
TT_EQ        = "eq"
TT_EE        = "ee"
TT_NE        = "ne"
TT_LT        = "lt"
TT_GT        = "gt"
TT_LTE       = "lte"
TT_GTE       = "gte"
TT_LPAREN    = "lparen"
TT_RPAREN    = "rparen"
TT_LCPAREN   = "lcparen"
TT_RCPAREN   = "rcparen"
TT_EOF       = "EOF"

KEYWORDS = {
    "int",
    "and",
    "or",
    "not",
    "if",
    "else"
}

COMPARATORS = [
    TT_EE,
    TT_NE,
    TT_LT,
    TT_GT,
    TT_LTE,
    TT_GTE
]

class Token:

    def __init__(self, type_, value=None, start_pos=Position(-1, 0, -1, "<unknown>", "<empty>"), end_pos=Position(-1, 0, -1, "<unknown>", "<empty>")):
        self.type = type_
        self.value = value
        self.start_pos = start_pos.copy()
        self.end_pos = end_pos.copy()
    
    def matches(self, type_, value):
        return type_ == self.type and value == self.value

    def __repr__(self):
        if self.value != None: return f"<Token {self.type}:{self.value}>"
        return f"<Token {self.type}>"

class Lexer:

    def __init__(self, fn, text):
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.cur_char = None
        self.advance()
    
    def advance(self):
        self.pos.advance(self.cur_char)
        self.cur_char = self.text[self.pos.index] if self.pos.index < len(self.text) else None
    
    def make_tokens(self):
        tokens = []

        while self.cur_char != None:
            if self.cur_char in " \t":
                self.advance()
            elif self.cur_char in DIGITS:
                tokens.append(self.make_number())
            elif self.cur_char in CHARS_LIMITED:
                tokens.append(self.make_identifier())
            elif self.cur_char == ".":
                tokens.append(self.make_number(num_str="0"))
                self.advance()
            elif self.cur_char == "+":
                tokens.append(Token(TT_PLUS, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "-":
                tokens.append(Token(TT_MINUS, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "*":
                tokens.append(Token(TT_MUL, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "/":
                tokens.append(Token(TT_DIV, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "^":
                tokens.append(Token(TT_POW, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "(":
                tokens.append(Token(TT_LPAREN, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == ")":
                tokens.append(Token(TT_RPAREN, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "{":
                tokens.append(Token(TT_LCPAREN, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "}":
                tokens.append(Token(TT_RCPAREN, None, self.pos, self.pos))
                self.advance()
            elif self.cur_char == "!":
                token, error = self.make_not_eq()
                if error: return [], error
                tokens.append(token)
            elif self.cur_char == "=":
                tokens.append(self.make_eq())
            elif self.cur_char == "<":
                tokens.append(self.make_lt())
            elif self.cur_char == ">":
                tokens.append(self.make_gt())
            else:
                char = self.cur_char
                pos = self.pos.copy()
                self.advance()
                return [], IllegalCharError(pos, self.pos, f"'{char}'", "lexing")

        return tokens + [Token(TT_EOF, None, self.pos, self.pos)], None
    
    def make_number(self, num_str="", dots=0):
        pos_start = self.pos.copy()

        while self.cur_char and self.cur_char in DIGITS + ".":
            if self.cur_char == ".":
                if dots: break
                dots += 1
            num_str += self.cur_char
            self.advance()
        
        if dots:
            return Token(TT_FLOAT, num_str, pos_start, self.pos)
        return Token(TT_INT, num_str, pos_start, self.pos)
    
    def make_identifier(self):
        string = ""
        start_pos = self.pos.copy()

        while self.cur_char != None and self.cur_char in CHARS:
            string += self.cur_char
            self.advance()
        
        ttype = TT_KEYW if string in KEYWORDS else TT_IDENF
        return Token(ttype, string, start_pos, self.pos)
    
    def make_not_eq(self):
        start_pos = self.pos.copy()
        self.advance()

        if self.cur_char == "=":
            self.advance()
            return Token(TT_NE, start_pos=start_pos, end_pos=self.pos), None
        
        self.advance()
        return None, InvalidSyntaxError(start_pos, self.pos, "Expected '='", "lexing")
    
    def make_eq(self):
        token_type = TT_EQ
        start_pos = self.pos.copy()
        self.advance()

        if self.cur_char == "=":
            self.advance()
            token_type = TT_EE
        
        return Token(token_type, start_pos=start_pos, end_pos=self.pos)
    
    def make_lt(self):
        token_type = TT_LT
        start_pos = self.pos.copy()
        self.advance()

        if self.cur_char == "=":
            self.advance()
            token_type = TT_LTE
        
        return Token(token_type, start_pos=start_pos, end_pos=self.pos)
    
    def make_gt(self):
        token_type = TT_GT
        start_pos = self.pos.copy()
        self.advance()

        if self.cur_char == "=":
            self.advance()
            token_type = TT_GTE
        
        return Token(token_type, start_pos=start_pos, end_pos=self.pos)

##################################################################
#--- NODES                                                    ---#
##################################################################

class VarAccessNode:

    def __init__(self, var_name):
        self.name = var_name

        self.start_pos = self.name.start_pos
        self.end_pos = self.name.end_pos

class VarDeclerationNode:

    def __init__(self, var_name, value):
        self.name = var_name
        self.value = value

        self.start_pos = self.name.start_pos
        self.end_pos = self.name.end_pos

class NumberNode:

    def __init__(self, token):
        self.token = token
    
        self.start_pos = self.token.start_pos
        self.end_pos = self.token.end_pos

    def __repr__(self):
        return f"<NumberNode {self.token}>"

class BinOpNode:

    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    
        self.start_pos = self.left.start_pos
        self.end_pos = self.right.end_pos

    def __repr__(self):
        return f"<BinOpNode {self.left} {self.op} {self.right}>"

class UnOpNode:

    def __init__(self, op, node):
        self.op = op
        self.node = node
    
        self.start_pos = self.node.start_pos
        self.end_pos = self.node.end_pos

    def __repr__(self):
        return f"<UnOpNode {self.op} {self.node}>"

class IfNode:

    def __init__(self, condition, body, else_body):
        self.condition = condition
        self.body = body
        self.else_body = else_body

        self.start_pos = self.condition.start_pos
        self.end_pos = (self.else_body or self.body).end_pos

##################################################################
#--- PARSE RESULT                                             ---#
##################################################################

class ParseResult:

    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0
    
    def register_advancement(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def succeed(self, node):
        self.node = node
        return self

    def fail(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self

##################################################################
#--- PARSER                                                   ---#
##################################################################

class Parser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.index = -1
        self.advance()
    
    ################

    def parse(self):
        res = self.expr()
        if not res.error and self.cur_tok.type != TT_EOF:
            return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Invalid syntax", "parsing"))
        return res

    ################

    def advance(self):
        self.index += 1

        if self.index < len(self.tokens):
            self.cur_tok = self.tokens[self.index]
        
        return self.cur_tok
    
    def factor(self):
        res = ParseResult()
        token = self.cur_tok

        if token.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.succeed(UnOpNode(token, factor))
        
        return self.power()
    
    def power(self):
        return self.bin_op(self.atom, (TT_POW,), self.factor)

    def if_expr(self):
        res = ParseResult()
        else_expr = None

        if not self.cur_tok.matches(TT_KEYW, "if"): return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Expected 'if'", "parsing"))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.cur_tok.type == TT_LCPAREN:
            return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Expected '{'", "parsing"))
        
        res.register_advancement()
        self.advance()

        expr = res.register(self.expr())
        if res.error: return res

        if not self.cur_tok.type == TT_RCPAREN:
            return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Unclosed Statement Body", "parsing"))
        
        if self.cur_tok.matches(TT_KEYW, "else"):
            res.register_advancement()
            self.advance()

            else_expr = res.register(self.expr)
            if res.error: return res
        
        return res.succeed(IfNode(condition, expr, else_expr))

    def atom(self):
        res = ParseResult()
        token = self.cur_tok

        if token.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.succeed(NumberNode(token))

        elif token.type == TT_IDENF:
            res.register_advancement()
            self.advance()
            return res.succeed(VarAccessNode(token))

        elif token.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.cur_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.succeed(expr)
            else: return res.fail(InvalidSyntaxError(token.start_pos, token.end_pos, "Missing ')'", "parsing"))
        
        elif token.matches(TT_KEYW, "if"):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.succeed(if_expr)
        
        return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Invalid Syntax", "parsing"))

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))
    
    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def comp_expr(self):
        res = ParseResult()

        if self.cur_tok.matches(TT_KEYW, "not"):
            op = self.cur_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.succeed(UnOpNode(op, node))
        
        node = res.register(self.bin_op(self.arith_expr, COMPARATORS))

        if res.error:
            return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Invalid Syntax", "parsing"))
        
        return res.succeed(node)

    def expr(self):
        res = ParseResult()
        
        if self.cur_tok.matches(TT_KEYW, "int"):
            res.register_advancement()
            self.advance()

            if self.cur_tok.type != TT_IDENF:
                return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Expected variable name", "parsing"))
            
            name = self.cur_tok
            res.register_advancement()
            self.advance()

            if self.cur_tok.type != TT_EQ:
                return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Expected '='", "parsing"))
            
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.succeed(VarDeclerationNode(name, expr))

        node = res.register(self.bin_op(self.comp_expr, ((TT_KEYW, "and"), (TT_KEYW, "or"))))

        if res.error: return res.fail(InvalidSyntaxError(self.cur_tok.start_pos, self.cur_tok.end_pos, "Invalid Syntax", "parsing"))

        return res.succeed(node)
    
    def bin_op(self, func, ops, func1=None):
        if func1 == None: func1 = func
        res = ParseResult()
        left = res.register(func())
        if res.error: return res

        while self.cur_tok.type in ops or (self.cur_tok.type, self.cur_tok.value) in ops:
            op = self.cur_tok
            res.register_advancement()
            self.advance()
            right = res.register(func1())
            if res.error: return res
            left = BinOpNode(left, op, right)
        
        return res.succeed(left)

##################################################################
#--- COMPILER                                                 ---#
##################################################################

class Compiler:
    
    def __init__(self):
        self.out = ""
        self.autovar_index = -1

    def compile(self, node):
        self.visit(node)
        
        for i in range(self.autovar_index + 1):
            self.destroy_var(f"mcl_autovar_{i}")
        
        return self.out

    def visit(self, node):
        func_name = f"visit_{node.__class__.__name__}"
        func = getattr(self, func_name, self.no_visit)
        return func(node)
    
    def no_visit(self, node):
        raise Exception(f"No visit method defined.\n\n=========\n DETAILS\n=========\nCurrent node: {node}")

    def visit_NumberNode(self, node: NumberNode):
        return self.create_autovar(node.token.value)
    
    def visit_BinOpNode(self, node: BinOpNode):
        op = None
        left, right = self.visit_left_right(node)
        if node.op.type == TT_KEYW:
            if node.op.value == "and":
                out = self.create_autovar(1)
                self.is_negative(left, self.assign_var, (out, 0))
                self.is_negative(right, self.assign_var, (out, 0))
                return out
            elif node.op.value == "or":
                out = self.create_autovar(0)
                self.is_positive(left, self.assign_var, (out, 1))
                self.is_positive(right, self.assign_var, (out, 1))
                return out
        elif node.op.type in COMPARATORS:
            invert = False
            if node.op.type == TT_EE: op = "="
            if node.op.type == TT_NE: op = "="; invert = True
            if node.op.type == TT_LT: op = "<"
            if node.op.type == TT_GT: op = ">"
            if node.op.type == TT_LTE: op = "<="
            if node.op.type == TT_GTE: op = ">="
            if op == None: raise Exception("Missing operation.")
            out = self.create_autovar()
            return self.compare(left, op, right, out)
        else:
            if node.op.type == TT_PLUS: op = "+="
            if node.op.type == TT_MINUS: op = "-="
            if node.op.type == TT_MUL: op = "*="
            if node.op.type == TT_DIV: op = "/="
            if op == None: raise Exception("Missing operation.")
            out = self.create_autovar()
            self.apply_operation(out, "+", left)
            return self.apply_operation(out, op, right)

    def visit_left_right(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return left,right

    def visit_UnOpNode(self, node: UnOpNode):
        var = self.visit(node.node)
        out = self.create_autovar(0)
        if node.op.matches(TT_KEYW, "not"):
            self.is_negative(var, self.assign_var, (out, 1))
        second = self.create_autovar(-1)
        self.apply_operation(out, "=", var)
        self.apply_operation(out, "*=", second)
        return out

    def visit_VarAccessNode(self, node: VarAccessNode):
        return node.name.value

    def visit_VarDeclerationNode(self, node: VarDeclerationNode):
        var = self.create_var(node.name.value)
        value = self.visit(node.value)
        self.apply_operation(var, "=", value)
        return var

    def visit_IfNode(self, node: IfNode):
        condition = self.visit(node.condition)
        


    ##############################################################
    
    def create_var(self, name: str, value = None):
        self.out += f"scoreboard objectives add {name} dummy\n"
        if value != None:
            self.assign_var(name, value)
        return name

    def create_autovar(self, value = None):
        self.autovar_index += 1
        num = self.autovar_index
        name = "mcl_autovar_" + str(num)
        self.create_var(name, value)
        return name
    
    def assign_var(self, var_name: str, value):
        self.out += f"scoreboard players set @e[tag=mcl_storage] {var_name} {value}\n"

    def destroy_var(self, var: str):
        self.out += f"scoreboard objectives remove {var}\n"

    def apply_operation(self, first: str, op: str, second: str):
        self.out += f"scoreboard players operation @e[tag=mcl_storage] {first} {op} @e[tag=mcl_storage] {second}\n"
        return first
    
    def compare(self, first: str, op: str, second: str, out: str):
        self.out += f"execute store success score @e[tag=mcl_storage] {out} run execute if score @e[tag=mcl_storage] {first} {op} @e[tag=mcl_storage] {second}\n"
        return out

    def is_positive(self, var: str, formula, args: tuple = None, kwargs: dict = None):
        second = self.create_autovar(0)
        if isinstance(formula, str):
            self.out += f"execute if score @e[tag=mcl_storage] {var} > @e[tag=mcl_storage] {second} run {formula}\n"
        else:
            self.out += f"execute if score @e[tag=mcl_storage] {var} > @e[tag=mcl_storage] {second} run "
            formula(*args, **kwargs)
    
    def is_negative(self, var: str, formula, args: tuple = (), kwargs: dict = {}):
        second = self.create_autovar(0)
        if isinstance(formula, str):
            self.out += f"execute if score @e[tag=mcl_storage] {var} <= @e[tag=mcl_storage] {second} run {formula}\n"
        else:
            self.out += f"execute if score @e[tag=mcl_storage] {var} <= @e[tag=mcl_storage] {second} run "
            formula(*args, **kwargs)

##################################################################
#--- COMPILATION                                              ---#
##################################################################

def compile(fn: str, text: str):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    compiler = Compiler()
    out = compiler.compile(ast.node)

    return out, None