# main.py
import sys
import re

# ----------------- TOKENS -----------------
class Token:
    def __init__(self, kind: str, value=None):
        self.kind = kind
        self.value = value

# ----------------- PREPROCESSOR -----------------
class PrePro:
    @staticmethod
    def filter(code: str) -> str:
        # remove //... até \n, preservando \n
        return re.sub(r"//[^\n]*", "", code)

# ----------------- SYMBOLS -----------------
class Variable:
    def __init__(self, value: int = 0):
        self.value = value

class SymbolTable:
    def __init__(self):
        self._table = {}

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, new_table):
        self._table = new_table

    def get(self, name: str) -> int:
        if name not in self._table:
            raise NameError(f"[SymbolTable] Undefined variable '{name}'")
        return self._table[name].value

    def set(self, name: str, value: int) -> None:
        self._table[name] = Variable(value)

# ----------------- AST -----------------
class Node:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children or []

    def evaluate(self, st: SymbolTable):
        raise NotImplementedError()

class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return self.value

class UnOp(Node):
    # value: '+' | '-', children: [expr]
    def evaluate(self, st: SymbolTable):
        v = self.children[0].evaluate(st)
        if self.value == '+':
            return +v
        if self.value == '-':
            return -v
        raise RuntimeError(f"[UnOp] Unknown op {self.value!r}")

class BinOp(Node):
    # value: '+', '-', '*', '/', children: [lhs, rhs]
    def evaluate(self, st: SymbolTable):
        l = self.children[0].evaluate(st)
        r = self.children[1].evaluate(st)
        if self.value == '+':
            return l + r
        if self.value == '-':
            return l - r
        if self.value == '*':
            return l * r
        if self.value == '/':
            if r == 0:
                raise ZeroDivisionError("[BinOp] Division by zero")
            return l // r
        raise RuntimeError(f"[BinOp] Unknown op {self.value!r}")

class Identifier(Node):
    # value: nome (str)
    def evaluate(self, st: SymbolTable):
        return st.get(self.value)

class Assignment(Node):
    # children: [Identifier(name), expr]
    def evaluate(self, st: SymbolTable):
        name_node = self.children[0]
        expr_node = self.children[1]
        if not isinstance(name_node, Identifier):
            raise RuntimeError("[Assignment] Left side must be an identifier")
        val = expr_node.evaluate(st)
        st.set(name_node.value, val)

class Print(Node):
    # children: [expr]
    def evaluate(self, st: SymbolTable):
        v = self.children[0].evaluate(st)
        print(v)

class Block(Node):
    # children: [stmt1, stmt2, ...]
    def evaluate(self, st: SymbolTable):
        for ch in self.children:
            ch.evaluate(st)

class NoOp(Node):
    def evaluate(self, st: SymbolTable):
        return None

# ----------------- LEXER -----------------
class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.next = None

    def _peek(self):
        if self.position >= len(self.source):
            return None
        return self.source[self.position]

    def _advance(self):
        ch = self._peek()
        self.position += 1
        return ch

    def _skip_spaces_no_newline(self):
        while True:
            c = self._peek()
            if c is not None and c in (' ', '\t', '\r'):
                self._advance()
            else:
                break

    def select_next(self) -> Token:
        self._skip_spaces_no_newline()

        c = self._peek()
        if c is None:
            self.next = Token('EOF', '')
            return self.next

        # newline -> END
        if c == '\n':
            self._advance()
            self.next = Token('END', '\n')
            return self.next

        # integers
        if c.isdigit():
            start = self.position
            while (c := self._peek()) is not None and c.isdigit():
                self._advance()
            self.next = Token('INT', int(self.source[start:self.position]))
            return self.next

        # identifiers / reserved
        if c.isalpha():
            start = self.position
            self._advance()
            while (c := self._peek()) is not None and (c.isalnum() or c == '_'):
                self._advance()
            lex = self.source[start:self.position]
            if lex == "print":
                self.next = Token('PRINT', lex)
            else:
                self.next = Token('IDEN', lex)
            return self.next

        # invalid identifiers starting with underscore, or other leading invalids
        if c == '_' or c == '@':
            raise SyntaxError(f"[Lexer] Invalid identifier start: {c!r}")

        # operators / punctuation
        if c == '+':
            self._advance()
            self.next = Token('PLUS', '+')
            return self.next
        if c == '-':
            self._advance()
            self.next = Token('MINUS', '-')
            return self.next
        if c == '*':
            self._advance()
            self.next = Token('MULT', '*')
            return self.next
        if c == '/':
            self._advance()
            self.next = Token('DIV', '/')
            return self.next
        if c == '(':
            self._advance()
            self.next = Token('OPEN_PAR', '(')
            return self.next
        if c == ')':
            self._advance()
            self.next = Token('CLOSE_PAR', ')')
            return self.next
        if c == '=':
            self._advance()
            self.next = Token('ASSIGN', '=')
            return self.next

        raise SyntaxError(f"[Lexer] Invalid symbol {c!r}")

# ----------------- PARSER -----------------
class Parser:
    lex: Lexer = None

    @staticmethod
    def parse_program() -> Node:
        block = Block(children=[])
        while Parser.lex.next.kind != 'EOF':
            if Parser.lex.next.kind == 'END':
                block.children.append(NoOp())
                Parser.lex.select_next()
                continue
            stmt = Parser.parse_statement()
            block.children.append(stmt)
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            elif Parser.lex.next.kind == 'EOF':
                break
            else:
                raise SyntaxError(f"[Parser] Expected END or EOF, found {Parser.lex.next.kind}")
        return block

    @staticmethod
    def parse_statement() -> Node:
        tok = Parser.lex.next

        # print ( expression )
        if tok.kind == 'PRINT':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Expected '(' after print")
            Parser.lex.select_next()
            expr = Parser.parse_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected ')' after print expression")
            Parser.lex.select_next()
            return Print(children=[expr])

        # assignment: IDEN = expression
        if tok.kind == 'IDEN':
            name = tok.value
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'ASSIGN':
                raise SyntaxError("[Parser] Expected '=' after identifier")
            Parser.lex.select_next()
            expr = Parser.parse_expression()
            return Assignment(children=[Identifier(name), expr])

        # empty line handled in parse_program
        raise SyntaxError(f"[Parser] Unexpected statement start: {tok.kind}")

    @staticmethod
    def parse_expression() -> Node:
        node = Parser.parse_term()
        while Parser.lex.next.kind in ('PLUS', 'MINUS'):
            op = '+' if Parser.lex.next.kind == 'PLUS' else '-'
            Parser.lex.select_next()
            rhs = Parser.parse_term()
            node = BinOp(op, [node, rhs])
        return node

    @staticmethod
    def parse_term() -> Node:
        node = Parser.parse_factor()
        while Parser.lex.next.kind in ('MULT', 'DIV'):
            op = '*' if Parser.lex.next.kind == 'MULT' else '/'
            Parser.lex.select_next()
            rhs = Parser.parse_factor()
            node = BinOp(op, [node, rhs])
        return node

    @staticmethod
    def parse_factor() -> Node:
        tok = Parser.lex.next

        if tok.kind in ('PLUS', 'MINUS'):
            op = '+' if tok.kind == 'PLUS' else '-'
            Parser.lex.select_next()
            child = Parser.parse_factor()
            return UnOp(op, [child])

        if tok.kind == 'OPEN_PAR':
            Parser.lex.select_next()
            node = Parser.parse_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected closing parenthesis")
            Parser.lex.select_next()
            return node

        if tok.kind == 'INT':
            node = IntVal(tok.value)
            Parser.lex.select_next()
            return node

        if tok.kind == 'IDEN':
            name = tok.value
            Parser.lex.select_next()
            return Identifier(name)

        raise SyntaxError(f"[Parser] Unexpected token {tok.kind}, expected INT, IDEN or '('")

    @staticmethod
    def run(code: str) -> Node:
        Parser.lex = Lexer(code)
        Parser.lex.select_next()
        root = Parser.parse_program()
        if Parser.lex.next.kind != 'EOF':
            raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind} after program")
        return root

# ----------------- MAIN -----------------
def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python3 main.py <source.go>")
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    filtered = PrePro.filter(raw)
    ast_root = Parser.run(filtered)
    st = SymbolTable()
    ast_root.evaluate(st)

if __name__ == "__main__":
    main()
