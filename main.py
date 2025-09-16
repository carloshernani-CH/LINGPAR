# main.py
import sys
import re
from typing import Dict, Optional

# =========================
# ====== Preprocessing =====
# =========================

class PrePro:
    @staticmethod
    def filter(code: str) -> str:
        # Remove tudo após '//' até o fim da linha, preservando '\n'
        return re.sub(r"//.*$", "", code, flags=re.MULTILINE)


# =========================
# ========= Lexer =========
# =========================

class Token:
    def __init__(self, kind: str, value=None):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"Token({self.kind!r}, {self.value!r})"


class Lexer:
    # Palavras reservadas (case-insensitive). PRINTLN será tratado como PRINT.
    RESERVED = {"PRINT", "PRINTLN"}

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.next: Optional[Token] = None

    def _peek(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]

    def _advance(self) -> Optional[str]:
        ch = self._peek()
        if ch is not None:
            self.position += 1
        return ch

    def _consume_while(self, predicate) -> str:
        start = self.position
        while (c := self._peek()) is not None and predicate(c):
            self._advance()
        return self.source[start:self.position]

    def select_next(self) -> Token:
        # Ignora espaços e tabs; não ignora '\n'
        while True:
            c = self._peek()
            if c is None:
                self.next = Token('EOF', '')
                return self.next
            if c in (' ', '\t', '\r'):
                self._advance()
                continue
            break

        c = self._peek()
        if c is None:
            self.next = Token('EOF', '')
            return self.next

        if c == '\n':
            self._advance()
            self.next = Token('END', '\n')
            return self.next

        if c.isdigit():
            num_str = self._consume_while(lambda ch: ch.isdigit())
            self.next = Token('INT', int(num_str))
            return self.next

        if c.isalpha():
            ident_str = self._consume_while(lambda ch: ch.isalnum() or ch == '_')
            upper = ident_str.upper()
            if upper in Lexer.RESERVED:
                # Normaliza: qualquer variação vira PRINT
                self.next = Token('PRINT', 'PRINT')
            else:
                self.next = Token('IDEN', ident_str)
            return self.next

        if c == '+':
            self._advance(); self.next = Token('PLUS', '+'); return self.next
        if c == '-':
            self._advance(); self.next = Token('MINUS', '-'); return self.next
        if c == '*':
            self._advance(); self.next = Token('MULT', '*'); return self.next
        if c == '/':
            self._advance(); self.next = Token('DIV', '/'); return self.next
        if c == '(':
            self._advance(); self.next = Token('OPEN_PAR', '('); return self.next
        if c == ')':
            self._advance(); self.next = Token('CLOSE_PAR', ')'); return self.next
        if c == '=':
            self._advance(); self.next = Token('ASSIGN', '='); return self.next

        raise SyntaxError(f"[Lexer] Invalid symbol {c!r} at position {self.position}")


# =========================
# == Symbol Table / Vars ==
# =========================

class Variable:
    def __init__(self, value: int):
        self.value = value

class SymbolTable:
    def __init__(self):
        self._table: Dict[str, Variable] = {}

    @property
    def table(self) -> Dict[str, Variable]:
        return self._table

    @table.setter
    def table(self, new_table: Dict[str, Variable]):
        self._table = new_table

    def get(self, name: str) -> Variable:
        if name not in self._table:
            raise NameError(f"[SymbolTable] Variable '{name}' not defined")
        return self._table[name]

    def set(self, name: str, value: int):
        if name.upper() in Lexer.RESERVED:
            raise SyntaxError(f"[SymbolTable] '{name}' is a reserved word and cannot be used as a variable name")
        self._table[name] = Variable(value)


# =========================
# ========= AST ===========
# =========================

class Node:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children or []

    def evaluate(self, st: SymbolTable):
        raise NotImplementedError()


class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return int(self.value)


class Identifier(Node):
    def evaluate(self, st: SymbolTable):
        return st.get(self.value).value


class UnOp(Node):
    def evaluate(self, st: SymbolTable):
        v = self.children[0].evaluate(st)
        if self.value == '+':
            return +v
        elif self.value == '-':
            return -v
        else:
            raise ValueError(f"[UnOp] Unknown unary operator {self.value}")


class BinOp(Node):
    def evaluate(self, st: SymbolTable):
        left = self.children[0].evaluate(st)
        right = self.children[1].evaluate(st)
        if self.value == '+':
            return left + right
        elif self.value == '-':
            return left - right
        elif self.value == '*':
            return left * right
        elif self.value == '/':
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left // right
        else:
            raise ValueError(f"[BinOp] Unknown binary operator {self.value}")


class Assignment(Node):
    def evaluate(self, st: SymbolTable):
        if not isinstance(self.children[0], Identifier):
            raise SyntaxError("[Assignment] Left-hand side must be an Identifier")
        name = self.children[0].value
        val = self.children[1].evaluate(st)
        st.set(name, val)
        return None


class Print(Node):
    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)
        print(val)
        return None


class Block(Node):
    def evaluate(self, st: SymbolTable):
        for child in self.children:
            child.evaluate(st)
        return None


class NoOp(Node):
    def evaluate(self, st: SymbolTable):
        return None


# =========================
# ========= Parser ========
# =========================

class Parser:
    lex: Lexer = None

    @staticmethod
    def parse_factor() -> Node:
        token = Parser.lex.next

        if token.kind in ('PLUS', 'MINUS'):
            op = token.value
            Parser.lex.select_next()
            node = Parser.parse_factor()
            return UnOp(op, [node])

        if token.kind == 'OPEN_PAR':
            Parser.lex.select_next()
            node = Parser.parse_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected closing parenthesis ')'")
            Parser.lex.select_next()
            return node

        if token.kind == 'INT':
            n = token.value
            Parser.lex.select_next()
            return IntVal(n)

        if token.kind == 'IDEN':
            name = token.value
            Parser.lex.select_next()
            return Identifier(name)

        raise SyntaxError(f"[Parser] Unexpected token {token.kind} in factor")

        # fim parse_factor

    @staticmethod
    def parse_term() -> Node:
        node = Parser.parse_factor()
        while Parser.lex.next.kind in ('MULT', 'DIV'):
            op = Parser.lex.next.value
            Parser.lex.select_next()
            rhs = Parser.parse_factor()
            node = BinOp(op, [node, rhs])
        return node

    @staticmethod
    def parse_expression() -> Node:
        node = Parser.parse_term()
        while Parser.lex.next.kind in ('PLUS', 'MINUS'):
            op = Parser.lex.next.value
            Parser.lex.select_next()
            rhs = Parser.parse_term()
            node = BinOp(op, [node, rhs])
        return node

    @staticmethod
    def parse_statement() -> Optional[Node]:
        token = Parser.lex.next

        if token.kind == 'END':
            Parser.lex.select_next()
            return NoOp()

        if token.kind == 'PRINT':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Expected '(' after PRINT")
            Parser.lex.select_next()
            expr = Parser.parse_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected ')' after PRINT expression")
            Parser.lex.select_next()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Print(children=[expr])

        if token.kind == 'IDEN':
            name = token.value
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'ASSIGN':
                raise SyntaxError("[Parser] Expected '=' after identifier")
            Parser.lex.select_next()
            expr = Parser.parse_expression()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Assignment(children=[Identifier(name), expr])

        if token.kind == 'EOF':
            return None

        raise SyntaxError(f"[Parser] Unexpected token {token.kind} at start of statement")

    @staticmethod
    def parse_program() -> Block:
        statements = []
        while Parser.lex.next.kind == 'END':
            Parser.lex.select_next()
        while Parser.lex.next.kind != 'EOF':
            stmt = Parser.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            while Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
        return Block(children=statements)

    @staticmethod
    def run(code: str) -> Block:
        Parser.lex = Lexer(code)
        Parser.lex.select_next()
        program = Parser.parse_program()
        if Parser.lex.next.kind != 'EOF':
            raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind} after program")
        return program


# =========================
# ========= main ==========
# =========================

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 main.py <arquivo-fonte>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename, "r", encoding="utf-8") as f:
            raw_code = f.read()
    except FileNotFoundError:
        print(f"Erro: arquivo '{filename}' não encontrado.")
        sys.exit(1)

    code = PrePro.filter(raw_code)
    ast_root = Parser.run(code)
    st = SymbolTable()
    ast_root.evaluate(st)


if __name__ == "__main__":
    main()
