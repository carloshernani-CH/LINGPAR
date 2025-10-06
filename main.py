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
        # Remove comentários C++ (//...) preservando quebras de linha
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
    # Palavras reservadas (case-insensitive)
    KEYWORDS = {
        "PRINT": "PRINT", "PRINTLN": "PRINT",
        "IF": "IF", "ELSE": "ELSE", "WHILE": "WHILE",
        "READ": "READ"
    }

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
        # Consome espaços; '\n' vira separador 'END'
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

        # Números
        if c.isdigit():
            num = self._consume_while(lambda ch: ch.isdigit())
            self.next = Token('INT', int(num))
            return self.next

        # Identificadores / keywords (case-insensitive)
        if c.isalpha() or c == '_':
            ident = self._consume_while(lambda ch: ch.isalnum() or ch == '_')
            upper = ident.upper()
            if upper in Lexer.KEYWORDS:
                self.next = Token(Lexer.KEYWORDS[upper], upper)
            else:
                self.next = Token('IDEN', ident)
            return self.next

        # Operadores compostos primeiro
        if c == '&' and self.position + 1 < len(self.source) and self.source[self.position+1] == '&':
            self.position += 2
            self.next = Token('AND', '&&'); return self.next
        if c == '|' and self.position + 1 < len(self.source) and self.source[self.position+1] == '|':
            self.position += 2
            self.next = Token('OR', '||'); return self.next
        if c == '=' and self.position + 1 < len(self.source) and self.source[self.position+1] == '=':
            self.position += 2
            self.next = Token('EQ', '=='); return self.next

        # Símbolos simples
        table = {
            '+': 'PLUS', '-': 'MINUS', '*': 'MULT', '/': 'DIV',
            '(': 'OPEN_PAR', ')': 'CLOSE_PAR',
            '{': 'OPEN_BRA', '}': 'CLOSE_BRA',
            '=': 'ASSIGN',
            '<': 'LT', '>': 'GT',
            '!': 'NOT'
        }
        if c in table:
            self._advance()
            self.next = Token(table[c], c)
            return self.next

        raise SyntaxError(f"[Lexer] Símbolo inválido {c!r} em {self.position}")


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
            raise NameError(f"[SymbolTable] Variável '{name}' não definida")
        return self._table[name]

    def set(self, name: str, value: int):
        # Bloqueia keywords
        if name.upper() in Lexer.KEYWORDS:
            raise SyntaxError(f"[SymbolTable] '{name}' é palavra reservada")
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


class Read(Node):
    def evaluate(self, st: SymbolTable):
        val = int(input())
        return val


class UnOp(Node):
    def evaluate(self, st: SymbolTable):
        v = self.children[0].evaluate(st)
        if self.value == '+':
            return +v
        if self.value == '-':
            return -v
        if self.value == '!':
            return 0 if v else 1
        raise ValueError(f"[UnOp] Operador desconhecido {self.value}")


class BinOp(Node):
    def evaluate(self, st: SymbolTable):
        a = self.children[0].evaluate(st)
        b = self.children[1].evaluate(st)

        if self.value == '+':   return a + b
        if self.value == '-':   return a - b
        if self.value == '*':   return a * b
        if self.value == '/':
            if b == 0: raise ZeroDivisionError("Divisão por zero")
            return a // b

        if self.value == '==':  return 1 if a == b else 0
        if self.value == '<':   return 1 if a <  b else 0
        if self.value == '>':   return 1 if a >  b else 0
        if self.value == '&&':  return 1 if (a != 0 and b != 0) else 0
        if self.value == '||':  return 1 if (a != 0 or  b != 0) else 0

        raise ValueError(f"[BinOp] Operador desconhecido {self.value}")


class Assignment(Node):
    def evaluate(self, st: SymbolTable):
        if not isinstance(self.children[0], Identifier):
            raise SyntaxError("[Assignment] LHS deve ser Identifier")
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


class If(Node):
    # value=None; children = [cond, thenBlock, (optional) elseBlock]
    def evaluate(self, st: SymbolTable):
        cond = self.children[0].evaluate(st)
        if cond != 0:
            return self.children[1].evaluate(st)
        elif len(self.children) == 3:
            return self.children[2].evaluate(st)
        return None


class While(Node):
    # children = [cond, body]
    def evaluate(self, st: SymbolTable):
        while self.children[0].evaluate(st) != 0:
            self.children[1].evaluate(st)
        return None


class NoOp(Node):
    def evaluate(self, st: SymbolTable):
        return None


# =========================
# ========= Parser ========
# =========================

class Parser:
    lex: Lexer = None

    # ---------- Factors now accept: INT | IDEN | READ() | ( BoolExpr ) | +Factor | -Factor | !Factor ----------
    @staticmethod
    def parse_factor() -> Node:
        t = Parser.lex.next

        if t.kind in ('PLUS', 'MINUS', 'NOT'):
            op_map = {'PLUS': '+', 'MINUS': '-', 'NOT': '!'}
            op = op_map[t.kind]
            Parser.lex.select_next()
            return UnOp(op, [Parser.parse_factor()])

        if t.kind == 'OPEN_PAR':
            Parser.lex.select_next()
            node = Parser.parse_bool_expression()  # permite (expr relacional/booleana) também
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Esperado ')'")
            Parser.lex.select_next()
            return node

        if t.kind == 'READ':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Esperado '(' após READ")
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Esperado ')' em READ()")
            Parser.lex.select_next()
            return Read()

        if t.kind == 'INT':
            Parser.lex.select_next()
            return IntVal(t.value)

        if t.kind == 'IDEN':
            Parser.lex.select_next()
            return Identifier(t.value)

        raise SyntaxError(f"[Parser] Token inesperado em Factor: {t.kind}")

    # ---------- Arithmetic ----------
    @staticmethod
    def parse_term() -> Node:
        node = Parser.parse_factor()
        while Parser.lex.next.kind in ('MULT', 'DIV'):
            op = Parser.lex.next.value
            Parser.lex.select_next()
            node = BinOp(op, [node, Parser.parse_factor()])
        return node

    @staticmethod
    def parse_expression() -> Node:
        node = Parser.parse_term()
        while Parser.lex.next.kind in ('PLUS', 'MINUS'):
            op = Parser.lex.next.value
            Parser.lex.select_next()
            node = BinOp(op, [node, Parser.parse_term()])
        return node

    # ---------- Relational ----------
    @staticmethod
    def parse_rel_expression() -> Node:
        node = Parser.parse_expression()
        while Parser.lex.next.kind in ('EQ', 'LT', 'GT'):
            kind = Parser.lex.next.kind
            op = '==' if kind == 'EQ' else '<' if kind == 'LT' else '>'
            Parser.lex.select_next()
            node = BinOp(op, [node, Parser.parse_expression()])
        return node

    # ---------- Bool (AND / OR) ----------
    @staticmethod
    def parse_bool_term() -> Node:
        node = Parser.parse_rel_expression()
        while Parser.lex.next.kind == 'AND':
            Parser.lex.select_next()
            node = BinOp('&&', [node, Parser.parse_rel_expression()])
        return node

    @staticmethod
    def parse_bool_expression() -> Node:
        node = Parser.parse_bool_term()
        while Parser.lex.next.kind == 'OR':
            Parser.lex.select_next()
            node = BinOp('||', [node, Parser.parse_bool_term()])
        return node

    # ---------- Blocks ----------
    @staticmethod
    def parse_block() -> Node:
        # Espera '{' Statement* '}'
        if Parser.lex.next.kind != 'OPEN_BRA':
            # também aceitamos "um único Statement" como bloco
            stmt = Parser.parse_statement()
            return Block(children=[] if stmt is None else [stmt])

        Parser.lex.select_next()  # consume '{'
        statements = []
        while Parser.lex.next.kind == 'END':
            Parser.lex.select_next()
        while Parser.lex.next.kind not in ('CLOSE_BRA', 'EOF'):
            stmt = Parser.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            while Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
        if Parser.lex.next.kind != 'CLOSE_BRA':
            raise SyntaxError("[Parser] Esperado '}' para fechar bloco")
        Parser.lex.select_next()
        return Block(children=statements)

    # ---------- Statements ----------
    @staticmethod
    def parse_statement() -> Optional[Node]:
        t = Parser.lex.next

        # Separador vazio
        if t.kind == 'END':
            Parser.lex.select_next()
            return NoOp()

        # PRINT(expr)
        if t.kind == 'PRINT':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Esperado '(' após PRINT")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Esperado ')' após PRINT")
            Parser.lex.select_next()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Print(children=[expr])

        # if (cond) block [else block]
        if t.kind == 'IF':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Esperado '(' após IF")
            Parser.lex.select_next()
            cond = Parser.parse_bool_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Esperado ')' após condição do IF")
            Parser.lex.select_next()
            then_blk = Parser.parse_block()

            if Parser.lex.next.kind == 'ELSE':
                Parser.lex.select_next()
                else_blk = Parser.parse_block()
                return If(children=[cond, then_blk, else_blk])
            return If(children=[cond, then_blk])

        # while (cond) block
        if t.kind == 'WHILE':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Esperado '(' após WHILE")
            Parser.lex.select_next()
            cond = Parser.parse_bool_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Esperado ')' após condição do WHILE")
            Parser.lex.select_next()
            body = Parser.parse_block()
            return While(children=[cond, body])

        # Bloco explícito iniciando com '{'
        if t.kind == 'OPEN_BRA':
            return Parser.parse_block()

        # Atribuição: id = BoolExpr
        if t.kind == 'IDEN':
            name = t.value
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'ASSIGN':
                raise SyntaxError("[Parser] Esperado '=' após identificador")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Assignment(children=[Identifier(name), expr])

        if t.kind == 'EOF':
            return None

        raise SyntaxError(f"[Parser] Início inesperado de statement: {t.kind}")

    # ---------- Program ----------
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
            raise SyntaxError(f"[Parser] Token inesperado após programa: {Parser.lex.next.kind}")
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
