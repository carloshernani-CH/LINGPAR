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
        """
        Remove comentários inline //... preservando quebras de linha.
        """
        # Remove tudo após '//' até o fim da linha, mas mantém '\n'
        return re.sub(r"//.*$", "", code, flags=re.MULTILINE)


# =========================
# ========= Lexer =========
# =========================

class Token:
    def __init__(self, kind: str, value=None):
        self.kind = kind   # e.g. 'INT', 'PLUS', 'MINUS', 'EOF', 'MULT', 'DIV', 'OPEN_PAR', 'CLOSE_PAR', 'ASSIGN', 'END', 'IDEN', 'PRINT'
        self.value = value # int | str | None

    def __repr__(self):
        return f"Token({self.kind!r}, {self.value!r})"


class Lexer:
    RESERVED = {"PRINT"}

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.next: Optional[Token] = None  # Token atual

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
        # Ignora espaços e tabs, mas NÃO ignora '\n' (pois é END)
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

        # Nova linha -> fim de instrução
        if c == '\n':
            self._advance()
            self.next = Token('END', '\n')
            return self.next

        # Inteiros
        if c.isdigit():
            num_str = self._consume_while(lambda ch: ch.isdigit())
            self.next = Token('INT', int(num_str))
            return self.next

        # Identificadores (devem começar com letra). Depois letras/dígitos/'_'
        if c.isalpha():
            ident_str = self._consume_while(lambda ch: ch.isalnum() or ch == '_')
            # Palavras reservadas
            if ident_str in Lexer.RESERVED:
                self.next = Token(ident_str, ident_str)
            else:
                self.next = Token('IDEN', ident_str)
            return self.next

        # Operadores e parênteses
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

        # Símbolo inválido
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
        # Impede sobrescrever palavra reservada como variável
        if name in Lexer.RESERVED:
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
    """
    Folha que representa o NOME da variável (em self.value).
    """
    def evaluate(self, st: SymbolTable):
        return st.get(self.value).value


class UnOp(Node):
    """
    value: operador '+' ou '-'
    children[0]: expressão
    """
    def evaluate(self, st: SymbolTable):
        v = self.children[0].evaluate(st)
        if self.value == '+':
            return +v
        elif self.value == '-':
            return -v
        else:
            raise ValueError(f"[UnOp] Unknown unary operator {self.value}")


class BinOp(Node):
    """
    value: '+', '-', '*', '/'
    children[0], children[1]: expressões
    """
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
    """
    children[0]: Identifier (NOME) - NÃO executar evaluate do filho 0
    children[1]: expressão cujo valor será atribuído
    """
    def evaluate(self, st: SymbolTable):
        if not isinstance(self.children[0], Identifier):
            raise SyntaxError("[Assignment] Left-hand side must be an Identifier")
        name = self.children[0].value
        val = self.children[1].evaluate(st)
        st.set(name, val)
        # Atribuição não retorna valor
        return None


class Print(Node):
    """
    children[0]: expressão a imprimir
    """
    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)
        print(val)
        return None


class Block(Node):
    """
    children: lista de statements
    """
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

    # --- expressão aritmética ---
    @staticmethod
    def parse_factor() -> Node:
        token = Parser.lex.next

        if token.kind in ('PLUS', 'MINUS'):
            # Unário
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

    # --- statements / program ---
    @staticmethod
    def parse_statement() -> Optional[Node]:
        token = Parser.lex.next

        # Linha vazia
        if token.kind == 'END':
            # Consome END e retorna NoOp
            Parser.lex.select_next()
            return NoOp()

        # PRINT (expr)
        if token.kind == 'PRINT':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Expected '(' after PRINT")
            Parser.lex.select_next()
            expr = Parser.parse_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected ')' after PRINT expression")
            Parser.lex.select_next()
            # Opcionalmente consumir END (quebra de linha) se existir
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Print(children=[expr])

        # Atribuição: IDEN '=' expression
        if token.kind == 'IDEN':
            name = token.value
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'ASSIGN':
                raise SyntaxError("[Parser] Expected '=' after identifier")
            Parser.lex.select_next()
            expr = Parser.parse_expression()
            # Opcionalmente consumir END (quebra de linha) se existir
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Assignment(children=[Identifier(name), expr])

        # EOF: não há mais statements
        if token.kind == 'EOF':
            return None

        raise SyntaxError(f"[Parser] Unexpected token {token.kind} at start of statement")

    @staticmethod
    def parse_program() -> Block:
        statements = []
        # Permite múltiplas quebras de linha iniciais
        while Parser.lex.next.kind == 'END':
            Parser.lex.select_next()

        while Parser.lex.next.kind != 'EOF':
            stmt = Parser.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            # Permite múltiplas linhas vazias entre statements
            while Parser.lex.next.kind == 'END':
                Parser.lex.select_next()

        return Block(children=statements)

    @staticmethod
    def run(code: str) -> Block:
        Parser.lex = Lexer(code)
        Parser.lex.select_next()          # posiciona no primeiro token
        program = Parser.parse_program()  # parseia o programa

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

    # Pré-processamento (remoção de comentários)
    code = PrePro.filter(raw_code)

    # Parsing
    ast_root = Parser.run(code)

    # Execução com tabela de símbolos
    st = SymbolTable()
    ast_root.evaluate(st)


if __name__ == "__main__":
    main()
