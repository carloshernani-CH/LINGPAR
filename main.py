import sys
import re
from typing import Dict, Optional

# =========================
# ====== Preprocessing ====
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
    # Palavras reservadas aceitas.
    RESERVED = {
        "PRINT", "PRINTLN", "if", "else", "while", "for", "read", "Scanln",
        "var", "true", "false", "int", "bool", "string"
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

    def _consume_string(self) -> str:
        # Já vimos '"'
        self._advance()
        start = self.position
        while True:
            c = self._peek()
            if c is None or c == '\n':
                raise SyntaxError("[Lexer] Unterminated string literal")
            if c == '"':
                s = self.source[start:self.position]
                self._advance()  # fecha aspas
                return s
            self._advance()

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

        # Terminadores de comando
        if c == '\n':
            self._advance()
            self.next = Token('END', '\n')
            return self.next
        if c == ';':
            self._advance()
            self.next = Token('END', ';')
            return self.next

        # Strings
        if c == '"':
            s = self._consume_string()
            self.next = Token('STR', s)
            return self.next

        # Números
        if c.isdigit():
            num_str = self._consume_while(lambda ch: ch.isdigit())
            self.next = Token('INT', int(num_str))
            return self.next

        # Identificadores / Palavras-chave
        if c.isalpha() or c == '_':
            ident_str = self._consume_while(lambda ch: ch.isalnum() or ch == '_')
            if ident_str in ("Println", "PRINTLN", "PRINT"):
                self.next = Token('PRINT', 'PRINT')
            elif ident_str == "if":
                self.next = Token('IF', 'if')
            elif ident_str == "else":
                self.next = Token('ELSE', 'else')
            elif ident_str == "while":
                self.next = Token('WHILE', 'while')
            elif ident_str == "for":
                # 'for' mapeado para WHILE (linguagem do semestre)
                self.next = Token('WHILE', 'for')
            elif ident_str in ("read", "Scanln"):
                self.next = Token('READ', 'read')
            elif ident_str == "var":
                self.next = Token('VAR', 'var')
            elif ident_str in ("true", "false"):
                self.next = Token('BOOL', True if ident_str == "true" else False)
            elif ident_str in ("int", "bool", "string"):
                self.next = Token('TYPE', ident_str)
            else:
                self.next = Token('IDEN', ident_str)
            return self.next

        # Operadores compostos
        if c == '&' and self.position + 1 < len(self.source) and self.source[self.position+1] == '&':
            self.position += 2
            self.next = Token('AND', '&&')
            return self.next
        if c == '|' and self.position + 1 < len(self.source) and self.source[self.position+1] == '|':
            self.position += 2
            self.next = Token('OR', '||')
            return self.next
        if c == '=' and self.position + 1 < len(self.source) and self.source[self.position+1] == '=':
            self.position += 2
            self.next = Token('EQ', '==')
            return self.next

        # Operadores/símbolos simples
        if c == '+':
            self._advance(); self.next = Token('PLUS', '+'); return self.next
        if c == '-':
            self._advance(); self.next = Token('MINUS', '-'); return self.next
        if c == '*':
            self._advance(); self.next = Token('MULT', '*'); return self.next
        if c == '/':
            self._advance(); self.next = Token('DIV', '/'); return self.next
        if c == '!':
            self._advance(); self.next = Token('NOT', '!'); return self.next
        if c == '(':
            self._advance(); self.next = Token('OPEN_PAR', '('); return self.next
        if c == ')':
            self._advance(); self.next = Token('CLOSE_PAR', ')'); return self.next
        if c == '{':
            self._advance(); self.next = Token('OPEN_BRA', '{'); return self.next
        if c == '}':
            self._advance(); self.next = Token('CLOSE_BRA', '}'); return self.next
        if c == '>':
            self._advance(); self.next = Token('GT', '>'); return self.next
        if c == '<':
            self._advance(); self.next = Token('LT', '<'); return self.next
        if c == '=':
            self._advance(); self.next = Token('ASSIGN', '='); return self.next
        if c == ':':
            self._advance(); self.next = Token('COLON', ':'); return self.next

        raise SyntaxError(f"[Lexer] Invalid symbol {c!r} at position {self.position}")


# =========================
# == Symbol Table / Vars ==
# =========================

class Variable:
    def __init__(self, value, vtype: str):
        self.value = value
        self.type = vtype  # "int" | "bool" | "string"


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

    def create_variable(self, name: str, vtype: str):
        if name.upper() in {"PRINT", "PRINTLN"} or name in {"if", "else", "while", "for", "read", "Scanln", "var"}:
            raise SyntaxError(f"[SymbolTable] '{name}' is a reserved word and cannot be used as a variable name")
        if name in self._table:
            raise NameError(f"[SymbolTable] Variable '{name}' already declared")
        if vtype not in ("int", "bool", "string"):
            raise TypeError(f"[SymbolTable] Unknown type '{vtype}'")
        self._table[name] = Variable(None, vtype)

    def set(self, name: str, value):
        if name not in self._table:
            raise NameError(f"[SymbolTable] Variable '{name}' not declared")
        var = self._table[name]
        # Checagem de tipos
        if var.type == "int":
            if not isinstance(value, int):
                raise TypeError(f"[SymbolTable] Expected int for '{name}', got {type(value).__name__}")
        elif var.type == "bool":
            if not (isinstance(value, int) and value in (0, 1)):
                raise TypeError(f"[SymbolTable] Expected bool (0/1) for '{name}', got {value!r}")
        elif var.type == "string":
            if not isinstance(value, str):
                raise TypeError(f"[SymbolTable] Expected string for '{name}', got {type(value).__name__}")
        else:
            raise TypeError(f"[SymbolTable] Unsupported variable type '{var.type}'")
        var.value = value


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


class StringVal(Node):
    def evaluate(self, st: SymbolTable):
        return str(self.value)


class BoolVal(Node):
    def evaluate(self, st: SymbolTable):
        # Representamos bool como 0/1
        return 1 if self.value else 0


class Identifier(Node):
    def evaluate(self, st: SymbolTable):
        return st.get(self.value).value


class Read(Node):
    def evaluate(self, st: SymbolTable):
        try:
            line = input()
        except EOFError:
            line = "0"
        line = line.strip()
        if not re.fullmatch(r"-?\d+", line or "0"):
            raise ValueError("[Read] Expected integer input")
        return int(line)


class UnOp(Node):
    def evaluate(self, st: SymbolTable):
        v = self.children[0].evaluate(st)
        if self.value == '+':
            if not isinstance(v, int):
                raise TypeError("[UnOp] Unary '+' expects int")
            return +v
        elif self.value == '-':
            if not isinstance(v, int):
                raise TypeError("[UnOp] Unary '-' expects int")
            return -v
        elif self.value == '!':
            if v not in (0, 1):
                raise TypeError("[UnOp] '!' expects bool")
            return 0 if v else 1
        else:
            raise ValueError(f"[UnOp] Unknown unary operator {self.value}")


class BinOp(Node):
    def evaluate(self, st: SymbolTable):
        left = self.children[0].evaluate(st)
        right = self.children[1].evaluate(st)
        op = self.value

        if op in ('+', '-', '*', '/'):
            # Aritmética: apenas int,int.
            # Exceção: '+' concatena string+string.
            if isinstance(left, str) or isinstance(right, str):
                if op == '+' and isinstance(left, str) and isinstance(right, str):
                    return left + right
                raise TypeError(f"[BinOp] '{op}' not supported for strings (only '+' for string+string)")
            if not (isinstance(left, int) and isinstance(right, int)):
                raise TypeError(f"[BinOp] Arithmetic '{op}' requires int,int")
            if op == '+':
                return left + right
            if op == '-':
                return left - right
            if op == '*':
                return left * right
            if op == '/':
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left // right

        if op in ('==', '>', '<'):
            # Comparação: tipos iguais (int com int, string com string)
            if type(left) != type(right):
                # Observação: em Python bool é subclass de int; aqui exigimos tipos idênticos.
                raise TypeError(f"[BinOp] Comparison '{op}' requires operands of the same type")
            if op == '==':
                return 1 if left == right else 0
            # Para > e <: permitimos para int e string; não para bool
            if isinstance(left, int) and left in (0, 1) and isinstance(right, int) and right in (0, 1):
                raise TypeError(f"[BinOp] '{op}' not supported for bool")
            if not (isinstance(left, int) or isinstance(left, str)):
                raise TypeError(f"[BinOp] '{op}' not supported for type {type(left).__name__}")
            if op == '>':
                return 1 if left > right else 0
            if op == '<':
                return 1 if left < right else 0

        if op in ('&&', '||'):
            # Lógico: espera 0/1
            if left not in (0, 1) or right not in (0, 1):
                raise TypeError(f"[BinOp] Logical '{op}' requires bool operands")
            if op == '&&':
                return 1 if (left == 1 and right == 1) else 0
            if op == '||':
                return 1 if (left == 1 or right == 1) else 0

        raise ValueError(f"[BinOp] Unknown operator {op}")


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


class If(Node):
    # children: [cond, then_block, else_block?]
    def evaluate(self, st: SymbolTable):
        cond = self.children[0].evaluate(st)
        if cond not in (0, 1):
            raise TypeError("[If] Condition must be bool")
        if cond != 0:
            return self.children[1].evaluate(st)
        elif len(self.children) == 3:
            return self.children[2].evaluate(st)
        return None


class While(Node):
    # children: [cond, body]
    def evaluate(self, st: SymbolTable):
        while True:
            cond = self.children[0].evaluate(st)
            if cond not in (0, 1):
                raise TypeError("[While] Condition must be bool")
            if cond == 0:
                break
            self.children[1].evaluate(st)
        return None


class VarDec(Node):
    # value: tipo ("int"/"bool"/"string")
    # children: [Identifier, (optional) expr]
    def evaluate(self, st: SymbolTable):
        vtype = self.value
        if not isinstance(self.children[0], Identifier):
            raise SyntaxError("[VarDec] First child must be Identifier")
        name = self.children[0].value
        st.create_variable(name, vtype)
        if len(self.children) == 2:
            init_val = self.children[1].evaluate(st)
            st.set(name, init_val)
        return None


class NoOp(Node):
    def evaluate(self, st: SymbolTable):
        return None


# =========================
# ========= Parser ========
# =========================

class Parser:
    lex: Lexer = None
    strict_block_after_control: bool = False

    # ---------- Factors / Terms / Expressions (precedences) ----------
    @staticmethod
    def parse_factor() -> Node:
        token = Parser.lex.next

        # Unary operators
        if token.kind in ('PLUS', 'MINUS', 'NOT'):
            op = {'PLUS': '+', 'MINUS': '-', 'NOT': '!'}[token.kind]
            Parser.lex.select_next()
            node = Parser.parse_factor()
            return UnOp(op, [node])

        if token.kind == 'OPEN_PAR':
            Parser.lex.select_next()
            node = Parser.parse_bool_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected closing parenthesis ')'")
            Parser.lex.select_next()
            return node

        if token.kind == 'INT':
            n = token.value
            Parser.lex.select_next()
            return IntVal(n)

        if token.kind == 'STR':
            s = token.value
            Parser.lex.select_next()
            return StringVal(s)

        if token.kind == 'BOOL':
            b = token.value
            Parser.lex.select_next()
            return BoolVal(b)

        if token.kind == 'IDEN':
            name = token.value
            Parser.lex.select_next()
            return Identifier(name)

        if token.kind == 'READ':
            # read() / Scanln()
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Expected '(' after read")
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected ')' after read(")
            Parser.lex.select_next()
            return Read()

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

    @staticmethod
    def parse_rel_expression() -> Node:
        node = Parser.parse_expression()
        while Parser.lex.next.kind in ('EQ', 'GT', 'LT'):
            op = {'EQ': '==', 'GT': '>', 'LT': '<'}[Parser.lex.next.kind]
            Parser.lex.select_next()
            rhs = Parser.parse_expression()
            node = BinOp(op, [node, rhs])
        return node

    @staticmethod
    def parse_bool_term() -> Node:
        node = Parser.parse_rel_expression()
        while Parser.lex.next.kind == 'AND':
            Parser.lex.select_next()
            rhs = Parser.parse_rel_expression()
            node = BinOp('&&', [node, rhs])
        return node

    @staticmethod
    def parse_bool_expression() -> Node:
        node = Parser.parse_bool_term()
        while Parser.lex.next.kind == 'OR':
            Parser.lex.select_next()
            rhs = Parser.parse_bool_term()
            node = BinOp('||', [node, rhs])
        return node

    # ---------- Statements / Blocks ----------
    @staticmethod
    def parse_block() -> Node:
        # Consome '{'
        if Parser.lex.next.kind != 'OPEN_BRA':
            raise SyntaxError("[Parser] Expected '{' to start block")
        Parser.lex.select_next()

        statements = []
        # Verifica alinhamento: se strict e '}' vier logo sem NEWLINE/; -> erro
        if Parser.strict_block_after_control and Parser.lex.next.kind == 'CLOSE_BRA':
            raise SyntaxError("[Parser] Unexpected token CLOSE_BRA")

        saw_end = False
        while Parser.lex.next.kind == 'END':
            saw_end = True
            Parser.lex.select_next()

        if Parser.strict_block_after_control and not saw_end and Parser.lex.next.kind == 'CLOSE_BRA':
            # '{ }' na mesma linha logo após if/else/while
            raise SyntaxError("[Parser] Unexpected token CLOSE_BRA")

        while Parser.lex.next.kind not in ('CLOSE_BRA', 'EOF'):
            stmt = Parser.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            while Parser.lex.next.kind == 'END':
                Parser.lex.select_next()

        if Parser.lex.next.kind != 'CLOSE_BRA':
            raise SyntaxError("[Parser] Expected '}' to end block")
        Parser.lex.select_next()
        return Block(children=statements)

    @staticmethod
    def parse_statement() -> Optional[Node]:
        token = Parser.lex.next

        if token.kind == 'END':
            Parser.lex.select_next()
            return NoOp()

        if token.kind == 'OPEN_BRA':
            return Parser.parse_block()

        # --- var declaration: var ID : TYPE [= expr] [END]
        if token.kind == 'VAR':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'IDEN':
                raise SyntaxError("[Parser] Expected identifier after 'var'")
            ident_name = Parser.lex.next.value
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'COLON':
                raise SyntaxError("[Parser] Expected ':' after identifier in var declaration")
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'TYPE':
                raise SyntaxError("[Parser] Expected a TYPE after ':'")
            vtype = Parser.lex.next.value
            Parser.lex.select_next()

            children = [Identifier(ident_name)]
            if Parser.lex.next.kind == 'ASSIGN':
                Parser.lex.select_next()
                expr = Parser.parse_bool_expression()
                children.append(expr)

            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()

            return VarDec(value=vtype, children=children)

        if token.kind == 'PRINT':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Expected '(' after PRINT")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected ')' after PRINT expression")
            Parser.lex.select_next()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Print(children=[expr])

        if token.kind == 'IF':
            Parser.lex.select_next()
            cond = None
            if Parser.lex.next.kind == 'OPEN_PAR':
                Parser.lex.select_next()
                cond = Parser.parse_bool_expression()
                if Parser.lex.next.kind != 'CLOSE_PAR':
                    raise SyntaxError("[Parser] Expected ')' after if condition")
                Parser.lex.select_next()
                # permite continuar após ')':  ) || expr
                while Parser.lex.next.kind in ('AND', 'OR'):
                    op = '&&' if Parser.lex.next.kind == 'AND' else '||'
                    Parser.lex.select_next()
                    right = Parser.parse_bool_term()
                    cond = BinOp(op, [cond, right])
            else:
                cond = Parser.parse_bool_expression()

            # proíbe quebra de linha antes do then-statement (compatível com seu strict)
            if Parser.lex.next.kind == 'END':
                raise SyntaxError("[Parser] Unexpected token NEWLINE")
            # se vier bloco, aplicar regras estritas
            if Parser.lex.next.kind == 'OPEN_BRA':
                Parser.strict_block_after_control = True
                then_stmt = Parser.parse_block()
                Parser.strict_block_after_control = False
            else:
                then_stmt = Parser.parse_statement()

            if Parser.lex.next.kind == 'ELSE':
                Parser.lex.select_next()
                if Parser.lex.next.kind == 'END':
                    raise SyntaxError("[Parser] Unexpected token NEWLINE")
                if Parser.lex.next.kind == 'OPEN_BRA':
                    Parser.strict_block_after_control = True
                    else_stmt = Parser.parse_block()
                    Parser.strict_block_after_control = False
                else:
                    else_stmt = Parser.parse_statement()
                return If(children=[cond, then_stmt, else_stmt])
            return If(children=[cond, then_stmt])

        if token.kind == 'WHILE':
            Parser.lex.select_next()
            cond = None
            if Parser.lex.next.kind == 'OPEN_PAR':
                Parser.lex.select_next()
                cond = Parser.parse_bool_expression()
                if Parser.lex.next.kind != 'CLOSE_PAR':
                    raise SyntaxError("[Parser] Expected ')' after while condition")
                Parser.lex.select_next()
                while Parser.lex.next.kind in ('AND', 'OR'):
                    op = '&&' if Parser.lex.next.kind == 'AND' else '||'
                    Parser.lex.select_next()
                    right = Parser.parse_bool_term()
                    cond = BinOp(op, [cond, right])
            else:
                cond = Parser.parse_bool_expression()

            if Parser.lex.next.kind == 'END':
                raise SyntaxError("[Parser] Unexpected token NEWLINE")
            if Parser.lex.next.kind == 'OPEN_BRA':
                Parser.strict_block_after_control = True
                body = Parser.parse_block()
                Parser.strict_block_after_control = False
            else:
                body = Parser.parse_statement()
            return While(children=[cond, body])

        if token.kind == 'IDEN':
            name = token.value
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'ASSIGN':
                raise SyntaxError("[Parser] Expected '=' after identifier")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
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
