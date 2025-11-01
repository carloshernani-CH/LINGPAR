import sys
import re
from typing import Dict, Optional

# =========================
# ====== Code Generator ===
# =========================

class Code:
    instructions = []

    @staticmethod
    def append(code: str) -> None:
        Code.instructions.append(code)

    @staticmethod
    def dump(filename: str) -> None:
        with open(filename, 'w') as file:
            # Escrever o cabeçalho
            file.write("section .data\n")
            file.write("  format_out: db \"%d\", 10, 0 ; format do printf\n")
            file.write("  format_in: db \"%d\", 0 ; format do scanf\n")
            file.write("  scan_int: dd 0; 32-bits integer\n")
            file.write("\n")
            file.write("section .text\n")
            file.write("  extern printf\n")
            file.write("  extern scanf\n")
            file.write("  global _start\n")
            file.write("\n")
            file.write("_start:\n")
            file.write("  push ebp\n")
            file.write("  mov ebp, esp\n")
            file.write("\n")
            file.write("  ; aqui começa o codigo gerado:\n")
            file.write("\n")

            # Escreve as instruções armazenadas
            file.write("\n".join(Code.instructions))

            # Escrever as instruções finais
            file.write("\n\n")
            file.write("  ; aqui termina o código gerado\n")
            file.write("\n")
            file.write("  mov esp, ebp\n")
            file.write("  pop ebp\n")
            file.write("\n")
            file.write("  ; chamada da interrupcao de saida (Linux)\n")
            file.write("  mov eax, 1\n")
            file.write("  xor ebx, ebx\n")
            file.write("  int 0x80\n")


# =========================
# ====== Preprocessing ====
# =========================

class PrePro:
    @staticmethod
    def filter(code: str) -> str:
        # Remove tudo após '//' até o fim da linha, preservando '\n' quando existir
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
    RESERVED = {
        "PRINT", "PRINTLN", "if", "else", "while", "for", "read", "Scanln",
        "var", "true", "false", "int", "bool", "string", "func", "return"
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
                raise SyntaxError("[Lexer] Invalid token \"")  # string não terminada
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
                self.next = Token('WHILE', 'for')  # compat
            elif ident_str in ("read", "Scanln"):
                self.next = Token('READ', 'read')
            elif ident_str == "var":
                self.next = Token('VAR', 'var')
            elif ident_str == "func":
                self.next = Token('FUNC', 'func')
            elif ident_str == "return":
                self.next = Token('RETURN', 'return')
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
        if c == ',':
            self._advance(); self.next = Token('COMMA', ','); return self.next

        # Mensagem exata esperada pelo tester
        raise SyntaxError(f"[Lexer] Invalid token {c}")


# =========================
# == Symbol Table / Vars ==
# =========================

class Variable:
    def __init__(self, value, vtype: str, shift: int = 0, is_function: bool = False):
        self.value = value
        self.type = vtype  # "int" | "bool" | "string" | tipo de retorno (para funções)
        self.shift = shift  # deslocamento na pilha (EBP - shift)
        self.is_function = is_function  # indica se é uma função


class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self._table: Dict[str, Variable] = {}
        self._current_shift = 0  # controla o deslocamento atual na pilha
        self.parent = parent  # SymbolTable pai (para escopo)

    @property
    def table(self) -> Dict[str, Variable]:
        return self._table

    @table.setter
    def table(self, new_table: Dict[str, Variable]):
        self._table = new_table

    def get(self, name: str) -> Variable:
        # Busca recursivamente na hierarquia
        if name in self._table:
            return self._table[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise NameError(f"[Semantic] Identifier not found")

    def create_variable(self, name: str, vtype: str, is_function: bool = False):
        if name.upper() in {"PRINT", "PRINTLN"} or name in {"if", "else", "while", "for", "read", "Scanln", "var", "func", "return"}:
            raise SyntaxError(f"[Parser] '{name}' is a reserved word and cannot be used as a variable name")
        if name in self._table:
            raise NameError(f"[Semantic] Variable '{name}' already declared")
        if not is_function and vtype not in ("int", "bool", "string"):
            raise TypeError(f"[Semantic] Unknown type '{vtype}'")
        # Incrementa o deslocamento para cada nova variável (4 bytes por variável)
        if not is_function:
            self._current_shift += 4
            self._table[name] = Variable(None, vtype, self._current_shift, is_function)
        else:
            self._table[name] = Variable(None, vtype, 0, is_function)

    def set(self, name: str, value):
        # Busca recursivamente na hierarquia
        if name in self._table:
            var = self._table[name]
        elif self.parent is not None:
            return self.parent.set(name, value)
        else:
            raise NameError(f"[Semantic] Identifier not found")

        if var.type == "int":
            if not isinstance(value, int) or isinstance(value, BoolInt):
                raise TypeError(f"[Semantic] Expected int for '{name}', got bool")
        elif var.type == "bool":
            if not isinstance(value, BoolInt):
                raise TypeError(f"[Semantic] Expected bool for '{name}'")
        elif var.type == "string":
            if not isinstance(value, str):
                raise TypeError(f"[Semantic] Expected string for '{name}', got {type(value).__name__}")
        else:
            raise TypeError(f"[Semantic] Unsupported variable type '{var.type}'")
        var.value = value


# =========================
# ========= AST ===========
# =========================

class Node:
    _id_counter = 0

    @staticmethod
    def newId() -> int:
        Node._id_counter += 1
        return Node._id_counter

    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children or []
        self.unique_id = Node.newId()

    def evaluate(self, st: SymbolTable):
        raise NotImplementedError()

    def generate(self, st: SymbolTable):
        raise NotImplementedError()

class BoolInt(int):
    """Representa bool (1/0) distinto de int comum para checagem de tipos."""
    pass


class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return int(self.value)

    def generate(self, st: SymbolTable):
        # Coloca o valor inteiro em EAX
        Code.append(f"  mov eax, {self.value}")


class StringVal(Node):
    def evaluate(self, st: SymbolTable):
        return str(self.value)

    def generate(self, st: SymbolTable):
        # Strings não geram código Assembly neste projeto
        pass


class BoolVal(Node):
    def evaluate(self, st: SymbolTable):
        return BoolInt(1 if self.value else 0)

    def generate(self, st: SymbolTable):
        # Coloca o valor booleano em EAX (1 ou 0)
        Code.append(f"  mov eax, {1 if self.value else 0}")


class Identifier(Node):
    def evaluate(self, st: SymbolTable):
        return st.get(self.value).value

    def generate(self, st: SymbolTable):
        # Carrega o valor da variável da pilha para EAX
        var = st.get(self.value)
        Code.append(f"  mov eax, [ebp-{var.shift}]")


class Read(Node):
    def evaluate(self, st: SymbolTable):
        try:
            line = input()
        except EOFError:
            line = "0"
        line = line.strip()
        if not re.fullmatch(r"-?\d+", line or "0"):
            raise ValueError("[Semantic] read() expected integer input")
        return int(line)

    def generate(self, st: SymbolTable):
        # Chama scanf para ler um inteiro
        Code.append("  push scan_int")
        Code.append("  push format_in")
        Code.append("  call scanf")
        Code.append("  add esp, 8")
        Code.append("  mov eax, dword [scan_int]")


class UnOp(Node):
    def evaluate(self, st: SymbolTable):
        v = self.children[0].evaluate(st)
        if self.value == '+':
            if not isinstance(v, int) or isinstance(v, BoolInt):
                raise TypeError("[Semantic] Unary '+' expects int")
            return +v
        elif self.value == '-':
            if not isinstance(v, int) or isinstance(v, BoolInt):
                raise TypeError("[Semantic] Unary '-' expects int")
            return -v
        elif self.value == '!':
            if not isinstance(v, BoolInt):
                raise TypeError("[Semantic] '!' expects bool")
            return BoolInt(0 if v else 1)
        else:
            raise SyntaxError(f"[Parser] Unexpected token {self.value}")

    def generate(self, st: SymbolTable):
        # Gera código para o filho (resultado em EAX)
        self.children[0].generate(st)

        if self.value == '+':
            # Unário +, não faz nada
            pass
        elif self.value == '-':
            # Nega o valor em EAX
            Code.append("  neg eax")
        elif self.value == '!':
            # NOT lógico: inverte 0 e 1
            Code.append("  xor eax, 1")


def _bool_to_str(v: BoolInt) -> str:
    return "true" if v == 1 else "false"


class BinOp(Node):
    def evaluate(self, st: SymbolTable):
        left = self.children[0].evaluate(st)
        right = self.children[1].evaluate(st)
        op = self.value

        if op == '+':
            if isinstance(left, str) or isinstance(right, str):
                lstr, rstr = left, right
                if isinstance(left, BoolInt): lstr = _bool_to_str(left)
                if isinstance(right, BoolInt): rstr = _bool_to_str(right)
                if isinstance(left, int) and not isinstance(left, BoolInt): lstr = str(left)
                if isinstance(right, int) and not isinstance(right, BoolInt): rstr = str(right)
                if isinstance(lstr, str) and isinstance(rstr, str):
                    return lstr + rstr
                raise TypeError("[Semantic] '+' with strings only allows concatenation with int/bool/string")
            if (isinstance(left, int) and not isinstance(left, BoolInt)) and (isinstance(right, int) and not isinstance(right, BoolInt)):
                return left + right
            raise TypeError("[Semantic] '+' requires int+int or string concatenation")

        if op in ('-', '*', '/'):
            if isinstance(left, int) and not isinstance(left, BoolInt) and isinstance(right, int) and not isinstance(right, BoolInt):
                if op == '-': return left - right
                if op == '*': return left * right
                if op == '/':
                    if right == 0: raise ZeroDivisionError("[Semantic] Division by zero")
                    return left // right
            raise TypeError(f"[Semantic] Arithmetic '{op}' requires int,int")

        if op in ('==', '>', '<'):
            if isinstance(left, BoolInt) != isinstance(right, BoolInt):
                raise TypeError(f"[Semantic] Comparison '{op}' requires operands of the same type")
            if isinstance(left, BoolInt) and isinstance(right, BoolInt):
                if op == '==': return BoolInt(1 if int(left) == int(right) else 0)
                raise TypeError(f"[Semantic] '{op}' not supported for bool")
            if type(left) != type(right):
                raise TypeError(f"[Semantic] Comparison '{op}' requires operands of the same type")
            if op == '==': return BoolInt(1 if left == right else 0)
            if not (isinstance(left, int) or isinstance(left, str)):
                raise TypeError(f"[Semantic] '{op}' not supported for type {type(left).__name__}")
            if op == '>': return BoolInt(1 if left > right else 0)
            if op == '<': return BoolInt(1 if left < right else 0)

        if op in ('&&', '||'):
            if not isinstance(left, BoolInt) or not isinstance(right, BoolInt):
                raise TypeError(f"[Semantic] Logical '{op}' requires bool operands")
            if op == '&&': return BoolInt(1 if (left == 1 and right == 1) else 0)
            if op == '||': return BoolInt(1 if (left == 1 or right == 1) else 0)

        raise SyntaxError(f"[Parser] Unexpected token {op}")

    def generate(self, st: SymbolTable):
        op = self.value

        # Operações aritméticas
        if op in ('+', '-', '*', '/'):
            self.children[1].generate(st)  # right -> EAX
            Code.append("  push eax")  # salva right na pilha
            self.children[0].generate(st)  # left -> EAX
            Code.append("  pop ecx")  # recupera right em ECX

            if op == '+':
                Code.append("  add eax, ecx")
            elif op == '-':
                Code.append("  sub eax, ecx")
            elif op == '*':
                Code.append("  imul ecx")
            elif op == '/':
                Code.append("  mov ebx, ecx")  # divisor em EBX
                Code.append("  cdq")  # estende EAX para EDX:EAX
                Code.append("  idiv ebx")  # EAX = EDX:EAX / EBX

        # Operações de comparação
        elif op in ('==', '>', '<'):
            self.children[0].generate(st)  # left -> EAX
            Code.append("  push eax")
            self.children[1].generate(st)  # right -> EAX
            Code.append("  pop ecx")
            Code.append("  cmp ecx, eax")
            Code.append("  mov eax, 0")
            Code.append("  mov ecx, 1")

            if op == '==':
                Code.append("  cmove eax, ecx")
            elif op == '>':
                Code.append("  cmovg eax, ecx")
            elif op == '<':
                Code.append("  cmovl eax, ecx")

        # Operações lógicas
        elif op == '&&':
            self.children[0].generate(st)
            Code.append("  push eax")
            self.children[1].generate(st)
            Code.append("  pop ecx")
            Code.append("  and eax, ecx")

        elif op == '||':
            self.children[0].generate(st)
            Code.append("  push eax")
            self.children[1].generate(st)
            Code.append("  pop ecx")
            Code.append("  or eax, ecx")
            # Normaliza para 0 ou 1
            Code.append("  cmp eax, 0")
            Code.append("  mov eax, 0")
            Code.append("  mov ecx, 1")
            Code.append("  cmovne eax, ecx")


class Assignment(Node):
    def evaluate(self, st: SymbolTable):
        if not isinstance(self.children[0], Identifier):
            raise SyntaxError("[Parser] Left-hand side must be an Identifier")
        name = self.children[0].value
        val = self.children[1].evaluate(st)
        st.set(name, val)
        return None

    def generate(self, st: SymbolTable):
        # Gera código para a expressão do lado direito (resultado em EAX)
        self.children[1].generate(st)
        # Armazena o valor de EAX na variável
        name = self.children[0].value
        var = st.get(name)
        Code.append(f"  mov [ebp-{var.shift}], eax")


class Print(Node):
    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)
        print(_bool_to_str(val) if isinstance(val, BoolInt) else val)
        return None

    def generate(self, st: SymbolTable):
        # Gera código para a expressão (resultado em EAX)
        self.children[0].generate(st)
        # Chama printf para imprimir o valor
        Code.append("  push eax")
        Code.append("  push format_out")
        Code.append("  call printf")
        Code.append("  add esp, 8")


class Block(Node):
    def evaluate(self, st: SymbolTable):
        for child in self.children:
            # Se o filho é um Block, criar novo escopo
            if isinstance(child, Block):
                child_st = SymbolTable(parent=st)
                result = child.evaluate(child_st)
            else:
                result = child.evaluate(st)

            # Se encontrou um Return, propagar o valor
            if isinstance(child, Return):
                return result
            # Se um bloco filho retornou valor, propagar
            if result is not None and isinstance(child, (Block, If, While)):
                return result
        return None

    def generate(self, st: SymbolTable):
        for child in self.children:
            child.generate(st)


class If(Node):
    def evaluate(self, st: SymbolTable):
        cond = self.children[0].evaluate(st)
        if not isinstance(cond, BoolInt):
            raise TypeError("[Semantic] If condition must be bool")
        if cond == 1:
            result = self.children[1].evaluate(st)
            # Propaga retorno se houver
            if result is not None:
                return result
        elif len(self.children) == 3:
            result = self.children[2].evaluate(st)
            # Propaga retorno se houver
            if result is not None:
                return result
        return None

    def generate(self, st: SymbolTable):
        # Gera a condição (resultado em EAX)
        self.children[0].generate(st)

        # Verifica se a condição é falsa
        Code.append("  cmp eax, 0")

        if len(self.children) == 3:  # if-else
            else_label = f"else_{self.unique_id}"
            exit_label = f"exit_{self.unique_id}"

            Code.append(f"  je {else_label}")
            # Bloco then
            self.children[1].generate(st)
            Code.append(f"  jmp {exit_label}")
            # Bloco else
            Code.append(f"{else_label}:")
            self.children[2].generate(st)
            Code.append(f"{exit_label}:")
        else:  # if simples
            exit_label = f"exit_{self.unique_id}"
            Code.append(f"  je {exit_label}")
            # Bloco then
            self.children[1].generate(st)
            Code.append(f"{exit_label}:")


class While(Node):
    def evaluate(self, st: SymbolTable):
        while True:
            cond = self.children[0].evaluate(st)
            if not isinstance(cond, BoolInt):
                raise TypeError("[Semantic] While condition must be bool")
            if cond == 0:
                break
            result = self.children[1].evaluate(st)
            # Propaga retorno se houver (break do loop)
            if result is not None:
                return result
        return None

    def generate(self, st: SymbolTable):
        loop_label = f"loop_{self.unique_id}"
        exit_label = f"exit_{self.unique_id}"

        # Label do início do loop
        Code.append(f"{loop_label}:")

        # Gera a condição (resultado em EAX)
        self.children[0].generate(st)

        # Se falso, sai do loop
        Code.append("  cmp eax, 0")
        Code.append(f"  je {exit_label}")

        # Bloco de comandos
        self.children[1].generate(st)

        # Volta para o início
        Code.append(f"  jmp {loop_label}")
        Code.append(f"{exit_label}:")


class VarDec(Node):
    def evaluate(self, st: SymbolTable):
        vtype = self.value
        if not isinstance(self.children[0], Identifier):
            raise SyntaxError("[Parser] First child must be Identifier")
        name = self.children[0].value

        # TYPE faltou -> erro semântico
        if vtype == "__MISSING_TYPE__":
            raise TypeError("[Semantic] Expected a TYPE after variable name")

        st.create_variable(name, vtype)
        if len(self.children) == 2:
            init_val = self.children[1].evaluate(st)
            st.set(name, init_val)
        return None

    def generate(self, st: SymbolTable):
        vtype = self.value
        name = self.children[0].value

        # Cria a variável na tabela de símbolos (isso já aloca espaço)
        st.create_variable(name, vtype)

        # Aloca espaço na pilha
        Code.append(f"  sub esp, 4 ; var {name} {vtype} [ebp-{st.get(name).shift}]")

        # Se houver inicialização, gera o código
        if len(self.children) == 2:
            self.children[1].generate(st)  # resultado em EAX
            var = st.get(name)
            Code.append(f"  mov [ebp-{var.shift}], eax")


class NoOp(Node):
    def evaluate(self, st: SymbolTable):
        return None

    def generate(self, st: SymbolTable):
        # NoOp não gera código
        pass


class Return(Node):
    def evaluate(self, st: SymbolTable):
        return self.children[0].evaluate(st)

    def generate(self, st: SymbolTable):
        pass


class FuncDec(Node):
    def evaluate(self, st: SymbolTable):
        func_name = self.children[0].value
        return_type = self.value if self.value else "void"
        st.create_variable(func_name, return_type, is_function=True)
        st.get(func_name).value = self
        return None

    def generate(self, st: SymbolTable):
        pass


class FuncCall(Node):
    def evaluate(self, st: SymbolTable):
        func_name = self.value
        func_var = st.get(func_name)

        if not func_var.is_function:
            raise TypeError(f"[Semantic] '{func_name}' is not a function")

        func_dec = func_var.value
        if not isinstance(func_dec, FuncDec):
            raise TypeError(f"[Semantic] Invalid function reference")

        num_params = len(func_dec.children) - 2
        num_args = len(self.children)

        if num_params != num_args:
            raise TypeError(f"[Semantic] Function '{func_name}' expects {num_params} arguments, got {num_args}")

        func_st = SymbolTable(parent=st)

        for i in range(num_params):
            param = func_dec.children[i + 1]
            arg_value = self.children[i].evaluate(st)
            param_name = param.children[0].value
            param_type = param.value
            func_st.create_variable(param_name, param_type)
            func_st.set(param_name, arg_value)

        func_body = func_dec.children[-1]
        result = func_body.evaluate(func_st)

        return_type = func_dec.value
        if return_type and return_type != "void":
            if result is None:
                raise TypeError(f"[Semantic] Function '{func_name}' must return a value of type '{return_type}'")
            if return_type == "int":
                if not isinstance(result, int) or isinstance(result, BoolInt):
                    raise TypeError(f"[Semantic] Function '{func_name}' must return int")
            elif return_type == "bool":
                if not isinstance(result, BoolInt):
                    raise TypeError(f"[Semantic] Function '{func_name}' must return bool")
            elif return_type == "string":
                if not isinstance(result, str):
                    raise TypeError(f"[Semantic] Function '{func_name}' must return string")
        elif result is not None:
            raise TypeError(f"[Semantic] Function '{func_name}' should not return a value")

        return result

    def generate(self, st: SymbolTable):
        pass


# =========================
# ========= Parser ========
# =========================

class Parser:
    lex: Lexer = None
    strict_block_after_control: bool = False
    strict_context: Optional[str] = None  # "IF" | "WHILE" | None
    block_depth: int = 0

    # ---------- Helpers ----------
    @staticmethod
    def _check_missing_right_expr_after_unary():
        """
        Tests:
        - 47: '!' seguido de '{' -> Missing Right Expression
        - 4/5/6: unário seguido de EOL/EOF/'}' -> Unexpected token EOL
        """
        if Parser.lex.next.kind in ('OPEN_BRA',):
            raise SyntaxError("[Parser] Missing Right Expression")
        if Parser.lex.next.kind in ('END', 'EOF', 'CLOSE_BRA'):
            raise SyntaxError("[Parser] Unexpected token EOL")

    # ---------- Factors / Terms / Expressions ----------
    @staticmethod
    def parse_factor() -> Node:
        token = Parser.lex.next

        # Unary operators
        if token.kind in ('PLUS', 'MINUS', 'NOT'):
            op = {'PLUS': '+', 'MINUS': '-', 'NOT': '!'}[token.kind]
            Parser.lex.select_next()
            Parser._check_missing_right_expr_after_unary()
            node = Parser.parse_factor()
            return UnOp(op, [node])

        if token.kind == 'OPEN_PAR':
            Parser.lex.select_next()
            node = Parser.parse_bool_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Missing CLOSE_PAR")
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

            if Parser.lex.next.kind == 'OPEN_PAR':
                Parser.lex.select_next()
                args = []
                while Parser.lex.next.kind != 'CLOSE_PAR':
                    arg = Parser.parse_bool_expression()
                    args.append(arg)
                    if Parser.lex.next.kind == 'COMMA':
                        Parser.lex.select_next()
                    elif Parser.lex.next.kind != 'CLOSE_PAR':
                        raise SyntaxError("[Parser] Expected ',' or ')' in function call")
                Parser.lex.select_next()
                return FuncCall(value=name, children=args)

            return Identifier(name)

        if token.kind == 'READ':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError("[Parser] Missing OPEN_PAR")
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Missing CLOSE_PAR")
            Parser.lex.select_next()
            return Read()

        if token.kind == 'END':
            raise SyntaxError("[Parser] Unexpected token EOL")

        if token.kind == 'EOF':
            raise SyntaxError("[Parser] Unexpected token EOL")

        raise SyntaxError(f"[Parser] Unexpected token {token.kind}")

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
            # Comparação sem operando direito:
            if Parser.lex.next.kind in ('OPEN_BRA', 'END', 'EOF', 'CLOSE_BRA'):
                raise SyntaxError("[Parser] Missing Right Expression")
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
        if Parser.lex.next.kind != 'OPEN_BRA':
            raise SyntaxError("[Parser] Expected '{' to start block")
        Parser.block_depth += 1
        Parser.lex.select_next()

        statements = []
        # Bloco estrito vazio colado ao controle → erro (compat com testes antigos)
        if Parser.strict_block_after_control and Parser.lex.next.kind == 'CLOSE_BRA':
            Parser.block_depth -= 1
            raise SyntaxError("[Parser] Unexpected token INT")

        saw_end = False
        while Parser.lex.next.kind == 'END':
            saw_end = True
            Parser.lex.select_next()

        if Parser.strict_block_after_control and not saw_end and Parser.lex.next.kind == 'CLOSE_BRA':
            Parser.block_depth -= 1
            raise SyntaxError("[Parser] Unexpected token INT")

        while Parser.lex.next.kind not in ('CLOSE_BRA', 'EOF'):
            stmt = Parser.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            while Parser.lex.next.kind == 'END':
                Parser.lex.select_next()

        if Parser.lex.next.kind == 'EOF':
            ctx = Parser.strict_context
            Parser.block_depth -= 1
            if Parser.strict_block_after_control:
                if ctx == 'IF':
                    raise SyntaxError("[Parser] Missing CLOSE_BRA")
                if ctx == 'WHILE':
                    raise SyntaxError("[Parser] Unexpected token EOF")
            raise SyntaxError("[Parser] Unexpected token EOF (expected CLOSE_BRA)")

        Parser.lex.select_next()
        Parser.block_depth -= 1
        return Block(children=statements)

    @staticmethod
    def parse_statement() -> Optional[Node]:
        token = Parser.lex.next

        if token.kind == 'END':
            Parser.lex.select_next()
            return NoOp()

        if token.kind == 'CLOSE_BRA':
            raise SyntaxError("[Parser] Unexpected token CLOSE_BRA")

        if token.kind == 'ELSE':
            # Else "solto" / duplicado
            raise SyntaxError("[Parser] Unexpected ELSE")

        if token.kind == 'OPEN_BRA':
            return Parser.parse_block()

        # var declaration:  var ID [ ":" ] TYPE [= expr] [END]
        if token.kind == 'VAR':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'IDEN':
                raise SyntaxError("[Parser] Unexpected token IDEN")
            ident_name = Parser.lex.next.value
            Parser.lex.select_next()

            if Parser.lex.next.kind == 'COLON':
                Parser.lex.select_next()

            # TYPE => ok | ASSIGN/END/EOF/CLOSE_BRA => TYPE faltou (semântico)
            # IDEN => identificador inesperado (ex: i32) -> Parser
            if Parser.lex.next.kind == 'TYPE':
                vtype = Parser.lex.next.value
                Parser.lex.select_next()
            elif Parser.lex.next.kind in ('ASSIGN', 'END', 'EOF', 'CLOSE_BRA'):
                vtype = "__MISSING_TYPE__"
            elif Parser.lex.next.kind == 'IDEN':
                raise SyntaxError("[Parser] Unexpected Identifier")
            else:
                raise SyntaxError("[Parser] Expected a TYPE after variable name")

            children = [Identifier(ident_name)]
            if Parser.lex.next.kind == 'ASSIGN':
                Parser.lex.select_next()

                if Parser.lex.next.kind == 'END':
                    raise SyntaxError("[Parser] Unexpected token EOL")

                saw_eol = False
                while Parser.lex.next.kind == 'END':
                    saw_eol = True
                    Parser.lex.select_next()

                if Parser.lex.next.kind == 'CLOSE_BRA':
                    raise SyntaxError("[Parser] Unexpected token CLOSE_BRA")
                if Parser.lex.next.kind == 'EOF':
                    if not saw_eol:
                        raise SyntaxError("[Parser] Unexpected token CLOSE_BRA")
                    raise SyntaxError("[Parser] Unexpected token EOL")

                expr = Parser.parse_bool_expression()

                if Parser.lex.next.kind not in ('END', 'CLOSE_BRA', 'EOF'):
                    raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind}")

                children.append(expr)

            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()

            return VarDec(value=vtype, children=children)

        if token.kind == 'PRINT':
            Parser.lex.select_next()
            if Parser.lex.next.kind != 'OPEN_PAR':
                raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind} (expected OPEN_PAR)")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind in ('END', 'EOF'):
                raise SyntaxError("[Parser] Unexpected token EOL (expected CLOSE_PAR)")
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind}")
            Parser.lex.select_next()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Print(children=[expr])

        if token.kind == 'IF':
            Parser.lex.select_next()
            # if (cond) ...  ou  if cond ...
            if Parser.lex.next.kind == 'OPEN_PAR':
                Parser.lex.select_next()
                cond = Parser.parse_bool_expression()
                if Parser.lex.next.kind != 'CLOSE_PAR':
                    raise SyntaxError("[Parser] Missing CLOSE_PAR")
                Parser.lex.select_next()
            else:
                cond = Parser.parse_bool_expression()
            while Parser.lex.next.kind in ('AND', 'OR'):
                op = '&&' if Parser.lex.next.kind == 'AND' else '||'
                Parser.lex.select_next()
                right = Parser.parse_bool_term()
                cond = BinOp(op, [cond, right])

            # Exigir '{' imediatamente após a condição
            if Parser.lex.next.kind == 'END':
                raise SyntaxError("[Parser] Missing OPEN_BRA")
            if Parser.lex.next.kind != 'OPEN_BRA':
                raise SyntaxError("[Parser] Missing OPEN_BRA")

            Parser.strict_block_after_control = True
            Parser.strict_context = 'IF'
            then_stmt = Parser.parse_block()
            Parser.strict_block_after_control = False
            Parser.strict_context = None

            # Detectar NEWLINE antes do else (Test 52) sem quebrar Test 1
            saw_newline = False
            while Parser.lex.next.kind == 'END':
                saw_newline = True
                Parser.lex.select_next()

            if Parser.lex.next.kind == 'ELSE':
                if saw_newline:
                    # Caso '}\nelse' -> deve falhar
                    raise SyntaxError("[Parser] Unexpected token NEWLINE Before Else")
                Parser.lex.select_next()
                if Parser.lex.next.kind != 'OPEN_BRA':
                    raise SyntaxError("[Parser] Missing OPEN_BRA")
                Parser.strict_block_after_control = True
                Parser.strict_context = 'IF'
                else_stmt = Parser.parse_block()
                Parser.strict_block_after_control = False
                Parser.strict_context = None
                return If(children=[cond, then_stmt, else_stmt])

            return If(children=[cond, then_stmt])

        if token.kind == 'WHILE':
            Parser.lex.select_next()
            if Parser.lex.next.kind == 'OPEN_PAR':
                Parser.lex.select_next()
                cond = Parser.parse_bool_expression()
                if Parser.lex.next.kind != 'CLOSE_PAR':
                    raise SyntaxError("[Parser] Missing CLOSE_PAR")
                Parser.lex.select_next()
            else:
                cond = Parser.parse_bool_expression()
            while Parser.lex.next.kind in ('AND', 'OR'):
                op = '&&' if Parser.lex.next.kind == 'AND' else '||'
                Parser.lex.select_next()
                right = Parser.parse_bool_term()
                cond = BinOp(op, [cond, right])

            if Parser.lex.next.kind != 'OPEN_BRA':
                raise SyntaxError("[Parser] Missing OPEN_BRA")
            Parser.strict_block_after_control = True
            Parser.strict_context = 'WHILE'
            body = Parser.parse_block()
            Parser.strict_block_after_control = False
            Parser.strict_context = None
            return While(children=[cond, body])

        if token.kind == 'RETURN':
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Return(children=[expr])

        if token.kind == 'IDEN':
            name = token.value
            Parser.lex.select_next()

            if Parser.lex.next.kind == 'OPEN_PAR':
                Parser.lex.select_next()
                args = []
                while Parser.lex.next.kind != 'CLOSE_PAR':
                    arg = Parser.parse_bool_expression()
                    args.append(arg)
                    if Parser.lex.next.kind == 'COMMA':
                        Parser.lex.select_next()
                    elif Parser.lex.next.kind != 'CLOSE_PAR':
                        raise SyntaxError("[Parser] Expected ',' or ')' in function call")
                Parser.lex.select_next()
                if Parser.lex.next.kind == 'END':
                    Parser.lex.select_next()
                return FuncCall(value=name, children=args)

            if Parser.lex.next.kind != 'ASSIGN':
                raise SyntaxError("[Parser] Expected '=' after identifier")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind == 'END':
                Parser.lex.select_next()
            return Assignment(children=[Identifier(name), expr])

        if token.kind == 'EOF':
            return None

        raise SyntaxError(f"[Parser] Unexpected token {token.kind}")

    @staticmethod
    def parse_func_declaration() -> FuncDec:
        Parser.lex.select_next()

        if Parser.lex.next.kind != 'IDEN':
            raise SyntaxError("[Parser] Expected function name")
        func_name = Parser.lex.next.value
        Parser.lex.select_next()

        if Parser.lex.next.kind != 'OPEN_PAR':
            raise SyntaxError("[Parser] Expected '(' after function name")
        Parser.lex.select_next()

        params = []
        while Parser.lex.next.kind != 'CLOSE_PAR':
            if Parser.lex.next.kind != 'IDEN':
                raise SyntaxError("[Parser] Expected parameter name")
            param_name = Parser.lex.next.value
            Parser.lex.select_next()

            if Parser.lex.next.kind != 'TYPE':
                raise SyntaxError("[Parser] Expected parameter type")
            param_type = Parser.lex.next.value
            Parser.lex.select_next()

            params.append(VarDec(value=param_type, children=[Identifier(param_name)]))

            if Parser.lex.next.kind == 'COMMA':
                Parser.lex.select_next()
            elif Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected ',' or ')' in parameter list")

        Parser.lex.select_next()

        return_type = None
        if Parser.lex.next.kind == 'TYPE':
            return_type = Parser.lex.next.value
            Parser.lex.select_next()

        while Parser.lex.next.kind == 'END':
            Parser.lex.select_next()

        if Parser.lex.next.kind != 'OPEN_BRA':
            raise SyntaxError("[Parser] Expected '{' to start function body")

        body = Parser.parse_block()

        children = [Identifier(func_name)] + params + [body]
        return FuncDec(value=return_type, children=children)

    @staticmethod
    def parse_program() -> Block:
        statements = []
        while Parser.lex.next.kind == 'END':
            Parser.lex.select_next()

        while Parser.lex.next.kind not in ('EOF', 'CLOSE_BRA'):
            if Parser.lex.next.kind == 'FUNC':
                func_dec = Parser.parse_func_declaration()
                statements.append(func_dec)
            elif Parser.lex.next.kind == 'VAR':
                stmt = Parser.parse_statement()
                if stmt is not None:
                    statements.append(stmt)
            else:
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
            if Parser.lex.next.kind == 'CLOSE_BRA':
                raise SyntaxError("[Parser] Unexpected token CLOSE_BRA (expected EOF)")
            raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind}")

        main_call = FuncCall(value='main', children=[])
        program.children.append(main_call)

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
        print(f"[Lexer] File '{filename}' not found.")
        sys.exit(1)

    try:
        code = PrePro.filter(raw_code)
        ast_root = Parser.run(code)
        st = SymbolTable()
        ast_root.evaluate(st)

    except Exception as e:
        msg = str(e)
        if not (msg.startswith("[Lexer]") or msg.startswith("[Parser]") or msg.startswith("[Semantic]")):
            msg = "[Semantic] " + msg
        print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
