import sys
import re
from typing import Dict, Optional

class PrePro:
    @staticmethod
    def filter(code: str) -> str:
        return re.sub(r"//.*$", "", code, flags=re.MULTILINE)

class Token:
    def __init__(self, kind: str, value=None):
        self.kind = kind
        self.value = value

class Lexer:
    # Apenas estes dois viram PRINT; demais variações NÃO.
    PRINT_FORMS = {"Println", "PRINTLN"}
    READ_FORMS  = {"Scanln", "SCANLN"}

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.next: Optional[Token] = None

    def _peek(self): 
        return None if self.position >= len(self.source) else self.source[self.position]
    def _advance(self):
        ch = self._peek()
        if ch is not None: self.position += 1
        return ch
    def _consume_while(self, p):
        s = self.position
        while (c:=self._peek()) is not None and p(c): self._advance()
        return self.source[s:self.position]

    def select_next(self) -> Token:
        while True:
            c = self._peek()
            if c is None: self.next = Token('EOF',''); return self.next
            if c in (' ','\t','\r'): self._advance(); continue
            break

        c = self._peek()
        if c == '\n':
            self._advance(); self.next = Token('END','\n'); return self.next

        if c.isdigit():
            num = self._consume_while(str.isdigit)
            self.next = Token('INT', int(num)); return self.next

        if c.isalpha() or c=='_':
            ident = self._consume_while(lambda ch: ch.isalnum() or ch=='_')
            # case-sensitive para Println / Scanln
            if ident in Lexer.PRINT_FORMS:
                self.next = Token('PRINT','PRINT'); return self.next
            if ident in Lexer.READ_FORMS:
                self.next = Token('READ','READ'); return self.next
            # case-insensitive para if/else/for/while
            low = ident.lower()
            if low == 'if':    self.next = Token('IF','if'); return self.next
            if low == 'else':  self.next = Token('ELSE','else'); return self.next
            if low == 'while': self.next = Token('WHILE','while'); return self.next
            if low == 'for':   self.next = Token('WHILE','for');   return self.next  # 'for' = while
            self.next = Token('IDEN', ident); return self.next

        # compostos
        if c == '&' and self.position+1 < len(self.source) and self.source[self.position+1]=='&':
            self.position += 2; self.next = Token('AND','&&'); return self.next
        if c == '|' and self.position+1 < len(self.source) and self.source[self.position+1]=='|':
            self.position += 2; self.next = Token('OR','||');  return self.next
        if c == '=' and self.position+1 < len(self.source) and self.source[self.position+1]=='=':
            self.position += 2; self.next = Token('EQ','==');  return self.next

        table = {'+':'PLUS','-':'MINUS','*':'MULT','/':'DIV',
                 '(':'OPEN_PAR',')':'CLOSE_PAR','{':'OPEN_BRA','}':'CLOSE_BRA',
                 '=':'ASSIGN','<':'LT','>':'GT','!':'NOT'}
        if c in table:
            self._advance(); self.next = Token(table[c], c); return self.next

        raise SyntaxError(f"[Lexer] Símbolo inválido {c!r} em {self.position}")

class Variable:
    def __init__(self, value: int): self.value = value

class SymbolTable:
    def __init__(self): self._table: Dict[str, Variable] = {}
    def get(self, name: str) -> Variable:
        if name not in self._table: raise NameError(f"[SymbolTable] Variável '{name}' não definida")
        return self._table[name]
    def set(self, name: str, value: int):
        if name in Lexer.PRINT_FORMS or name in Lexer.READ_FORMS or name.lower() in {'if','else','while','for'}:
            raise SyntaxError(f"[SymbolTable] '{name}' é palavra reservada")
        self._table[name] = Variable(value)

class Node:
    def __init__(self, value=None, children=None): self.value=value; self.children=children or []
    def evaluate(self, st: SymbolTable): raise NotImplementedError()

class IntVal(Node):
    def evaluate(self, st): return int(self.value)
class Identifier(Node):
    def evaluate(self, st): return st.get(self.value).value
class Read(Node):
    def evaluate(self, st): return int(input())
class UnOp(Node):
    def evaluate(self, st):
        v = self.children[0].evaluate(st)
        if self.value=='+': return +v
        if self.value=='-': return -v
        if self.value=='!': return 0 if v else 1
        raise ValueError(f"[UnOp] {self.value}")
class BinOp(Node):
    def evaluate(self, st):
        a = self.children[0].evaluate(st); b = self.children[1].evaluate(st)
        if self.value=='+': return a+b
        if self.value=='-': return a-b
        if self.value=='*': return a*b
        if self.value=='/':
            if b==0: raise ZeroDivisionError("Divisão por zero"); 
            return a//b
        if self.value=='==': return 1 if a==b else 0
        if self.value=='<':  return 1 if a<b  else 0
        if self.value=='>':  return 1 if a>b  else 0
        if self.value=='&&': return 1 if (a!=0 and b!=0) else 0
        if self.value=='||': return 1 if (a!=0 or  b!=0) else 0
        raise ValueError(f"[BinOp] {self.value}")

class Assignment(Node):
    def evaluate(self, st):
        name = self.children[0].value
        st.set(name, self.children[1].evaluate(st))
class Print(Node):
    def evaluate(self, st): print(self.children[0].evaluate(st))
class Block(Node):
    def evaluate(self, st):
        for c in self.children: c.evaluate(st)
class If(Node):
    def evaluate(self, st):
        if self.children[0].evaluate(st)!=0: self.children[1].evaluate(st)
        elif len(self.children)==3: self.children[2].evaluate(st)
class While(Node):
    def evaluate(self, st):
        while self.children[0].evaluate(st)!=0: self.children[1].evaluate(st)
class NoOp(Node):
    def evaluate(self, st): return None

class Parser:
    lex: Lexer = None

    @staticmethod
    def parse_factor() -> Node:
        t = Parser.lex.next
        if t.kind in ('PLUS','MINUS','NOT'):
            op = {'PLUS':'+','MINUS':'-','NOT':'!'}[t.kind]
            Parser.lex.select_next()
            return UnOp(op,[Parser.parse_factor()])
        if t.kind=='OPEN_PAR':
            Parser.lex.select_next()
            node = Parser.parse_bool_expression()
            if Parser.lex.next.kind!='CLOSE_PAR': raise SyntaxError("[Parser] Esperado ')'")
            Parser.lex.select_next(); return node
        if t.kind=='READ':
            Parser.lex.select_next()
            if Parser.lex.next.kind!='OPEN_PAR':  raise SyntaxError("[Parser] Esperado '(' após Scanln")
            Parser.lex.select_next()
            if Parser.lex.next.kind!='CLOSE_PAR': raise SyntaxError("[Parser] Esperado ')' em Scanln()")
            Parser.lex.select_next(); return Read()
        if t.kind=='INT': Parser.lex.select_next(); return IntVal(t.value)
        if t.kind=='IDEN': Parser.lex.select_next(); return Identifier(t.value)
        raise SyntaxError(f"[Parser] Token inesperado em Factor: {t.kind}")

    @staticmethod
    def parse_term() -> Node:
        node = Parser.parse_factor()
        while Parser.lex.next.kind in ('MULT','DIV'):
            op = Parser.lex.next.value; Parser.lex.select_next()
            node = BinOp(op,[node, Parser.parse_factor()])
        return node

    @staticmethod
    def parse_expression() -> Node:
        node = Parser.parse_term()
        while Parser.lex.next.kind in ('PLUS','MINUS'):
            op = Parser.lex.next.value; Parser.lex.select_next()
            node = BinOp(op,[node, Parser.parse_term()])
        return node

    @staticmethod
    def parse_rel_expression() -> Node:
        node = Parser.parse_expression()
        while Parser.lex.next.kind in ('EQ','LT','GT'):
            kind = Parser.lex.next.kind
            op = '==' if kind=='EQ' else '<' if kind=='LT' else '>'
            Parser.lex.select_next()
            node = BinOp(op,[node, Parser.parse_expression()])
        return node

    @staticmethod
    def parse_bool_term() -> Node:
        node = Parser.parse_rel_expression()
        while Parser.lex.next.kind=='AND':
            Parser.lex.select_next()
            node = BinOp('&&',[node, Parser.parse_rel_expression()])
        return node

    @staticmethod
    def parse_bool_expression() -> Node:
        node = Parser.parse_bool_term()
        while Parser.lex.next.kind=='OR':
            Parser.lex.select_next()
            node = BinOp('||',[node, Parser.parse_bool_term()])
        return node

    @staticmethod
    def parse_block() -> Node:
        if Parser.lex.next.kind!='OPEN_BRA':
            stmt = Parser.parse_statement()
            return Block(children=[] if stmt is None else [stmt])
        Parser.lex.select_next()
        stmts=[]
        while Parser.lex.next.kind=='END': Parser.lex.select_next()
        while Parser.lex.next.kind not in ('CLOSE_BRA','EOF'):
            s = Parser.parse_statement()
            if s is not None: stmts.append(s)
            while Parser.lex.next.kind=='END': Parser.lex.select_next()
        if Parser.lex.next.kind!='CLOSE_BRA': raise SyntaxError("[Parser] Esperado '}'")
        Parser.lex.select_next()
        return Block(children=stmts)

    @staticmethod
    def _parse_cond_allow_optional_parens() -> Node:
        # if/for podem vir como "(BoolExpr)" ou direto "BoolExpr"
        if Parser.lex.next.kind=='OPEN_PAR':
            Parser.lex.select_next()
            cond = Parser.parse_bool_expression()
            if Parser.lex.next.kind!='CLOSE_PAR': raise SyntaxError("[Parser] Esperado ')' na condição")
            Parser.lex.select_next()
            return cond
        return Parser.parse_bool_expression()

    @staticmethod
    def parse_statement() -> Optional[Node]:
        t = Parser.lex.next
        if t.kind=='END': Parser.lex.select_next(); return NoOp()
        if t.kind=='PRINT':
            Parser.lex.select_next()
            if Parser.lex.next.kind!='OPEN_PAR': raise SyntaxError("[Parser] Esperado '(' após Println")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind!='CLOSE_PAR': raise SyntaxError("[Parser] Esperado ')' após Println")
            Parser.lex.select_next()
            if Parser.lex.next.kind=='END': Parser.lex.select_next()
            return Print(children=[expr])
        if t.kind=='IF':
            Parser.lex.select_next()
            cond = Parser._parse_cond_allow_optional_parens()
            then_blk = Parser.parse_block()
            if Parser.lex.next.kind=='ELSE':
                Parser.lex.select_next()
                else_blk = Parser.parse_block()
                return If(children=[cond, then_blk, else_blk])
            return If(children=[cond, then_blk])
        if t.kind=='WHILE':
            Parser.lex.select_next()
            cond = Parser._parse_cond_allow_optional_parens()
            body = Parser.parse_block()
            return While(children=[cond, body])
        if t.kind=='OPEN_BRA':
            return Parser.parse_block()
        if t.kind=='IDEN':
            name = t.value; Parser.lex.select_next()
            if Parser.lex.next.kind!='ASSIGN': raise SyntaxError("[Parser] Esperado '=' após identificador")
            Parser.lex.select_next()
            expr = Parser.parse_bool_expression()
            if Parser.lex.next.kind=='END': Parser.lex.select_next()
            return Assignment(children=[Identifier(name), expr])
        if t.kind=='EOF': return None
        raise SyntaxError(f"[Parser] Início inesperado de statement: {t.kind}")

    @staticmethod
    def parse_program() -> Block:
        stmts=[]
        while Parser.lex.next.kind=='END': Parser.lex.select_next()
        while Parser.lex.next.kind!='EOF':
            s = Parser.parse_statement()
            if s is not None: stmts.append(s)
            while Parser.lex.next.kind=='END': Parser.lex.select_next()
        return Block(children=stmts)

    @staticmethod
    def run(code: str) -> Block:
        Parser.lex = Lexer(code); Parser.lex.select_next()
        prog = Parser.parse_program()
        if Parser.lex.next.kind!='EOF':
            raise SyntaxError(f"[Parser] Token inesperado após programa: {Parser.lex.next.kind}")
        return prog

def main():
    if len(sys.argv)<2:
        print("Uso: python3 main.py <arquivo-fonte>"); sys.exit(1)
    with open(sys.argv[1],'r',encoding='utf-8') as f: raw = f.read()
    code = PrePro.filter(raw)
    ast = Parser.run(code)
    st = SymbolTable()
    ast.evaluate(st)

if __name__=="__main__":
    main()
