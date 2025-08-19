# main.py
import sys
import re

# --------- LÉXICO ---------
class Token:
    def __init__(self, kind, value=None):
        self.kind = kind  # 'NUM', 'PLUS', 'MINUS', 'EOF'
        self.value = value

class Lexer:
    def __init__(self, text: str):
        self.text = text # string de entrada
        self.i = 0 # índice atual
        self.n = len(text) # comprimento da string

    def _peek(self):
        return self.text[self.i] if self.i < self.n else None #retorna o caractere  e verifica se não é o ultimo

    def _advance(self): #retorna o caractere atual e vai pro proximo
        ch = self._peek() 
        self.i += 1
        return ch

    def next_token(self) -> Token:
        # pular espaços
        while (c := self._peek()) is not None and c.isspace():
            self._advance()

        c = self._peek()
        if c is None:
            return Token('EOF')

        if c.isdigit():
            start = self.i
            while (c := self._peek()) is not None and c.isdigit():
                self._advance()
            return Token('NUM', int(self.text[start:self.i]))

        if c == '+':
            self._advance()
            return Token('PLUS')

        if c == '-':
            self._advance()
            return Token('MINUS')

        # Qualquer outro símbolo é inválido
        raise SyntaxError(f"Caractere inválido: {c!r}")

# --------- SINTÁTICO/AVALIADOR ---------
class Parser:
    """
    Gramática simples (sem precedência extra):
    expr  -> NUM ((PLUS|MINUS) NUM)*
    """
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.cur = self.lexer.next_token()

    def _eat(self, kind: str):
        if self.cur.kind == kind:
            self.cur = self.lexer.next_token()
        else:
            raise SyntaxError(f"Esperado {kind}, encontrei {self.cur.kind}")

    def parse_and_eval(self) -> int:
        # expr inicia obrigatoriamente por NUM
        if self.cur.kind != 'NUM':
            raise SyntaxError("Expressão deve começar com número")
        result = self.cur.value
        self._eat('NUM')

        # ((+|-) NUM)*
        while self.cur.kind in ('PLUS', 'MINUS'):
            op = self.cur.kind
            self._eat(op)
            if self.cur.kind != 'NUM':
                raise SyntaxError("Operador deve ser seguido por número")
            if op == 'PLUS':
                result += self.cur.value
            else:
                result -= self.cur.value
            self._eat('NUM')

        # Nada mais deve restar (além de EOF)
        if self.cur.kind != 'EOF':
            # Ex.: "1 1" vira NUM NUM -> erro aqui
            raise SyntaxError("Entrada malformada")
        return result

# --------- ENTRADA/SAÍDA ---------
def main():
    if len(sys.argv) != 2:
        # Não imprimir instruções: o roteiro quer apenas número em sucesso.
        # Em erro, lançar exceção (vai para stderr) e encerrar.
        raise RuntimeError("Forneça exatamente 1 argumento entre aspas, ex: python3 main.py '1 + 1'")
    text = sys.argv[1]
    # Checagem opcional: permitir apenas dígitos, +, -, espaços
    if not re.fullmatch(r"[0-9+\-\s]+", text):
        raise SyntaxError("Somente números, +, - e espaços são permitidos")
    lexer = Lexer(text)
    parser = Parser(lexer)
    value = parser.parse_and_eval()
    print(value)

if __name__ == "__main__":
    main()
