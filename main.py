# main.py
import sys

class Token:
    def __init__(self, kind: str, value=None):
        self.kind = kind   # 'INT', 'PLUS', 'MINUS', 'EOF'
        self.value = value # int | str | ''

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.next = None  # Token atual

    def _peek(self):
        if self.position >= len(self.source):
            return None
        return self.source[self.position]

    def _advance(self):
        ch = self._peek()
        self.position += 1
        return ch

    def select_next(self) -> Token:
        # Ignora espaços
        while (c := self._peek()) is not None and c.isspace():
            self._advance()

        c = self._peek()
        if c is None:
            self.next = Token('EOF', '')
            return self.next

        if c.isdigit():
            start = self.position
            while (c := self._peek()) is not None and c.isdigit():
                self._advance()
            self.next = Token('INT', int(self.source[start:self.position]))
            return self.next

        if c == '+':
            self._advance()
            self.next = Token('PLUS', '+')
            return self.next

        if c == '-':
            self._advance()
            self.next = Token('MINUS', '-')
            return self.next

        # Símbolo inválido
        raise SyntaxError(f"[Lexer] Invalid symbol {c!r}")

class Parser:
    # Atributo estático: o Lexer em uso
    lex: Lexer = None

    @staticmethod
    def parse_expression() -> int:
        # Primeiro token deve ser INT
        if Parser.lex.next.kind != 'INT':
            raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind}, expected INT")
        result = Parser.lex.next.value
        Parser.lex.select_next()

        # Enquanto houver + ou -
        while Parser.lex.next.kind in ('PLUS', 'MINUS'):
            op = Parser.lex.next.kind
            Parser.lex.select_next()

            if Parser.lex.next.kind != 'INT':
                exp = '+' if op == 'PLUS' else '-'
                raise SyntaxError(f"[Parser] Expected INT after '{exp}'")

            if op == 'PLUS':
                result += Parser.lex.next.value
            else:
                result -= Parser.lex.next.value

            Parser.lex.select_next()

        return result

    @staticmethod
    def run(code: str) -> int:
        Parser.lex = Lexer(code)
        Parser.lex.select_next()              # posiciona no primeiro token
        result = Parser.parse_expression()    # parseia expressão

        # Verifica consumo total
        if Parser.lex.next.kind != 'EOF':
            raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind} after expression")

        return result

def main():
    # Aceita argumento ou stdin
    code = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    value = Parser.run(code)
    print(value)

if __name__ == "__main__":
    main()
