# main.py
import sys

class Token:
    def __init__(self, kind: str, value=None):
        self.kind = kind   # 'INT', 'PLUS', 'MINUS', 'EOF', 'MULT', 'DIV'
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


        # Símbolo inválido
        raise SyntaxError(f"[Lexer] Invalid symbol {c!r}")

class Parser:
    # Atributo estático: o Lexer em uso
    lex: Lexer = None

    @staticmethod
    def parse_term():
        result = Parser.parse_factor()
        while Parser.lex.next.kind in ('MULT', 'DIV'):
            op = Parser.lex.next.kind
            Parser.lex.select_next()
            rhs = Parser.parse_factor()
            if op == 'DIV' and rhs == 0:
                raise ZeroDivisionError("[Parser] Division by zero")
            result = result * rhs if op == 'MULT' else result // rhs
        return result

    @staticmethod
    def parse_factor():
        if Parser.lex.next.kind in ('PLUS', 'MINUS'):
            op = Parser.lex.next.kind
            Parser.lex.select_next()
            value = Parser.parse_factor()
            return value if op == 'PLUS' else -value
        if Parser.lex.next.kind == 'OPEN_PAR':
            Parser.lex.select_next()
            value = Parser.parse_expression()
            if Parser.lex.next.kind != 'CLOSE_PAR':
                raise SyntaxError("[Parser] Expected closing parenthesis")
            Parser.lex.select_next()
            return value
        if Parser.lex.next.kind == 'INT':
            value = Parser.lex.next.value
            Parser.lex.select_next()
            return value
        else:
            raise SyntaxError(f"[Parser] Unexpected token {Parser.lex.next.kind}, expected INT or '('")

    @staticmethod
    def parse_expression() -> int:
        result = Parser.parse_term()
        # Enquanto houver + ou -
        while Parser.lex.next.kind in ('PLUS', 'MINUS'):
            op = Parser.lex.next.kind
            Parser.lex.select_next()
            rhs = Parser.parse_term()
            result = result + rhs if op == 'PLUS' else result - rhs
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
