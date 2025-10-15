# >>> Permitir newline(s) antes do else (Test 1)
while Parser.lex.next.kind == 'END':
    Parser.lex.select_next()

if Parser.lex.next.kind == 'ELSE':
    Parser.lex.select_next()
    if Parser.lex.next.kind != 'OPEN_BRA':
        # Mensagem consistente
        raise SyntaxError("[Parser] Missing OPEN_BRA")
    Parser.strict_block_after_control = True
    Parser.strict_context = 'IF'
    else_stmt = Parser.parse_block()
    Parser.strict_block_after_control = False
    Parser.strict_context = None
    return If(children=[cond, then_stmt, else_stmt])

return If(children=[cond, then_stmt])
