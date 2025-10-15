# LINGPAR

[![Compilation Status](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)

## EBNF (v2.2)

```
Program     := { Statement | END }

Block       := "{" { Statement | END } "}"

Statement   := PRINT "(" BoolExpr ")" [END]
             | IF "(" BoolExpr ")" Block [ELSE Block]
             | WHILE "(" BoolExpr ")" Block
             | VAR IDEN ":" TYPE [ "=" BoolExpr ] [END]
             | IDEN "=" BoolExpr [END]
             | Block
             | END

BoolExpr    := BoolTerm { "||" BoolTerm }
BoolTerm    := RelExpr  { "&&" RelExpr  }
RelExpr     := Expression { ("==" | "<" | ">") Expression }
Expression  := Term { ("+" | "-") Term }
Term        := Factor { ("*" | "/") Factor }

Factor      := INT
             | STR
             | TRUE
             | FALSE
             | IDEN
             | READ "(" ")"
             | "(" BoolExpr ")"
             | ("+" | "-" | "!") Factor

TYPE        := "int" | "bool" | "string"
STR         := '"' { any_char_except_quote_or_newline } '"'
TRUE        := "true"
FALSE       := "false"
END         := "\n" | ";"
```
