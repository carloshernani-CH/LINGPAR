# LINGPAR

[![Compilation Status](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)

## EBNF (v2.1)

```
Program     := { Statement | END }
Block       := "{" { Statement | END } "}" | Statement
Statement   := PRINT "(" BoolExpr ")" [END]
            | IF "(" BoolExpr ")" Block [ELSE Block]
            | WHILE "(" BoolExpr ")" Block
            | IDEN "=" BoolExpr [END]
            | Block
            | END
BoolExpr    := BoolTerm { "||" BoolTerm }
BoolTerm    := RelExpr  { "&&" RelExpr  }
RelExpr     := Expression { ("=="|"<"|">") Expression }
Expression  := Term { ("+"|"-") Term }
Term        := Factor { ("*"|"/") Factor }
Factor      := INT
            | IDEN
            | READ "(" ")"
            | "(" BoolExpr ")"
            | ("+"|"-"|"!") Factor
```
