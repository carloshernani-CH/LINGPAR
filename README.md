# LINGPAR

[![Compilation Status](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)

## EBNF (v1.1)

```
EXPRESSION = TERM, { ("+" | "-"), TERM } ;
TERM       = FACTOR, { ("*" | "/"), FACTOR } ;
FACTOR     = ("+" | "-"), FACTOR | "(", EXPRESSION, ")" | NUMBER ;
NUMBER     = DIGIT, { DIGIT } ;
DIGIT      = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 ;
```

## Diagrama Sintático (texto)

### EXPRESSION

```
EXPRESSION
 └── TERM
      └── { (+ | -) TERM }
```

### TERM

```
TERM
 └── FACTOR
      └── { (* | /) FACTOR }
```

### FACTOR

```
FACTOR
 ├── (+ | -) FACTOR   (operadores unários)
 ├── "(" EXPRESSION ")"   (parênteses)
 └── NUMBER
```
