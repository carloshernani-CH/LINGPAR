
# LINGPAR

[![Compilation Status](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)](https://compiler-tester.insper-comp.com.br/svg/carloshernani-CH/LINGPAR)

## EBNF (v1.1)

```ebnf
EXPRESSION = TERM, { ("+" | "-"), TERM } ;
TERM       = FACTOR, { ("*" | "/"), FACTOR } ;
FACTOR     = ("+" | "-"), FACTOR | "(", EXPRESSION, ")" | NUMBER ;
NUMBER     = DIGIT, { DIGIT } ;
DIGIT      = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
````

---

## Diagrama — EXPRESSION

```mermaid
flowchart LR
    A([Start]) --> B([TERM])
    B --> C{next == "+" or "-"?}
    C -- sim --> OP[consome + ou -]
    OP --> T2([TERM])
    T2 --> UPD[result = result (+|-) TERM]
    UPD --> C
    C -- não --> Z([Return result])
```

## Diagrama — TERM

```mermaid
flowchart LR
    A([Start]) --> B([FACTOR])
    B --> C{next == "*" or "/"?}
    C -- sim --> OP[consome * ou /]
    OP --> F2([FACTOR])
    F2 --> UPD[result = result (*|/) FACTOR]
    UPD --> C
    C -- não --> Z([Return result])
```

## Diagrama — FACTOR

```mermaid
flowchart LR
    A([Start]) --> J{Token?}

    J -- "+" ou "-" --> S[consome sinal]
    S --> R[recursão: FACTOR]
    R --> APPLY[retorna ±FACTOR]
    
    J -- "(" --> LP[consome "("]
    LP --> E([EXPRESSION])
    E --> RP{next == ")"?}
    RP -- sim --> CP[consome ")"]
    CP --> RETP[retorna valor de EXPRESSION]
    RP -- não --> ERR1[[SyntaxError: expected ")"]]

    J -- "INT" --> NUM[consome número]
    NUM --> RETN[retorna valor]

    J -- outro --> ERR2[[SyntaxError: token inesperado em FACTOR]]
```

