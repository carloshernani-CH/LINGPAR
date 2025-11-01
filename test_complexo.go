var contador int = 0

func incrementa() {
  contador = contador + 1
}

func soma(a int, b int) int {
  return a + b
}

func maiorQue(x int, limite int) bool {
  return x > limite
}

func main() {
  Println(soma(10, 20))
  
  incrementa()
  incrementa()
  incrementa()
  Println(contador)
  
  if maiorQue(15, 10) {
    Println(1)
  } else {
    Println(0)
  }
}
