var b int = 5

func soma(x int, y int) int {
  var a int
  a = x + y 
  Println(a)
  return a
}

func main()  {
  var a int
  {
    var b int
    a = 3
    b = soma(a, 4)
    Println(b)
  }
  Println(a)
  Println(b)
}
