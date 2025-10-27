section .data
  format_out: db "%d", 10, 0 ; format do printf
  format_in: db "%d", 0 ; format do scanf
  scan_int: dd 0; 32-bits integer

section .text
  extern printf
  extern scanf
  global _start

_start:
  push ebp
  mov ebp, esp

  ; aqui começa o codigo gerado:

  sub esp, 4 ; var a int [ebp-4]
  mov eax, 5
  mov [ebp-4], eax
  sub esp, 4 ; var b int [ebp-8]
  mov eax, 3
  mov [ebp-8], eax
  sub esp, 4 ; var c int [ebp-12]
  mov eax, [ebp-8]
  push eax
  mov eax, [ebp-4]
  pop ecx
  add eax, ecx
  mov [ebp-12], eax
  mov eax, [ebp-12]
  push eax
  push format_out
  call printf
  add esp, 8

  ; aqui termina o código gerado

  mov esp, ebp
  pop ebp

  ; chamada da interrupcao de saida (Linux)
  mov eax, 1
  xor ebx, ebx
  int 0x80
