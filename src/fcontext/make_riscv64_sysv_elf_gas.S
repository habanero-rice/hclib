.text
.globl make_fcontext
.type   make_fcontext, @function

make_fcontext:
    andi a0, a0, -16
    addi a0, a0, -208
    
    la t0, finish
    sd t0, 0(a0)
    sd a2, 200(a0)
    
    ret
    
finish:
    addi sp, sp, -16
    sd   a0, 0(sp)
    mv   a0, zero
    call _exit

.size  make_fcontext, .-make_fcontext
