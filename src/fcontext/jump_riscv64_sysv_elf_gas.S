.text
.globl jump_fcontext
.type   jump_fcontext, @function

jump_fcontext:
    addi sp, sp, -208
    
    sd ra,   0(sp)
    sd s0,   8(sp)
    sd s1,  16(sp)
    sd s2,  24(sp)
    sd s3,  32(sp)
    sd s4,  40(sp)
    sd s5,  48(sp)
    sd s6,  56(sp)
    sd s7,  64(sp)
    sd s8,  72(sp)
    sd s9,  80(sp)
    sd s10, 88(sp)
    sd s11, 96(sp)
    
    beqz a3, 1f

    fsd fs0,  104(sp)
    fsd fs1,  112(sp)
    fsd fs2,  120(sp)
    fsd fs3,  128(sp)
    fsd fs4,  136(sp)
    fsd fs5,  144(sp)
    fsd fs6,  152(sp)
    fsd fs7,  160(sp)
    fsd fs8,  168(sp)
    fsd fs9,  176(sp)
    fsd fs10, 184(sp)
    fsd fs11, 192(sp)

1:
    sd ra, 200(sp)
    sd sp, 0(a0)
    mv sp, a1
    
    ld ra,    0(sp) # original ra
    ld s0,    8(sp)
    ld s1,   16(sp)
    ld s2,   24(sp)
    ld s3,   32(sp)
    ld s4,   40(sp)
    ld s5,   48(sp)
    ld s6,   56(sp)
    ld s7,   64(sp)
    ld s8,   72(sp)
    ld s9,   80(sp)
    ld s10,  88(sp)
    ld s11,  96(sp)

    beqz a3, 2f

    fld fs0,  104(sp)
    fld fs1,  112(sp)
    fld fs2,  120(sp)
    fld fs3,  128(sp)
    fld fs4,  136(sp)
    fld fs5,  144(sp)
    fld fs6,  152(sp)
    fld fs7,  160(sp)
    fld fs8,  168(sp)
    fld fs9,  176(sp)
    fld fs10, 184(sp)
    fld fs11, 192(sp)
    
2:
    ld t0, 200(sp)
    
    addi sp, sp, 208
    
    mv a0, a2
    
    jr t0
    
.size  jump_fcontext, .-jump_fcontext
