define float @my_func(float %x) {
  %mx = fneg float %x
  %exp = call  float @llvm.pow.f32( float 2.7182817459106445 , float %mx )
  %one_p_ex = fadd float 1.0 , %exp
  %d = fdiv float 1.0 , %one_p_ex
  %res = fmul float %x , %d
  ret float %res
}

declare float @llvm.pow.f32(float, float)
