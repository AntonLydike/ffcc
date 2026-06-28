define <8 x float> @my_func(<8 x float> %x, i32 %sigma_1_val, i32 %sigma_val) {
  %vec = insertelement <8 x i32> undef, i32 %sigma_val, i32 0
  %sigma = shufflevector <8 x i32> %vec, <8 x i32> undef, <8 x i32> zeroinitializer

  %vec1 = insertelement <8 x i32> undef, i32 %sigma_1_val, i32 0
  %sigma_1 = shufflevector <8 x i32> %vec1, <8 x i32> undef, <8 x i32> zeroinitializer

  %mx = fneg <8 x float> %x
  %6 = fmul <8 x float> <float 12102203.0, float 12102203.0, float 12102203.0, float 12102203.0, float 12102203.0, float 12102203.0, float 12102203.0, float 12102203.0> , %mx
  %cast = fptosi <8 x float> %6 to <8 x i32>
  %8 = add <8 x i32> %sigma_1 , %cast
  %9 = bitcast <8 x i32> %8 to <8 x float>
  %one_p_ex = fadd <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0> , %9
  %11 = bitcast <8 x float> %one_p_ex to <8 x i32>
  %12 = sub <8 x i32> %sigma , %11
  %13 = bitcast <8 x i32> %12 to <8 x float>
  ret <8 x float> %13
}
