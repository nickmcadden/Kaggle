DATA_SECTION

  init_int nobs
  init_number decaymod
  init_ivector hscore(1,5131)
  init_ivector ascore(1,5131)
  init_vector hteam(1,5131)
  init_vector ateam(1,5131)
  init_vector yearlag(1,5131)
  init_vector hmean(1,5131)
  init_vector amean(1,5131)
  init_vector distance(1,5131)
  init_vector attendance(1,5131)
  init_ivector iconf(1,5131)
  init_ivector hbias(1,5131)
  init_int nobs
  init_number decaymod
  init_ivector hscore(1,5131)
  init_ivector ascore(1,5131)
  init_vector hteam(1,5131)
  init_vector ateam(1,5131)
  init_vector yearlag(1,5131)
  init_vector hmean(1,5131)
  init_vector amean(1,5131)
  init_vector distance(1,5131)
  init_vector attendance(1,5131)
  init_ivector iconf(1,5131)
  init_ivector hbias(1,5131)
  init_int dummy
PARAMETER_SECTION
  objective_function_value f
  init_bounded_vector att(1,364,0,10)
  init_bounded_vector def(1,364,0,5)
  init_bounded_number dcf(0.0001,0.07)
  init_bounded_number ccf(0.0001,0.07)
  init_bounded_number hcf(0.0001,0.07)
  init_bounded_number icf(0.0001,0.07)
  number lamda
  number mu
  number p
  number weight
  number mean_weight

PROCEDURE_SECTION
  int h=0;
  int a=0;
  double x=0;
  double y=0;
  weight=0;
  mean_weight=0;

  for(int i=1;i<=nobs;i++)
  {
  x = hscore(i);
  y = ascore(i);
  h = hteam(i);
  a = ateam(i);
  lamda = hmean(i) * att(h) * def(a) * (hcf * hbias(i) + 1) * (dcf * distance(i) + 1) * (ccf  * attendance(i) + 1) * (icf * iconf(i) + 1);
  mu = amean(i) * att(a) * def(h);
  p = log_density_poisson(x,lamda) + log_density_poisson(y,mu);
  weight = 1.70-hbias(i)-iconf(i)/3;
  f -= (p*weight);
  mean_weight += (weight/nobs);
  }
  f = f/nobs/mean_weight;
