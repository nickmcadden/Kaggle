DATA_SECTION

  init_int nobs
  init_number decaymod
  init_ivector wscore(1,5018)
  init_ivector lscore(1,5018)
  init_vector wteam(1,5018)
  init_vector lteam(1,5018)
  init_vector yearlag(1,5018)
  init_vector xmean(1,5018)
  init_vector ymean(1,5018)
  init_int dummy
PARAMETER_SECTION
  objective_function_value f
  init_bounded_vector att(1,364,0,10)
  init_bounded_vector def(1,364,0,5)
  number lamda
  number mu
  number p
  number weight
  number mean_weight

PROCEDURE_SECTION
  int w=0;
  int l=0;
  double x=0;
  double y=0;
  weight=0;
  mean_weight=0;

  for(int i=1;i<=nobs;i++)
  {
  x = wscore(i);
  y = lscore(i);
  w = wteam(i);
  l = lteam(i);
  lamda = xmean(i) * att(w) * def(l);
  mu = ymean(i) * att(l) * def(w);
  p = log_density_poisson(x,lamda) + log_density_poisson(y,mu);
  weight = pow(decaymod,yearlag(i));
  f -= (p*weight);
  mean_weight += (weight/nobs);
  }
  f = f/nobs/mean_weight;
