DATA_SECTION
  init_int dummy

PARAMETER_SECTION
  objective_function_value f
  init_bounded_vector att(1,364,0,10)
  init_bounded_vector def(1,364,0,5)
  init_bounded_number dcf(0.01,0.05)
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
  lamda = hmean(i) * att(h) * def(a) + (dcf * distance(i));
  // + dcf * distance(i)
  mu = amean(i) * att(a) * def(h);
  p = log_density_poisson(x,lamda) + log_density_poisson(y,mu);
  weight = pow(decaymod,yearlag(i));
  f -= (p*weight);
  mean_weight += (weight/nobs);
  }
  f = f/nobs/mean_weight;
