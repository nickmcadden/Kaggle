#include <admodel.h>

  extern "C"  {
    void ad_boundf(int i);
  }
#include <ncaa_exponential_gen_gen.htp>

model_data::model_data(int argc,char * argv[]) : ad_comm(argc,argv)
{
  nobs.allocate("nobs");
  decaymod.allocate("decaymod");
  wscore.allocate(1,4936,"wscore");
  lscore.allocate(1,4936,"lscore");
  wteam.allocate(1,4936,"wteam");
  lteam.allocate(1,4936,"lteam");
  yearlag.allocate(1,4936,"yearlag");
  xmean.allocate(1,4936,"xmean");
  ymean.allocate(1,4936,"ymean");
  nobs.allocate("nobs");
  decaymod.allocate("decaymod");
  wscore.allocate(1,5052,"wscore");
  lscore.allocate(1,5052,"lscore");
  wteam.allocate(1,5052,"wteam");
  lteam.allocate(1,5052,"lteam");
  yearlag.allocate(1,5052,"yearlag");
  xmean.allocate(1,5052,"xmean");
  ymean.allocate(1,5052,"ymean");
  dummy.allocate("dummy");
}

model_parameters::model_parameters(int sz,int argc,char * argv[]) : 
 model_data(argc,argv) , function_minimizer(sz)
{
  initializationfunction();
  f.allocate("f");
  prior_function_value.allocate("prior_function_value");
  likelihood_function_value.allocate("likelihood_function_value");
  att.allocate(1,364,0,10,"att");
  def.allocate(1,364,0,5,"def");
  lamda.allocate("lamda");
  #ifndef NO_AD_INITIALIZE
  lamda.initialize();
  #endif
  mu.allocate("mu");
  #ifndef NO_AD_INITIALIZE
  mu.initialize();
  #endif
  p.allocate("p");
  #ifndef NO_AD_INITIALIZE
  p.initialize();
  #endif
  weight.allocate("weight");
  #ifndef NO_AD_INITIALIZE
  weight.initialize();
  #endif
  mean_weight.allocate("mean_weight");
  #ifndef NO_AD_INITIALIZE
  mean_weight.initialize();
  #endif
}

void model_parameters::userfunction(void)
{
  f =0.0;
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
}

void model_parameters::preliminary_calculations(void){
  admaster_slave_variable_interface(*this);
}

model_data::~model_data()
{}

model_parameters::~model_parameters()
{}

void model_parameters::report(void){}

void model_parameters::final_calcs(void){}

void model_parameters::set_runtime(void){}

#ifdef _BORLANDC_
  extern unsigned _stklen=10000U;
#endif


#ifdef __ZTC__
  extern unsigned int _stack=10000U;
#endif

  long int arrmblsize=0;

int main(int argc,char * argv[])
{
    ad_set_new_handler();
  ad_exit=&ad_boundf;
    gradient_structure::set_NO_DERIVATIVES();
    gradient_structure::set_YES_SAVE_VARIABLES_VALUES();
    if (!arrmblsize) arrmblsize=15000000;
    model_parameters mp(arrmblsize,argc,argv);
    mp.iprint=10;
    mp.preliminary_calculations();
    mp.computations(argc,argv);
    return 0;
}

extern "C"  {
  void ad_boundf(int i)
  {
    /* so we can stop here */
    exit(i);
  }
}
