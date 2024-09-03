/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#include "nmodlmutex.h" 
#define nrn_init _nrn_init__AmpaNmda
#define _nrn_initial _nrn_initial__AmpaNmda
#define nrn_cur _nrn_cur__AmpaNmda
#define _nrn_current _nrn_current__AmpaNmda
#define nrn_jacob _nrn_jacob__AmpaNmda
#define nrn_state _nrn_state__AmpaNmda
#define _net_receive _net_receive__AmpaNmda 
#define release release__AmpaNmda 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define mg _p[0]
#define mg_columnindex 0
#define gmax _p[1]
#define gmax_columnindex 1
#define x _p[2]
#define x_columnindex 2
#define mgid _p[3]
#define mgid_columnindex 3
#define ggid _p[4]
#define ggid_columnindex 4
#define srcgid _p[5]
#define srcgid_columnindex 5
#define i _p[6]
#define i_columnindex 6
#define inmda _p[7]
#define inmda_columnindex 7
#define iampa _p[8]
#define iampa_columnindex 8
#define gnmda _p[9]
#define gnmda_columnindex 9
#define Ron _p[10]
#define Ron_columnindex 10
#define Roff _p[11]
#define Roff_columnindex 11
#define gampa _p[12]
#define gampa_columnindex 12
#define synon _p[13]
#define synon_columnindex 13
#define DRon _p[14]
#define DRon_columnindex 14
#define DRoff _p[15]
#define DRoff_columnindex 15
#define Dgampa _p[16]
#define Dgampa_columnindex 16
#define v _p[17]
#define v_columnindex 17
#define _g _p[18]
#define _g_columnindex 18
#define _tsav _p[19]
#define _tsav_columnindex 19
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_mgblock(void*);
 static double _hoc_norm_weight_to_sig(void*);
 static double _hoc_plast(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "mgblock", _hoc_mgblock,
 "norm_weight_to_sig", _hoc_norm_weight_to_sig,
 "plast", _hoc_plast,
 0, 0
};
#define _f_mgblock _f_mgblock_AmpaNmda
#define mgblock mgblock_AmpaNmda
#define norm_weight_to_sig norm_weight_to_sig_AmpaNmda
#define plast plast_AmpaNmda
 extern double _f_mgblock( _threadargsprotocomma_ double );
 extern double mgblock( _threadargsprotocomma_ double );
 extern double norm_weight_to_sig( _threadargsprotocomma_ double );
 extern double plast( _threadargsprotocomma_ double );
 
static void _check_mgblock(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_mgblock(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define Alpha Alpha_AmpaNmda
 double Alpha = 0.35;
#define Beta Beta_AmpaNmda
 double Beta = 0.035;
#define Cdur Cdur_AmpaNmda
 double Cdur = 1;
#define E E_AmpaNmda
 double E = 0;
#define Rtau Rtau_AmpaNmda
 double Rtau = 0;
#define Rinf Rinf_AmpaNmda
 double Rinf = 0;
#define ampatau ampatau_AmpaNmda
 double ampatau = 3;
#define gampafactor gampafactor_AmpaNmda
 double gampafactor = 0.001;
#define ltpinvl ltpinvl_AmpaNmda
 double ltpinvl = 33.33;
#define ltdinvl ltdinvl_AmpaNmda
 double ltdinvl = 250;
#define nmdafactor nmdafactor_AmpaNmda
 double nmdafactor = 0.0035;
#define sigslope sigslope_AmpaNmda
 double sigslope = 3;
#define sighalf sighalf_AmpaNmda
 double sighalf = 25;
#define usetable usetable_AmpaNmda
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_AmpaNmda", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Cdur_AmpaNmda", "ms",
 "Alpha_AmpaNmda", "/ms",
 "Beta_AmpaNmda", "/ms",
 "E_AmpaNmda", "mV",
 "gampafactor_AmpaNmda", "1",
 "nmdafactor_AmpaNmda", "1",
 "ltdinvl_AmpaNmda", "ms",
 "ltpinvl_AmpaNmda", "ms",
 "sighalf_AmpaNmda", "1",
 "sigslope_AmpaNmda", "1",
 "ampatau_AmpaNmda", "ms",
 "Rtau_AmpaNmda", "ms",
 "mg", "mM",
 "gmax", "umho",
 "x", "um",
 "gampa", "umho",
 "i", "nA",
 "inmda", "nA",
 "iampa", "nA",
 "gnmda", "umho",
 0,0
};
 static double Roff0 = 0;
 static double Ron0 = 0;
 static double delta_t = 0.01;
 static double gampa0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Cdur_AmpaNmda", &Cdur_AmpaNmda,
 "Alpha_AmpaNmda", &Alpha_AmpaNmda,
 "Beta_AmpaNmda", &Beta_AmpaNmda,
 "E_AmpaNmda", &E_AmpaNmda,
 "gampafactor_AmpaNmda", &gampafactor_AmpaNmda,
 "nmdafactor_AmpaNmda", &nmdafactor_AmpaNmda,
 "ltdinvl_AmpaNmda", &ltdinvl_AmpaNmda,
 "ltpinvl_AmpaNmda", &ltpinvl_AmpaNmda,
 "sighalf_AmpaNmda", &sighalf_AmpaNmda,
 "sigslope_AmpaNmda", &sigslope_AmpaNmda,
 "ampatau_AmpaNmda", &ampatau_AmpaNmda,
 "Rinf_AmpaNmda", &Rinf_AmpaNmda,
 "Rtau_AmpaNmda", &Rtau_AmpaNmda,
 "usetable_AmpaNmda", &usetable_AmpaNmda,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"AmpaNmda",
 "mg",
 "gmax",
 "x",
 "mgid",
 "ggid",
 "srcgid",
 0,
 "i",
 "inmda",
 "iampa",
 "gnmda",
 0,
 "Ron",
 "Roff",
 "gampa",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 20, _prop);
 	/*initialize range parameters*/
 	mg = 1;
 	gmax = 2;
 	x = 0;
 	mgid = -1;
 	ggid = -1;
 	srcgid = -1;
  }
 	_prop->param = _p;
 	_prop->param_size = 20;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ampanmda_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 20, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 6;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 AmpaNmda /home/mimo/Desktop/NeuronWithNetworkx/mod/ampanmda.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_mgblock;
static int _reset;
static char *modelname = "simple NMDA receptors";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static double _n_mgblock(_threadargsprotocomma_ double _lv);
 static int _slist1[3], _dlist1[3];
 static int release(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DRon = ( synon * Rinf - Ron ) / Rtau ;
   DRoff = - Beta * Roff ;
   Dgampa = - gampa / ampatau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DRon = DRon  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Rtau )) ;
 DRoff = DRoff  / (1. - dt*( ( - Beta )*( 1.0 ) )) ;
 Dgampa = Dgampa  / (1. - dt*( ( - 1.0 ) / ampatau )) ;
  return 0;
}
 /*END CVODE*/
 static int release (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    Ron = Ron + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Rtau)))*(- ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - Ron) ;
    Roff = Roff + (1. - exp(dt*(( - Beta )*( 1.0 ))))*(- ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - Roff) ;
    gampa = gampa + (1. - exp(dt*(( - 1.0 ) / ampatau)))*(- ( 0.0 ) / ( ( - 1.0 ) / ampatau ) - gampa) ;
   }
  return 0;
}
 static double _mfac_mgblock, _tmin_mgblock;
  static void _check_mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_mg;
  if (!usetable) {return;}
  if (_sav_mg != mg) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_mgblock =  - 140.0 ;
   _tmax =  80.0 ;
   _dx = (_tmax - _tmin_mgblock)/1000.; _mfac_mgblock = 1./_dx;
   for (_i=0, _x=_tmin_mgblock; _i < 1001; _x += _dx, _i++) {
    _t_mgblock[_i] = _f_mgblock(_p, _ppvar, _thread, _nt, _x);
   }
   _sav_mg = mg;
  }
 }

 double mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 return _n_mgblock(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_mgblock(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_mgblock * (_lv - _tmin_mgblock);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_mgblock[0];
 }
 if (_xi >= 1000.) {
 return _t_mgblock[1000];
 }
 _i = (int) _xi;
 return _t_mgblock[_i] + (_xi - (double)_i)*(_t_mgblock[_i+1] - _t_mgblock[_i]);
 }

 
double _f_mgblock ( _threadargsprotocomma_ double _lv ) {
   double _lmgblock;
 _lmgblock = 1.0 / ( 1.0 + exp ( 0.062 * - _lv ) * ( mg / 3.57 ) ) ;
   
return _lmgblock;
 }
 
static double _hoc_mgblock(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 
#if 1
 _check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 _r =  mgblock ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double plast ( _threadargsprotocomma_ double _lstep ) {
   double _lplast;
 _lplast = 1.0 - 1.0 / ( 1.0 + exp ( ( _lstep - sighalf ) / sigslope ) ) ;
   
return _lplast;
 }
 
static double _hoc_plast(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  plast ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double norm_weight_to_sig ( _threadargsprotocomma_ double _lw ) {
   double _lnorm_weight_to_sig;
 _lnorm_weight_to_sig = floor ( 0.4999 + log ( ( ( - 1.0 / ( _lw - 1.0 ) ) - 1.0 ) / exp ( - sighalf / sigslope ) ) * sigslope ) ;
   
return _lnorm_weight_to_sig;
 }
 
static double _hoc_norm_weight_to_sig(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  norm_weight_to_sig ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 0.0 ) {
     if ( t - _args[3] < ltpinvl ) {
       _args[1] = _args[1] + 1.0 ;
       if ( _args[1] > 2.0 * sighalf ) {
         _args[1] = 2.0 * sighalf ;
         }
       }
     else if ( t - _args[3] > ltdinvl ) {
       }
     else {
       _args[1] = _args[1] - 1.0 ;
       if ( _args[1] < 0.0 ) {
         _args[1] = 0.0 ;
         }
       }
     _args[3] = t ;
     _args[2] = _args[0] * plast ( _threadargscomma_ _args[1] ) ;
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = gampa;
    double __primary = (gampa + _args[2] * gmax * gampafactor) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / ampatau ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / ampatau ) - __primary );
    gampa += __primary;
  } else {
 gampa = gampa + _args[2] * gmax * gampafactor ;
       }
 _args[4] = _args[4] * exp ( - Beta * ( t - _args[5] ) ) ;
     _args[5] = t ;
     synon = synon + _args[2] ;
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron + _args[4]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron + _args[4] ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff - _args[4]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff - _args[4] ;
       }
 net_send ( _tqitem, _args, _pnt, t +  Cdur , _args[2] + 1.0 ) ;
     }
   else {
     _args[4] = ( _lflag - 1.0 ) * Rinf + ( _args[4] - ( _lflag - 1.0 ) * Rinf ) * exp ( - ( t - _args[5] ) / Rtau ) ;
     _args[5] = t ;
     synon = synon - ( _lflag - 1.0 ) ;
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron - _args[4]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron - _args[4] ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff + _args[4]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff + _args[4] ;
       }
 }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    NrnThread* _nt = (NrnThread*)_pnt->_vnt;
 _args[1] = 0.0 ;
   _args[2] = 0.0 ;
   _args[3] = - 1e9 ;
   _args[4] = 0.0 ;
   _args[5] = - 1e9 ;
   }
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  Roff = Roff0;
  Ron = Ron0;
  gampa = gampa0;
 {
   /* PROTECT */_NMODLMUTEXLOCK
 Rinf = Alpha / ( Alpha + Beta ) ;
   
 _NMODLMUTEXUNLOCK /* end PROTECT */
 /* PROTECT */_NMODLMUTEXLOCK
 Rtau = 1.0 / ( Alpha + Beta ) ;
   
 _NMODLMUTEXUNLOCK /* end PROTECT */
 synon = 0.0 ;
   gampa = 0.0 ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];

#if 0
 _check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gnmda = mgblock ( _threadargscomma_ v ) * ( Ron + Roff ) * gmax * nmdafactor ;
   inmda = gnmda * ( v - E ) ;
   iampa = gampa * ( v - E ) ;
   i = iampa + inmda ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {   release(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = Ron_columnindex;  _dlist1[0] = DRon_columnindex;
 _slist1[1] = Roff_columnindex;  _dlist1[1] = DRoff_columnindex;
 _slist1[2] = gampa_columnindex;  _dlist1[2] = Dgampa_columnindex;
   _t_mgblock = makevector(1001*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/mimo/Desktop/NeuronWithNetworkx/mod/ampanmda.mod";
static const char* nmodl_file_text = 
  "TITLE simple NMDA receptors\n"
  "\n"
  ": Hines combined AMPA and NMDA and spike dependent plasticity\n"
  "\n"
  ": Modified from the original AMPA.mod, M.Migliore Jan 2003\n"
  ": A weight of 0.0035 gives a peak conductance of 1nS in 0Mg\n"
  "\n"
  "COMMENT\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "	Simple model for glutamate AMPA receptors\n"
  "	=========================================\n"
  "\n"
  "  - FIRST-ORDER KINETICS, FIT TO WHOLE-CELL RECORDINGS\n"
  "\n"
  "    Whole-cell recorded postsynaptic currents mediated by AMPA/Kainate\n"
  "    receptors (Xiang et al., J. Neurophysiol. 71: 2552-2556, 1994) were used\n"
  "    to estimate the parameters of the present model; the fit was performed\n"
  "    using a simplex algorithm (see Destexhe et al., J. Computational Neurosci.\n"
  "    1: 195-230, 1994).\n"
  "\n"
  "  - SHORT PULSES OF TRANSMITTER (0.3 ms, 0.5 mM)\n"
  "\n"
  "    The simplified model was obtained from a detailed synaptic model that \n"
  "    included the release of transmitter in adjacent terminals, its lateral \n"
  "    diffusion and uptake, and its binding on postsynaptic receptors (Destexhe\n"
  "    and Sejnowski, 1995).  Short pulses of transmitter with first-order\n"
  "    kinetics were found to be the best fast alternative to represent the more\n"
  "    detailed models.\n"
  "\n"
  "  - ANALYTIC EXPRESSION\n"
  "\n"
  "    The first-order model can be solved analytically, leading to a very fast\n"
  "    mechanism for simulating synapses, since no differential equation must be\n"
  "    solved (see references below).\n"
  "\n"
  "\n"
  "\n"
  "References\n"
  "\n"
  "   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  An efficient method for\n"
  "   computing synaptic conductances based on a kinetic model of receptor binding\n"
  "   Neural Computation 6: 10-14, 1994.  \n"
  "\n"
  "   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for\n"
  "   excitable membranes, synaptic transmission and neuromodulation using a \n"
  "   common kinetic formalism, Journal of Computational Neuroscience 1: \n"
  "   195-230, 1994.\n"
  "\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS AmpaNmda\n"
  "	RANGE R, g, mg, inmda, iampa, gnmda, gampa\n"
  "	RANGE x, mgid, ggid, srcgid, gmax\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	GLOBAL Cdur, Alpha, Beta, E, Rinf, Rtau, ampatau\n"
  "	GLOBAL gampafactor, nmdafactor\n"
  "	GLOBAL ltdinvl, ltpinvl, sighalf, sigslope\n"
  "}\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	Cdur	= 1		(ms)	: transmitter duration (rising phase)\n"
  "	Alpha	= 0.35		(/ms)	: forward (binding) rate\n"
  "	Beta	= 0.035		(/ms)	: backward (unbinding) rate\n"
  "	E	= 0	(mV)		: reversal potential\n"
  "	mg	= 1    (mM)		: external magnesium concentration\n"
  "	gmax = 2 (umho)\n"
  "	gampafactor = 0.001 (1)\n"
  "	nmdafactor = 0.0035 (1)\n"
  "	ltdinvl = 250 (ms)		: longer intervals, no change\n"
  "	ltpinvl = 33.33 (ms)		: shorter interval, LTP\n"
  "	sighalf = 25 (1)\n"
  "	sigslope = 3 (1)\n"
  "	ampatau = 3 (ms)\n"
  "	x = 0 (um) : cartesian synapse location\n"
  "	mgid = -1 : associated mitral gid\n"
  "	ggid = -1 : associated granule gid\n"
  "	srcgid = -1 : gid of the mitral detector\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v		(mV)		: postsynaptic voltage\n"
  "	i 		(nA)		: total current = iampa+inmda\n"
  "	inmda 		(nA)		: current = gnmda*(v - E)\n"
  "	iampa 		(nA)		: current = gampa*(v - E)\n"
  "	gnmda 		(umho)		: \n"
  "	Rinf				: steady state channels open\n"
  "	Rtau		(ms)		: time constant of channel binding\n"
  "	synon\n"
  "}\n"
  "\n"
  "STATE {Ron Roff\n"
  "	gampa 		(umho)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	PROTECT Rinf = Alpha / (Alpha + Beta)\n"
  "	PROTECT Rtau = 1 / (Alpha + Beta)\n"
  "	synon = 0\n"
  "	gampa = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE release METHOD cnexp\n"
  "	gnmda = mgblock(v)*(Ron + Roff)*gmax*nmdafactor\n"
  "	inmda = gnmda*(v - E)\n"
  "	iampa = gampa*(v - E)\n"
  "	i = iampa + inmda\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "	Ron' = (synon*Rinf - Ron)/Rtau\n"
  "	Roff' = -Beta*Roff\n"
  "	gampa' = -gampa/ampatau\n"
  "}\n"
  "\n"
  ": following supports both saturation from single input and\n"
  ": summation from multiple inputs\n"
  ": if spike occurs during CDur then new off time is t + CDur\n"
  ": ie. transmitter concatenates but does not summate\n"
  ": Note: automatic initialization of all reference args to 0 except first\n"
  "\n"
  "\n"
  "FUNCTION mgblock(v(mV)) {\n"
  "	TABLE \n"
  "	DEPEND mg\n"
  "	FROM -140 TO 80 WITH 1000\n"
  "\n"
  "	: from Jahr & Stevens\n"
  "\n"
  "	mgblock = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))\n"
  "}\n"
  "\n"
  "FUNCTION plast(step(1))(1) {\n"
  "	plast = 1 - 1/(1 + exp((step - sighalf)/sigslope))\n"
  "}\n"
  "\n"
  "FUNCTION norm_weight_to_sig(w) {\n"
  "    norm_weight_to_sig = floor(0.4999 + log(((-1/(w-1))-1)/exp(-sighalf/sigslope))*sigslope)\n"
  "}\n"
  "\n"
  "NET_RECEIVE(weight, s, w, tlast (ms), r0, t0 (ms)) {\n"
  "	INITIAL {\n"
  "		s = 0\n"
  "		w = 0\n"
  "		tlast = -1e9 (ms)\n"
  "		r0 = 0\n"
  "		t0 = -1e9 (ms)\n"
  "	}\n"
  "	: flag is an implicit argument of NET_RECEIVE and  normally 0\n"
  "        if (flag == 0) { : a spike, so turn on if not already in a Cdur pulse\n"
  "		: plasticity affects this spike. If desired to affect\n"
  "		: the next spike then put following group after\n"
  "		: net_send\n"
  "		if (t - tlast < ltpinvl) { : LTP\n"
  "			s = s + 1\n"
  "			if (s > 2*sighalf) { s = 2*sighalf }\n"
  "		}else if (t - tlast > ltdinvl) { : no change\n"
  "		}else{ : LTD\n"
  "			s = s - 1\n"
  "			if (s < 0) { s = 0 }\n"
  "		}\n"
  "		tlast = t\n"
  "\n"
  "		w = weight*plast(s)\n"
  "		gampa = gampa + w*gmax*gampafactor\n"
  "		r0 = r0*exp(-Beta*(t - t0))\n"
  "		t0 = t\n"
  "		synon = synon + w\n"
  "		Ron = Ron + r0\n"
  "		Roff = Roff - r0\n"
  "		: come again in Cdur with flag = current value of w+1\n"
  "		net_send(Cdur, w + 1)\n"
  "        }else{ : turn off what was added Cdur ago\n"
  "		r0 = (flag-1)*Rinf + (r0 - (flag-1)*Rinf)*exp(-(t - t0)/Rtau)\n"
  "		t0 = t\n"
  "		synon = synon - (flag-1)\n"
  "		Ron = Ron - r0\n"
  "		Roff = Roff + r0\n"
  "	}\n"
  "}\n"
  "\n"
  ;
#endif
