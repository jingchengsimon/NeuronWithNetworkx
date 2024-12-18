#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _ampanmda_reg(void);
extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _epsp_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _int2pyr_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _Nap_Et2_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTg_reg(void);
extern void _NaTs2_t_reg(void);
extern void _netgaba_reg(void);
extern void _netglutamate_reg(void);
extern void _ProbAMPA_reg(void);
extern void _ProbAMPANMDA2_reg(void);
extern void _pyr2pyr_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);
extern void _vecevent_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"./mod//ampanmda.mod\"");
    fprintf(stderr, " \"./mod//CaDynamics_E2.mod\"");
    fprintf(stderr, " \"./mod//Ca_HVA.mod\"");
    fprintf(stderr, " \"./mod//Ca_LVAst.mod\"");
    fprintf(stderr, " \"./mod//epsp.mod\"");
    fprintf(stderr, " \"./mod//Ih.mod\"");
    fprintf(stderr, " \"./mod//Im.mod\"");
    fprintf(stderr, " \"./mod//int2pyr.mod\"");
    fprintf(stderr, " \"./mod//K_Pst.mod\"");
    fprintf(stderr, " \"./mod//K_Tst.mod\"");
    fprintf(stderr, " \"./mod//Nap_Et2.mod\"");
    fprintf(stderr, " \"./mod//NaTa_t.mod\"");
    fprintf(stderr, " \"./mod//NaTg.mod\"");
    fprintf(stderr, " \"./mod//NaTs2_t.mod\"");
    fprintf(stderr, " \"./mod//netgaba.mod\"");
    fprintf(stderr, " \"./mod//netglutamate.mod\"");
    fprintf(stderr, " \"./mod//ProbAMPA.mod\"");
    fprintf(stderr, " \"./mod//ProbAMPANMDA2.mod\"");
    fprintf(stderr, " \"./mod//pyr2pyr.mod\"");
    fprintf(stderr, " \"./mod//SK_E2.mod\"");
    fprintf(stderr, " \"./mod//SKv3_1.mod\"");
    fprintf(stderr, " \"./mod//vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _ampanmda_reg();
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _epsp_reg();
  _Ih_reg();
  _Im_reg();
  _int2pyr_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _Nap_Et2_reg();
  _NaTa_t_reg();
  _NaTg_reg();
  _NaTs2_t_reg();
  _netgaba_reg();
  _netglutamate_reg();
  _ProbAMPA_reg();
  _ProbAMPANMDA2_reg();
  _pyr2pyr_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
  _vecevent_reg();
}

#if defined(__cplusplus)
}
#endif
