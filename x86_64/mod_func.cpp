#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTg_reg(void);
extern void _NaTs2_t_reg(void);
extern void _Nap_Et2_reg(void);
extern void _ProbAMPA_reg(void);
extern void _ProbAMPANMDA2_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);
extern void _ampanmda_reg(void);
extern void _epsp_reg(void);
extern void _int2pyr_reg(void);
extern void _netgaba_reg(void);
extern void _netglutamate_reg(void);
extern void _pyr2pyr_reg(void);
extern void _vecevent_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"./CaDynamics_E2.mod\"");
    fprintf(stderr, " \"./Ca_HVA.mod\"");
    fprintf(stderr, " \"./Ca_LVAst.mod\"");
    fprintf(stderr, " \"./Ih.mod\"");
    fprintf(stderr, " \"./Im.mod\"");
    fprintf(stderr, " \"./K_Pst.mod\"");
    fprintf(stderr, " \"./K_Tst.mod\"");
    fprintf(stderr, " \"./NaTa_t.mod\"");
    fprintf(stderr, " \"./NaTg.mod\"");
    fprintf(stderr, " \"./NaTs2_t.mod\"");
    fprintf(stderr, " \"./Nap_Et2.mod\"");
    fprintf(stderr, " \"./ProbAMPA.mod\"");
    fprintf(stderr, " \"./ProbAMPANMDA2.mod\"");
    fprintf(stderr, " \"./SK_E2.mod\"");
    fprintf(stderr, " \"./SKv3_1.mod\"");
    fprintf(stderr, " \"./ampanmda.mod\"");
    fprintf(stderr, " \"./epsp.mod\"");
    fprintf(stderr, " \"./int2pyr.mod\"");
    fprintf(stderr, " \"./netgaba.mod\"");
    fprintf(stderr, " \"./netglutamate.mod\"");
    fprintf(stderr, " \"./pyr2pyr.mod\"");
    fprintf(stderr, " \"./vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _Ih_reg();
  _Im_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _NaTa_t_reg();
  _NaTg_reg();
  _NaTs2_t_reg();
  _Nap_Et2_reg();
  _ProbAMPA_reg();
  _ProbAMPANMDA2_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
  _ampanmda_reg();
  _epsp_reg();
  _int2pyr_reg();
  _netgaba_reg();
  _netglutamate_reg();
  _pyr2pyr_reg();
  _vecevent_reg();
}

#if defined(__cplusplus)
}
#endif
