//////////////////////////////////////////////////////////////////
// (c) Copyright 2005- by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)
//
// Supported by
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_COSTFUNCTIONSINGLE_H
#define QMCPLUSPLUS_COSTFUNCTIONSINGLE_H

#include "QMCDrivers/QMCCostFunctionBase.h"

namespace qmcplusplus
{

/** @ingroup QMCDrivers
 * @brief Implements wave-function optimization
 *
 * Optimization by correlated sampling method with configurations
 * generated from VMC running on a single thread.
 */
class QMCCostFunctionSingle: public QMCCostFunctionBase
{
public:

  ///Constructor.
  QMCCostFunctionSingle(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h);

  ///Destructor
  ~QMCCostFunctionSingle();

  void getConfigurations(const string& aroot);
  void checkConfigurations();
  void GradCost(vector<QMCTraits::RealType>& PGradient, const vector<QMCTraits::RealType>& PM, QMCTraits::RealType FiniteDiff=0) ;
  Return_t fillOverlapHamiltonianMatrices(Matrix<Return_t>& H2, Matrix<Return_t>& Hamiltonian, Matrix<Return_t>& Variance, Matrix<Return_t>& Overlap);
protected:
  vector<vector<Return_t> > TempDerivRecords;
  vector<vector<Return_t> > TempHDerivRecords;
  Return_t CSWeight;
  void resetPsi(bool final_reset=false);
  Return_t correlatedSampling(bool needGrad=true);
};
}
#endif
/***************************************************************************
 * $RCSfile$   $Author: jnkim $
 * $Revision: 5807 $   $Date: 2013-04-30 14:42:29 -0400 (Tue, 30 Apr 2013) $
 * $Id: QMCCostFunctionSingle.h 5807 2013-04-30 18:42:29Z jnkim $
 ***************************************************************************/
