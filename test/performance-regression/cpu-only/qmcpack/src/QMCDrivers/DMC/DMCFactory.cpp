//////////////////////////////////////////////////////////////////
// (c) Copyright 2003- by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//
// Supported by
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#include "QMCDrivers/DMC/DMCFactory.h"
#include "QMCDrivers/DMC/DMCOMP.h"
#include "Message/OpenMP.h"

#ifdef QMC_CUDA
#include "QMCDrivers/DMC/DMC_CUDA.h"
#endif

//#define PETA_DMC_TEST
namespace qmcplusplus
{

QMCDriver* DMCFactory::create(MCWalkerConfiguration& w, TrialWaveFunction& psi
                              , QMCHamiltonian& h, HamiltonianPool& hpool,WaveFunctionPool& ppool)
{
#ifdef QMC_CUDA
  if (GPU)
    return new DMCcuda (w, psi, h,ppool);
#endif
  app_log() << "Creating DMCMP for the qmc driver" << endl;
  QMCDriver*  qmc = new DMCOMP(w,psi,h,hpool,ppool);
  qmc->setUpdateMode(PbyPUpdate);
  return qmc;
}
}
/***************************************************************************
 * $RCSfile: DMCFactory.cpp,v $   $Author: jnkim $
 * $Revision: 6035 $   $Date: 2013-10-28 10:34:33 -0400 (Mon, 28 Oct 2013) $
 * $Id: DMCFactory.cpp 6035 2013-10-28 14:34:33Z jnkim $
 ***************************************************************************/
