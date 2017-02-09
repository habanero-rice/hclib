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
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)
//
// Supported by
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//   Department of Physics, Ohio State University
//   Ohio Supercomputer Center
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_DMC_FACTORY_H
#define QMCPLUSPLUS_DMC_FACTORY_H
#include "QMCDrivers/QMCDriver.h"
#include "QMCApp/HamiltonianPool.h"

namespace qmcplusplus
{
struct DMCFactory
{
  bool PbyPUpdate, GPU;
  xmlNodePtr myNode;
  DMCFactory(bool pbyp, bool gpu, xmlNodePtr cur) :
    PbyPUpdate(pbyp), myNode(cur), GPU(gpu)
  { }

  QMCDriver* create(MCWalkerConfiguration& w,
                    TrialWaveFunction& psi,
                    QMCHamiltonian& h, HamiltonianPool& hpool,WaveFunctionPool& ppool);
};
}

#endif
/***************************************************************************
 * $RCSfile: DMCFactory.h,v $   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: DMCFactory.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
