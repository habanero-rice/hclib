//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
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
#ifndef QMCPLUSPLUS_ORIGINAL_JASTROW_AB_BUILDER_H
#define QMCPLUSPLUS_ORIGINAL_JASTROW_AB_BUILDER_H
#include "QMCWaveFunctions/OrbitalBuilderBase.h"

namespace qmcplusplus
{

//forward declaration
class ParticleSet;

/** Generic two-body Jastrow builder
 *
 * Replacement of JastrowBuilder::createTwoBodySpin and JastrowBuilder::createTwoBodyNoSpin
 */
struct JABBuilder: public OrbitalBuilderBase
{

  JABBuilder(ParticleSet& p, TrialWaveFunction& psi,
             PtclPoolType& psets):OrbitalBuilderBase(p,psi), ptclPool(psets) {}

  bool put(xmlNodePtr cur);

  template<class FN> bool createJAB(xmlNodePtr cur, const string& jname);


  PtclPoolType& ptclPool;
};

}
#endif
/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: JABBuilder.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
