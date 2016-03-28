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
#ifndef QMCPLUSPLUS_NUMERICAL_JASTROW_AB_BUILDER_H
#define QMCPLUSPLUS_NUMERICAL_JASTROW_AB_BUILDER_H
#include "QMCWaveFunctions/OrbitalBuilderBase.h"
#include "QMCWaveFunctions/Jastrow/NumericalJastrowFunctor.h"

namespace qmcplusplus
{

//forward declaration
class ParticleSet;

/**@ingroup WFSBuilder
 *A builder class to add a numerical two-body Jastrow function to a TrialWaveFunction
 *
 *A xml node with OrbtialBuilderBase::jastrow_tag is parsed recursively.
 */
struct NJABBuilder: public OrbitalBuilderBase
{

  typedef NumericalJastrow<RealType> FuncType;
  typedef FuncType::FNIN              InFuncType;
  typedef FuncType::FNOUT             OutFuncType;

  ///reference to the pool
  PtclPoolType& ptclPool;
  ///pointer to the center ParticleSet
  ParticleSet* sourcePtcl;
  ///pointer to <grid/>
  xmlNodePtr gridPtr;
  ///unique analytic functions
  vector<InFuncType*> InFunc;

  NJABBuilder(ParticleSet& p, TrialWaveFunction& psi, PtclPoolType& psets);

  /**@param cur the current xmlNodePtr to be processed by NumericalJastrowBuilder
   *@return true if succesful
   */
  bool put(xmlNodePtr cur);

  /** build UniqueIn and InFunc
   *
   * Initialize the input analytic functions
   */
  bool putInFunc(xmlNodePtr cur);

  InFuncType* createInFunc(const string& jastfunction);

};
}
#endif
/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: NJABBuilder.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
