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
#ifndef QMCPLUSPLUS_DUMMY_H
#define QMCPLUSPLUS_DUMMY_H
#include "QMCDrivers/QMCDriver.h"
namespace qmcplusplus
{

/** @ingroup QMCDrivers
 *@brief A dummy QMCDriver for testing
 */
class DummyQMC: public QMCDriver
{
public:
  /// Constructor.
  DummyQMC(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h);

  bool run();
  bool put(xmlNodePtr cur);

private:
  /// Copy Constructor (disabled)
  DummyQMC(const DummyQMC& a): QMCDriver(a) { }
  /// Copy operator (disabled).
  DummyQMC& operator=(const DummyQMC&)
  {
    return *this;
  }

};
}

#endif
/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: DummyQMC.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
