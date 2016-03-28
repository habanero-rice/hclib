//////////////////////////////////////////////////////////////////
// (c) Copyright 2003  by Jeongnim Kim
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
#ifndef QMCPLUSPLUS_STO2RADIALGRIDFUNCTOR_H
#define QMCPLUSPLUS_STO2RADIALGRIDFUNCTOR_H

#include "QMCWaveFunctions/MolecularOrbitals/RGFBuilderBase.h"

namespace qmcplusplus
{

/**Class to convert SlaterTypeOrbital to a radial orbital on a log grid.
 *
 * For a center,
 *   - only one grid is used
 *   - any number of radial orbitals
 */
struct STO2GridBuilder: public RGFBuilderBase
{

  ///constructor
  STO2GridBuilder() {}
  bool addRadialOrbital(xmlNodePtr cur, const QuantumNumberType& nlms);
  bool putCommon(xmlNodePtr cur);
};

}
#endif
/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: STO2GridBuilder.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
