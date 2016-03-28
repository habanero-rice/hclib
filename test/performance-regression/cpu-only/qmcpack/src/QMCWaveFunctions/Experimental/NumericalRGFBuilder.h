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
#ifndef QMCPLUSPLUS_NUMERICALRADIALGRIDFUNCTOR_H
#define QMCPLUSPLUS_NUMERICALRADIALGRIDFUNCTOR_H

#include "QMCWaveFunctions/MolecularOrbitals/RGFBuilderBase.h"

namespace qmcplusplus
{

/**Class to create a set of radial orbitals on a grid (e.g., AtomHF/Siesta)
 *
 * The grid and orbitals are stored in HDF5 format.
 */
struct NumericalRGFBuilder: public RGFBuilderBase
{
  ///constructor
  NumericalRGFBuilder(xmlNodePtr cur);
  bool putCommon(xmlNodePtr cur);
  bool addGrid(xmlNodePtr cur);
  bool addRadialOrbital(xmlNodePtr cur, const QuantumNumberType& nlms);

};

}
#endif
/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: NumericalRGFBuilder.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
