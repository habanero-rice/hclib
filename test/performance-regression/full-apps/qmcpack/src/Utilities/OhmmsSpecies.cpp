//////////////////////////////////////////////////////////////////
// (c) Copyright 1998-2002 by Jeongnim Kim
//
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

#include "Utilities/OhmmsSpecies.h"

SpeciesBase::SpeciesBase()
{
  TotalNum = 0;
  Name.reserve(10); // expect less than 10 species
}

SpeciesBase::~SpeciesBase()
{
  AttribList_t::iterator dit = d_attrib.begin();
  for(; dit != d_attrib.end(); ++dit)
    delete (*dit);
}

int SpeciesBase::addAttrib()
{
  int n = d_attrib.size();
  d_attrib.push_back(new SpeciesAttrib_t(TotalNum));
  return n;
}

/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: OhmmsSpecies.cpp 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/


/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: OhmmsSpecies.cpp 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/

