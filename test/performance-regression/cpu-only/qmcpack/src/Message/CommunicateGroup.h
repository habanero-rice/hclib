//////////////////////////////////////////////////////////////////
// (c) Copyright 2006- by Jeongnim Kim
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

#ifndef QMCPACK_COMMUNICATIONGROUP_H
#define QMCPACK_COMMUNICATIONGROUP_H

#include "Message/Communicate.h"

struct CommunicateGroup: public Communicate
{
  ///parent communicator
  Communicate& parent;
  ///constructor
  CommunicateGroup(Communicate& acomm, int ndiv=1);
};
#endif //QMCPACK_COMMUNICATIONGROUP_H 
/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: CommunicateGroup.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
