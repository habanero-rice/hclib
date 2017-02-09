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

#ifndef OHMMS_TAG_MAKER_H
#define OHMMS_TAG_MAKER_H
/*!\class TagMaker
 * \brief Assign a unique tag whenver TagMaker::TagMaker() is called.
 */
class TagMaker
{
public:

  TagMaker()
  {
    MyTag = (++CurrentTag);
  }
  ~TagMaker() {}
  int operator()()const
  {
    return MyTag;
  }
private:
  int MyTag;
  static int CurrentTag;
};
#endif // OHMMS_TAG_MAKER_H

/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: TagMaker.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
