//////////////////////////////////////////////////////////////////
// (c) Copyright 2003-  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
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
/*!\file sysutil.cpp
 * Implement function to get system information in Unix environment.
 */

#include <string>
#include <sstream>
#include <iostream>
using std::string;
#include <time.h>

#if defined(_CRAYMPI) || defined(XT_CATAMOUNT)
string getUserName()
{
  return "auser";
}
string getHostName()
{
  return "jaguar";
}
#else
#include <unistd.h>
#include <sys/utsname.h>
#include <pwd.h>

string getUserName()
{
  struct passwd *who;
  if((who = getpwuid(getuid())) != NULL)
  {
    return who->pw_name;
  }
  return "auser";
}

string getHostName()
{
  utsname mysys;
  uname(&mysys);
  return string(mysys.nodename);
}
#endif
string getDateAndTime()
{
  time_t now;
  time(&now);
  return ctime(&now);
}

string getDateAndTime(const char* format)
{
  time_t now;
  time(&now);
  tm* now_c = localtime(&now);
  char d[32];
  strftime(d,32,format,now_c);
  return string(d);
}

/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: sysutil.cpp 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
