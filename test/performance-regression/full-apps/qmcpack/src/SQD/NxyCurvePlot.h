#ifndef OHMMS_SQD_NXYCURVEPLOT_H_
#define OHMMS_SQD_NXYCURVEPLOT_H_

#include <qwt_plot.h>
#include <map>

class QwtLegend;

/**class to plot multiple 2D plots sharing a common x axis
 *@brief provide functionality of xmgrace -nxy
 */
class NxyCurvePlot: public QwtPlot
{

  Q_OBJECT

public:

  NxyCurvePlot(QWidget *parent, const char* aname=NULL);

  virtual QSize sizeHint() const;

  virtual void appendCurveData(long curveId,
                               double *x,
                               double *y,
                               int size);

public slots:

  void clear();

  void modifyCurves(const QMouseEvent& e);

private:

  void initCurve();
  QwtLegend *d_legend_in;
  static std::map<long,QColor> ColorList;
};

#endif /*_RANDOMPLOT_H_*/

/***************************************************************************
 * $RCSfile$   $Author: jmcminis $
 * $Revision: 5794 $   $Date: 2013-04-25 20:14:53 -0400 (Thu, 25 Apr 2013) $
 * $Id: NxyCurvePlot.h 5794 2013-04-26 00:14:53Z jmcminis $
 ***************************************************************************/
